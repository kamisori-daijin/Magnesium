import torch
import torch.nn as nn

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=64, shadow_steps=16):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.shadow_steps = shadow_steps
        self.dt = 0.08
        
        # 4D Tensor
        self.register_buffer("eps", torch.tensor([[[[0.02]]]]).half())
        self.register_buffer("floor_y", torch.tensor([[[[-0.8]]]]).half())
        
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        self.register_buffer("light_dx", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dy", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dz", torch.tensor([[[[0.5773]]]]).half())

        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(self.shadow_steps).view(1, self.shadow_steps, 1, 1) * self.dt).half())
        self.register_buffer("voxel_data_64", torch.randint(0, 2, (1, 64, 64, 64)).half()) # [1, C=64, H=64, W=64]
        self.register_buffer("voxel_indices", torch.linspace(-1.0, 1.0, 64).view(1, 64, 1, 1).half()) #
        y_tex = torch.linspace(-1.0, 1.0, self.h).view(1, 1, self.h, 1)
        x_tex = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        base_mask_x = torch.clamp(1.0 - (torch.abs(x_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        base_mask_y = torch.clamp(1.0 - (torch.abs(y_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        cube_2d_mask = (base_mask_x * base_mask_y).half()

        self.register_buffer("base_multiview_textures", torch.cat([cube_2d_mask, cube_2d_mask, cube_2d_mask], dim=1))

        # Define Conv2d for ANE cumsum
        self.ane_cumsum_conv = nn.Conv2d(self.max_steps, self.max_steps, kernel_size=1, bias=False)
        weight_matrix = torch.tril(torch.ones(self.max_steps, self.max_steps))
        self.ane_cumsum_conv.weight.data = weight_matrix.view(self.max_steps, self.max_steps, 1, 1).half()
        self.ane_cumsum_conv.weight.requires_grad = False

    def check_multiview_hit(self, px, py, pz):
        out_x = torch.relu(torch.abs(px) - 1.0)
        out_y = torch.relu(torch.abs(py) - 1.0)
        out_z = torch.relu(torch.abs(pz) - 1.0)
        any_out = torch.clamp((out_x + out_y + out_z) * 100.0, min=0.0, max=1.0)
        box_check = 1.0 - any_out

        proj_xy = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0)
        proj_xz = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)
        proj_yz = torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)

        mask_xy = self.base_multiview_textures[:, 0:1, :, :] * proj_xy
        mask_xz = self.base_multiview_textures[:, 1:2, :, :] * proj_xz
        mask_yz = self.base_multiview_textures[:, 2:3, :, :] * proj_yz

        return mask_xy * mask_xz * mask_yz * box_check
    def check_voxel_hit(self, px, py, pz, voxel_data_64):
        # 1マスあたりの半幅 (2.0 / 64 / 2 = 約0.0156)
        half_w = 0.0156

        # 💡 すべてのサンプリング基準を [1, 64, 1, 1]（チャネル方向）に完全に固定！
        # これにより、レイの持つ [1, max_steps(64), 256, 256] という形状の
        # H(256) や W(256) の次元と完璧にブロードキャスト（自動拡張）が噛み合います。
        grid_bins_shared = self.voxel_indices.view(1, 64, 1, 1)

        # 全ての軸のゲート判定を、完全にチャネル方向（dim=1のC=64）で一斉評価
        z_gate = torch.clamp((half_w - torch.abs(pz - grid_bins_shared)) * 100000.0, min=0.0, max=1.0)
        y_gate = torch.clamp((half_w - torch.abs(py - grid_bins_shared)) * 100000.0, min=0.0, max=1.0)
        x_gate = torch.clamp((half_w - torch.abs(px - grid_bins_shared)) * 100000.0, min=0.0, max=1.0)

        # 💡 ボクセルデータ（4Dアトラス）との一斉積和
        # voxel_data_64 の形状は [1, 64, 64, 64] です。
        # x_gate, y_gate, z_gate はすべて [1, 64, 256, 256] に自動拡張されます。
        # ANEの上で [1, 64, 256, 256] 同士の要素ごとの掛け算（Hadamard product）として超並列に処理されます。
        sampled_plane = voxel_data_64 * y_gate * x_gate
            # 最後にZ軸のゲートを掛け合わせて、チャネル方向（dim=1のC=64）を足し算（sum）で潰す
        # これで、4次元のまま、末尾のH(256)やW(256)を1ミリも破壊せずに一撃ルックアップが完了します！
        final_hit = torch.sum(sampled_plane * z_gate, dim=1, keepdim=True).clamp(0.0, 1.0)

        return final_hit


 
    def fast_rsqrt(self, x):
        # Initial guess for inverse square root
        y = 1.0

        # Newton's method step repeated 2-3 times
        # y_new = y * (1.5 - 0.5 * x * y * y)
        y = y * (1.5 - 0.5 * x * y * y)
        y = y * (1.5 - 0.5 * x * y * y)

        return y

    def forward(self, multiview_textures, inv_view_matrix_64d, voxel_data_64):
        def get_mat_val(mat, idx):
            return mat[:, idx:idx+1, :, :]

        # --- 1. カメラレイの生成 ---
        r00, r01, r02 = get_mat_val(inv_view_matrix_64d, 0), get_mat_val(inv_view_matrix_64d, 1), get_mat_val(inv_view_matrix_64d, 2)
        r10, r11, r12 = get_mat_val(inv_view_matrix_64d, 4), get_mat_val(inv_view_matrix_64d, 5), get_mat_val(inv_view_matrix_64d, 6)
        r20, r21, r22 = get_mat_val(inv_view_matrix_64d, 8), get_mat_val(inv_view_matrix_64d, 9), get_mat_val(inv_view_matrix_64d, 10)

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = self.fast_rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx, init_dy, init_dz = dx * inv_len, dy * inv_len, dz * inv_len
        init_px, init_py, init_pz = get_mat_val(inv_view_matrix_64d, 3), get_mat_val(inv_view_matrix_64d, 7), get_mat_val(inv_view_matrix_64d, 11)

        # 💡 [ReLU状態制御] レイの状態とカラーバッファの初期化
        ray_energy_r = self.ONES
        ray_energy_g = self.ONES
        ray_energy_b = self.ONES
        accum_color_r = self.ZEROS
        accum_color_g = self.ZEROS
        accum_color_b = self.ZEROS

        # 全ステップの位置展開
        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        # --- 2. オブジェクト空間へのトランスフォーム ---
        m1_00, m1_01, m1_02, m1_03 = get_mat_val(inv_view_matrix_64d, 16), get_mat_val(inv_view_matrix_64d, 17), get_mat_val(inv_view_matrix_64d, 18), get_mat_val(inv_view_matrix_64d, 19)
        m1_10, m1_11, m1_12, m1_13 = get_mat_val(inv_view_matrix_64d, 20), get_mat_val(inv_view_matrix_64d, 21), get_mat_val(inv_view_matrix_64d, 22), get_mat_val(inv_view_matrix_64d, 23)
        m1_20, m1_21, m1_22, m1_23 = get_mat_val(inv_view_matrix_64d, 24), get_mat_val(inv_view_matrix_64d, 25), get_mat_val(inv_view_matrix_64d, 26), get_mat_val(inv_view_matrix_64d, 27)

        local1_px_all = m1_00 * px_all + m1_01 * py_all + m1_02 * pz_all + m1_03
        local1_py_all = m1_10 * px_all + m1_11 * py_all + m1_12 * pz_all + m1_13
        local1_pz_all = m1_20 * px_all + m1_21 * py_all + m1_22 * pz_all + m1_23

        m2_00, m2_01, m2_02, m2_03 = get_mat_val(inv_view_matrix_64d, 32), get_mat_val(inv_view_matrix_64d, 33), get_mat_val(inv_view_matrix_64d, 34), get_mat_val(inv_view_matrix_64d, 35)
        m2_10, m2_11, m2_12, m2_13 = get_mat_val(inv_view_matrix_64d, 36), get_mat_val(inv_view_matrix_64d, 37), get_mat_val(inv_view_matrix_64d, 38), get_mat_val(inv_view_matrix_64d, 39)
        m2_20, m2_21, m2_22, m2_23 = get_mat_val(inv_view_matrix_64d, 40), get_mat_val(inv_view_matrix_64d, 41), get_mat_val(inv_view_matrix_64d, 42), get_mat_val(inv_view_matrix_64d, 43)

        local2_px_all = m2_00 * px_all + m2_01 * py_all + m2_02 * pz_all + m2_03
        local2_py_all = m2_10 * px_all + m2_11 * py_all + m2_12 * pz_all + m2_13
        local2_pz_all = m2_20 * px_all + m2_21 * py_all + m2_22 * pz_all + m2_23

        # --- 3. 衝突判定の一括レイマーチング ---
        # 💡 外から引数で受け取った `voxel_data_64` を、関数にそのまま引き渡します！
        hit1_all = self.check_multiview_hit(local1_px_all, local1_py_all, local1_pz_all)
        hit2_all = self.check_voxel_hit(local2_px_all, local2_py_all, local2_pz_all, voxel_data_64)
        object_hit_all = torch.max(hit1_all, hit2_all)

        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        # 1x1 Convによる超並列累積和
        cum_hit = self.ane_cumsum_conv(any_hit_all)
        prior_hit = torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        # 衝突座標（ファーストヒット位置）の確定
        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=1, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=1, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=1, keepdim=True)

        # マスクの確定
        hit_object_mask = torch.sum(is_first_object_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)

        # 💡 [ReLU制御] 空（背景）に抜けたレイの色蓄積と生存終了
        is_sky = torch.relu(1.0 - accum_hit)
        accum_color_r = accum_color_r + is_sky * 0.0  # 背景色は黒
        accum_color_g = accum_color_g + is_sky * 0.0
        accum_color_b = accum_color_b + is_sky * 0.0
        ray_energy_r = ray_energy_r * (1.0 - is_sky)
        ray_energy_g = ray_energy_g * (1.0 - is_sky)
        ray_energy_b = ray_energy_b * (1.0 - is_sky)
        # --- 4. 衝突点でのオブジェクトローカル座標と法線（Normal）計算 ---
        local1_px = m1_00 * px + m1_01 * py + m1_02 * pz + m1_03
        local1_py = m1_10 * px + m1_11 * py + m1_12 * pz + m1_13
        local1_pz = m1_20 * px + m1_21 * py + m1_22 * pz + m1_23

        local2_px = m2_00 * px + m2_01 * py + m2_02 * pz + m2_03
        local2_py = m2_10 * px + m2_11 * py + m2_12 * pz + m2_13
        local2_pz = m2_20 * px + m2_21 * py + m2_22 * pz + m2_23

        # 💡 [外部入力適合] 法線用の周囲差分サンプリングにも voxel_data_64 を引き渡します
        f1_c = self.check_multiview_hit(local1_px, local1_py, local1_pz)
        f1_x = self.check_multiview_hit(local1_px + self.eps, local1_py, local1_pz)
        f1_y = self.check_multiview_hit(local1_px, local1_py + self.eps, local1_pz)
        f1_z = self.check_multiview_hit(local1_px, local1_py, local1_pz + self.eps)

        f2_c = self.check_voxel_hit(local2_px, local2_py, local2_pz, voxel_data_64)
        f2_x = self.check_voxel_hit(local2_px + self.eps, local2_py, local2_pz, voxel_data_64)
        f2_y = self.check_voxel_hit(local2_px, local2_py + self.eps, local2_pz, voxel_data_64)
        f2_z = self.check_voxel_hit(local2_px, local2_py, local2_pz + self.eps, voxel_data_64)

        obj1_mask = torch.sum(is_first_object_all * hit1_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        obj2_mask = torch.sum(is_first_object_all * (1.0 - hit1_all) * hit2_all, dim=1, keepdim=True).clamp(0.0, 1.0)

        raw_nx1, raw_ny1, raw_nz1 = f1_c - f1_x, f1_c - f1_y, f1_c - f1_z
        raw_nx2, raw_ny2, raw_nz2 = f2_c - f2_x, f2_c - f2_y, f2_c - f2_z

        world_nx = obj1_mask * (m1_00 * raw_nx1 + m1_10 * raw_ny1 + m1_20 * raw_nz1) + obj2_mask * (m2_00 * raw_nx2 + m2_10 * raw_ny2 + m2_20 * raw_nz2)
        world_ny = obj1_mask * (m1_01 * raw_nx1 + m1_11 * raw_ny1 + m1_21 * raw_nz1) + obj2_mask * (m2_01 * raw_nx2 + m2_10 * raw_ny2 + m2_20 * raw_nz2)
        world_nz = obj1_mask * (m1_02 * raw_nx1 + m1_12 * raw_ny1 + m1_22 * raw_nz1) + obj2_mask * (m2_02 * raw_nx2 + m2_12 * raw_ny2 + m2_20 * raw_nz2)

        inv_true_n_len = self.fast_rsqrt(world_nx*world_nx + world_ny*world_ny + world_nz*world_nz + 1e-5)
        
        first_nx = hit_object_mask * (world_nx * inv_true_n_len)
        first_ny = hit_object_mask * (world_ny * inv_true_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (world_nz * inv_true_n_len)

        # --- 5. シャドウレイ（影）のキャスト ---
        shadow_start_x, shadow_start_y, shadow_start_z = px + 0.04 * self.light_dx, py + 0.04 * self.light_dy, pz + 0.04 * self.light_dz
        spx_all = shadow_start_x + self.light_dx * self.shadow_ratios
        spy_all = shadow_start_y + self.light_dy * self.shadow_ratios
        spz_all = shadow_start_z + self.light_dz * self.shadow_ratios

        l1_spx = m1_00 * spx_all + m1_01 * spy_all + m1_02 * spz_all + m1_03
        l1_spy = m1_10 * spx_all + m1_11 * spy_all + m1_12 * spz_all + m1_13
        l1_spz = m1_20 * spx_all + m1_21 * spy_all + m1_22 * spz_all + m1_23

        l2_spx = m2_00 * spx_all + m2_01 * spy_all + m2_02 * spz_all + m2_03
        l2_spy = m2_10 * spx_all + m2_11 * spy_all + m2_12 * spz_all + m2_13
        l2_spz = m2_20 * spx_all + m2_21 * spy_all + m2_22 * spz_all + m2_23

        s_hit1 = self.check_multiview_hit(l1_spx, l1_spy, l1_spz)
        # 💡 [外部入力適合] 影の判定にも voxel_data_64 を引き渡します
        s_hit2 = self.check_voxel_hit(l2_spx, l2_spy, l2_spz, voxel_data_64)
        accum_shadow = torch.sum(torch.max(s_hit1, s_hit2), dim=1, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

        # 明暗の計算
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15

        # 床のチェッカー模様
        sign_x = torch.clamp(px * 3.0 * 100.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 100.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5

        # --- 6. [ReLU制御マテリアルゲート] インクリメンタルカラー蓄積 ---
        # ① 床ゲート
        gate_floor = torch.relu(hit_floor_mask)
        mat_floor_color = 0.3 + 0.2 * checker
        accum_color_r = accum_color_r + gate_floor * (mat_floor_color * light_modifier) * ray_energy_r
        accum_color_g = accum_color_g + gate_floor * (mat_floor_color * light_modifier) * ray_energy_g
        accum_color_b = accum_color_b + gate_floor * (mat_floor_color * light_modifier) * ray_energy_b
        ray_energy_r = ray_energy_r * (1.0 - gate_floor)
        ray_energy_g = ray_energy_g * (1.0 - gate_floor)
        ray_energy_b = ray_energy_b * (1.0 - gate_floor)

        # ② オブジェクト1ゲート（白い滑らかなマテリアル）
        gate_obj1 = torch.relu(obj1_mask)
        accum_color_r = accum_color_r + gate_obj1 * (1.0 * light_modifier) * ray_energy_r
        accum_color_g = accum_color_g + gate_obj1 * (1.0 * light_modifier) * ray_energy_g
        accum_color_b = accum_color_b + gate_obj1 * (1.0 * light_modifier) * ray_energy_b
        ray_energy_r = ray_energy_r * (1.0 - gate_obj1)
        ray_energy_g = ray_energy_g * (1.0 - gate_obj1)
        ray_energy_b = ray_energy_b * (1.0 - gate_obj1)

        # ③ オブジェクト2ゲート（外部指定できる64^3ボクセル素材）
        gate_obj2 = torch.relu(obj2_mask)
        accum_color_r = accum_color_r + gate_obj2 * (0.7 * light_modifier) * ray_energy_r
        accum_color_g = accum_color_g + gate_obj2 * (0.8 * light_modifier) * ray_energy_g
        accum_color_b = accum_color_b + gate_obj2 * (1.0 * light_modifier) * ray_energy_b
        ray_energy_r = ray_energy_r * (1.0 - gate_obj2)
        ray_energy_g = ray_energy_g * (1.0 - gate_obj2)
        ray_energy_b = ray_energy_b * (1.0 - gate_obj2)

        # --- 7. 出力カラーの結合 ---
        final_color = torch.cat([accum_color_r, accum_color_g, accum_color_b], dim=1)
        return accum_hit * final_color
