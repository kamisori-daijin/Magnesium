import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANERayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=64, shadow_steps=16):
        super().__init__()
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.shadow_steps = shadow_steps
        self.dt = 0.08
        self.eps = 0.02  # 数値微分用の微小変化量
        
        # --- カメラ空間での初期レイ（方向・位置）の生成 ---
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.floor_y = -0.8
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        # 平行光源の方向（ワールド空間）
        light_dir_x = torch.full((1, 1, 1, 1), 1.0)
        light_dir_y = torch.full((1, 1, 1, 1), 1.0)
        light_dir_z = torch.full((1, 1, 1, 1), 1.0)
        inv_l_len = torch.rsqrt(light_dir_x*light_dir_x + light_dir_y*light_dir_y + light_dir_z*light_dir_z + 1e-5)
        self.register_buffer("light_dx", (light_dir_x * inv_l_len).half())
        self.register_buffer("light_dy", (light_dir_y * inv_l_len).half())
        self.register_buffer("light_dz", (light_dir_z * inv_l_len).half())

        # ANE最適化: permuteを全滅させるため、最初からステップ数を【Channel次元 (Dim 1)】に配置
        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(self.shadow_steps).view(1, self.shadow_steps, 1, 1) * self.dt).half())

        # RTコア化ハック: 3面図マスク金型をモデル内部に完全にアセット焼き込み！
        y_tex = torch.linspace(-1.0, 1.0, self.h).view(1, 1, self.h, 1)
        x_tex = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        base_mask_x = torch.clamp(1.0 - (torch.abs(x_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        base_mask_y = torch.clamp(1.0 - (torch.abs(y_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        cube_2d_mask = (base_mask_x * base_mask_y).half()

        # XY, XZ, YZ としてモデル内部にバッファ登録
        self.register_buffer("base_multiview_textures", torch.cat([cube_2d_mask, cube_2d_mask, cube_2d_mask], dim=1))

        # 🌟 🛠️ ANE専用：torch.cumsum を駆逐する 1x1 畳み込みフィルターの定義
        # [max_steps -> max_steps] への 1x1 Conv を累積和（下三角行列）の重みで初期化
        self.ane_cumsum_conv = nn.Conv2d(
            in_channels=self.max_steps, 
            out_channels=self.max_steps, 
            kernel_size=1, 
            bias=False
        )
        weight_matrix = torch.tril(torch.ones(self.max_steps, self.max_steps))
        self.ane_cumsum_conv.weight.data = weight_matrix.view(self.max_steps, self.max_steps, 1, 1).half()
        self.ane_cumsum_conv.weight.requires_grad = False

    def check_multiview_hit(self, px, py, pz):
        """
        px, py, pz: warp coordinates
        """
        # Box boundary check（オブジェクトのローカル空間の境界判定）
        out_x = torch.relu(torch.abs(px) - 1.0)
        out_y = torch.relu(torch.abs(py) - 1.0)
        out_z = torch.relu(torch.abs(pz) - 1.0)
        any_out = torch.clamp((out_x + out_y + out_z) * 100.0, min=0.0, max=1.0)
        box_check = 1.0 - any_out

        # CPUで行っていた「abs(軸) <= 0.4」をReLUと一斉クランプの数式だけでANE上に完全シミュレート！
        proj_xy = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0)
                  
        proj_xz = torch.clamp(1.0 - torch.relu(torch.abs(px) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)
                  
        proj_yz = torch.clamp(1.0 - torch.relu(torch.abs(py) - 0.4) * 10.0, 0.0, 1.0) * \
                  torch.clamp(1.0 - torch.relu(torch.abs(pz) - 0.4) * 10.0, 0.0, 1.0)

        # マスクの幾何学結合
        mask_xy = self.base_multiview_textures[:, 0:1, :, :] * proj_xy
        mask_xz = self.base_multiview_textures[:, 1:2, :, :] * proj_xz
        mask_yz = self.base_multiview_textures[:, 2:3, :, :] * proj_yz

        object_hit = mask_xy * mask_xz * mask_yz * box_check
        return object_hit
    def forward(self, multiview_textures, inv_view_matrix_64d):
        """
        Input:
              multiview_textures: [1, 3, 256, 256] (※内蔵バッファ化したためダミー入力でOK)
              inv_view_matrix_64d: [1, 64, 1, 1] (Swift側から一撃で届く統合バッファ)
        """
        # ==========================================
        # 🗺️ 1. カメラ逆行列 & オブジェクト逆モデル行列の展開
        # ==========================================
        # 最初の16chからカメラの逆行列を復元
        inv_view = inv_view_matrix_64d[0, :16, 0, 0].view(4, 4).half()

        # 次の16ch（16〜31）からオブジェクトの「モデル逆行列」を復元
        inv_model = inv_view_matrix_64d[0, 16:32, 0, 0].view(4, 4).half()

        # カメラ行列の分解によるレイ方向の算出
        r00, r01, r02 = inv_view[0, 0], inv_view[0, 1], inv_view[0, 2]
        r10, r11, r12 = inv_view[1, 0], inv_view[1, 1], inv_view[1, 2]
        r20, r21, r22 = inv_view[2, 0], inv_view[2, 1], inv_view[2, 2]

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx = dx * inv_len
        init_dy = dy * inv_len
        init_dz = dz * inv_len

        init_px = inv_view[0, 3].view(1, 1, 1, 1)
        init_py = inv_view[1, 3].view(1, 1, 1, 1)
        init_pz = inv_view[2, 3].view(1, 1, 1, 1)

        # ==========================================
        # 🚀 2. メインの視線レイマーチング (🌟完全にChannel次元並列)
        # ==========================================
        # [1, max_steps, H, W] へ一撃ブロードキャスト
        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        # GPUのRTコアを模倣した「レイ座標のオブジェクト逆空間ワープ」
        m00, m01, m02, m03 = inv_model[0, 0], inv_model[0, 1], inv_model[0, 2], inv_model[0, 3]
        m10, m11, m12, m13 = inv_model[1, 0], inv_model[1, 1], inv_model[1, 2], inv_model[1, 3]
        m20, m21, m22, m23 = inv_model[2, 0], inv_model[2, 1], inv_model[2, 2], inv_model[2, 3]

        local_px_all = m00 * px_all + m01 * py_all + m02 * pz_all + m03
        local_py_all = m10 * px_all + m11 * py_all + m12 * pz_all + m13
        local_pz_all = m20 * px_all + m21 * py_all + m22 * pz_all + m23

        # 変形されたローカル空間の座標を使って、新・ソリッド金型シルエットと交差判定！
        object_hit_all = self.check_multiview_hit(local_px_all, local_py_all, local_pz_all)

        # 床はワールド空間の元の高さ（py_all）で不変判定
        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        # 🌟 🛠️ ANE最適化：非対応の torch.cumsum を 1x1 Conv で完全置き換え
        cum_hit = self.ane_cumsum_conv(any_hit_all)

        prior_hit = torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

        # 各ステップのフラグをDim 1（Channel）で総和して縮約 [1, 1, H, W] へ戻す
        hit_object_mask = torch.sum(is_first_object_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        hit_floor_mask = torch.sum(is_first_floor_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        accum_hit = torch.clamp(hit_object_mask + hit_floor_mask, min=0.0, max=1.0)

        # 最終衝突した『ワールド空間』の3D座標を確定
        px = torch.sum(is_first_object_all * px_all + is_first_floor_all * px_all, dim=1, keepdim=True)
        py = torch.sum(is_first_object_all * py_all + is_first_floor_all * py_all, dim=1, keepdim=True)
        pz = torch.sum(is_first_object_all * pz_all + is_first_floor_all * pz_all, dim=1, keepdim=True)

        # ==========================================
        # 📐 3. 全自動・数値微分による法線計算
        # ==========================================
        # 微分用のサンプリング点も、一度オブジェクトのローカル空間に射影
        local_px = m00 * px + m01 * py + m02 * pz + m03
        local_py = m10 * px + m11 * py + m12 * pz + m13
        local_pz = m20 * px + m21 * py + m22 * pz + m23

        # 新しい check_multiview_hit を通してローカル空間での正確な法線勾配を中央差分で抽出
        f_center = self.check_multiview_hit(local_px, local_py, local_pz)
        f_dx = self.check_multiview_hit(local_px + self.eps, local_py, local_pz)
        f_dy = self.check_multiview_hit(local_px, local_py + self.eps, local_pz)
        f_dz = self.check_multiview_hit(local_px, local_py, local_pz + self.eps)

        raw_nx = f_center - f_dx
        raw_ny = f_center - f_dy
        raw_nz = f_center - f_dz

        # グラフィックスハック: ローカル空間の法線をモデル行列の回転成分でワールド空間に戻す
        world_nx = m00 * raw_nx + m10 * raw_ny + m20 * raw_nz
        world_ny = m01 * raw_nx + m11 * raw_ny + m21 * raw_nz
        world_nz = m02 * raw_nx + m12 * raw_ny + m22 * raw_nz

        inv_true_n_len = torch.rsqrt(world_nx*world_nx + world_ny*world_ny + world_nz*world_nz + 1e-5)
        # 物体表面ならワールド法線、床なら真上(1.0)にする
        first_nx = hit_object_mask * (world_nx * inv_true_n_len)
        first_ny = hit_object_mask * (world_ny * inv_true_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (world_nz * inv_true_n_len)

        # ==========================================
        # 👤 4. シャドウレイ (🌟こちらも完全にChannel並列)
        # ==========================================
        shadow_start_x = px + 0.04 * self.light_dx
        shadow_start_y = py + 0.04 * self.light_dy
        shadow_start_z = pz + 0.04 * self.light_dz

        # [1, shadow_steps, H, W] に一撃ブロードキャスト展開
        spx_all = shadow_start_x + self.light_dx * self.shadow_ratios
        spy_all = shadow_start_y + self.light_dy * self.shadow_ratios
        spz_all = shadow_start_z + self.light_dz * self.shadow_ratios

        # シャドウの衝突判定も、同様にレイの座標をオブジェクトのローカル空間にワープさせて判定
        local_spx_all = m00 * spx_all + m01 * spy_all + m02 * spz_all + m03
        local_spy_all = m10 * spx_all + m11 * spy_all + m12 * spz_all + m13
        local_spz_all = m20 * spx_all + m21 * spy_all + m22 * spz_all + m23

        shadow_hit_all = self.check_multiview_hit(local_spx_all, local_spy_all, local_spz_all)
        accum_shadow = torch.sum(shadow_hit_all, dim=1, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

        # ==========================================
        # 🎨 5. ライティング ＆ チェッカー床
        # ==========================================
        diffuse = first_nx * self.light_dx + first_ny * self.light_dy + first_nz * self.light_dz
        shading = torch.relu(diffuse) + 0.15

        sign_x = torch.clamp(px * 3.0 * 100.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 100.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5
        floor_color = hit_floor_mask * (0.3 + 0.2 * checker)

        base_color = hit_object_mask * self.ONES + floor_color
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15

        return accum_hit * base_color * light_modifier
