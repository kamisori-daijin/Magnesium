import os
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
        
        # 🌟 ANEのブロードキャスト能力を100%引き出すための4D Tensor定数化
        self.register_buffer("eps", torch.tensor([[[[0.02]]]]).half())
        self.register_buffer("floor_y", torch.tensor([[[[-0.8]]]]).half())
        
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.register_buffer("ONES", torch.ones(1, 1, self.h, self.w).half())
        self.register_buffer("ZEROS", torch.zeros(1, 1, self.h, self.w).half())
        
        # 正規化済みの平行光源方向
        self.register_buffer("light_dx", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dy", torch.tensor([[[[0.5773]]]]).half())
        self.register_buffer("light_dz", torch.tensor([[[[0.5773]]]]).half())

        # ステップ数を最初から【Channel次元 (Dim 1)】に配置してフォールバックを完全阻止
        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())
        self.register_buffer("shadow_ratios", (torch.arange(self.shadow_steps).view(1, self.shadow_steps, 1, 1) * self.dt).half())

        # 3面図マスク金型のモデル内アセット焼き込み
        y_tex = torch.linspace(-1.0, 1.0, self.h).view(1, 1, self.h, 1)
        x_tex = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        base_mask_x = torch.clamp(1.0 - (torch.abs(x_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        base_mask_y = torch.clamp(1.0 - (torch.abs(y_tex) - 0.4) * 100.0, min=0.0, max=1.0)
        cube_2d_mask = (base_mask_x * base_mask_y).half()

        self.register_buffer("base_multiview_textures", torch.cat([cube_2d_mask, cube_2d_mask, cube_2d_mask], dim=1))

        # 🛠️ ANE最速王：非対応の累積和を 1x1 Conv2d に化けさせる専用ウェイト
        self.ane_cumsum_conv = nn.Conv2d(self.max_steps, self.max_steps, kernel_size=1, bias=False)
        weight_matrix = torch.tril(torch.ones(self.max_steps, self.max_steps))
        self.ane_cumsum_conv.weight.data = weight_matrix.view(self.max_steps, self.max_steps, 1, 1).half()
        self.ane_cumsum_conv.weight.requires_grad = False
        
        bg_r, bg_g, bg_b = 0.8, 0.85, 0.9
        bg_tensor = torch.tensor([[[[bg_r]], [[bg_g]], [[bg_b]]]]).half()
        self.register_buffer("bg_color", bg_tensor)

    def fast_rsqrt(self, x):
        """
        🌟 ニュートン法（Fast Inverse Square Root）の ANE 完全シミュレート
        一切の数学特殊演算を排除し、純粋な積和算（加減乗算）だけで平方根の逆数を弾き出します。
        """
        y = 1.0
        y = y * (1.5 - 0.5 * x * y * y)
        y = y * (1.5 - 0.5 * x * y * y)
        return y

    def check_multiview_hit(self, px, py, pz):
        """
        オブジェクト空間の金型シルエット交差判定（純度100%の算術演算）
        """
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
    def forward(self, multiview_textures, inv_view_matrix_64d):
        """
        Input:
            multiview_textures: (ダミー入力)
            inv_view_matrix_64d: (Swift側から届く統合マトリクス・アレイ)
        """
        def get_mat_val(mat, idx):
            return mat[:, idx:idx+1, :, :]

        # ==========================================
        # 🗺️ 1. カメラ逆行列 & オブジェクト逆モデル行列(2個分)の展開
        # ==========================================
        r00 = get_mat_val(inv_view_matrix_64d, 0)
        r01 = get_mat_val(inv_view_matrix_64d, 1)
        r02 = get_mat_val(inv_view_matrix_64d, 2)
        
        r10 = get_mat_val(inv_view_matrix_64d, 4)
        r11 = get_mat_val(inv_view_matrix_64d, 5)
        r12 = get_mat_val(inv_view_matrix_64d, 6)
        
        r20 = get_mat_val(inv_view_matrix_64d, 8)
        r21 = get_mat_val(inv_view_matrix_64d, 9)
        r22 = get_mat_val(inv_view_matrix_64d, 10)
    
        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz
    
        # 🌟 ANEネイティブ高速rsqrtの適用
        inv_len = self.fast_rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx = dx * inv_len
        init_dy = dy * inv_len
        init_dz = dz * inv_len

        init_px = get_mat_val(inv_view_matrix_64d, 3)
        init_py = get_mat_val(inv_view_matrix_64d, 7)
        init_pz = get_mat_val(inv_view_matrix_64d, 11)

        # ==========================================
        # 🚀 2. メインの視線レイマーチング (🌟完全にChannel次元並列・Reshapeゼロ)
        # ==========================================
        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        # オブジェクト1の空間ワープ
        m1_00 = get_mat_val(inv_view_matrix_64d, 16)
        m1_01 = get_mat_val(inv_view_matrix_64d, 17)
        m1_02 = get_mat_val(inv_view_matrix_64d, 18)
        m1_03 = get_mat_val(inv_view_matrix_64d, 19)

        m1_10 = get_mat_val(inv_view_matrix_64d, 20)
        m1_11 = get_mat_val(inv_view_matrix_64d, 21)
        m1_12 = get_mat_val(inv_view_matrix_64d, 22)
        m1_13 = get_mat_val(inv_view_matrix_64d, 23)

        m1_20 = get_mat_val(inv_view_matrix_64d, 24)
        m1_21 = get_mat_val(inv_view_matrix_64d, 25)
        m1_22 = get_mat_val(inv_view_matrix_64d, 26)
        m1_23 = get_mat_val(inv_view_matrix_64d, 27)

        # オブジェクト2の空間ワープ
        m2_00 = get_mat_val(inv_view_matrix_64d, 32)
        m2_01 = get_mat_val(inv_view_matrix_64d, 33)
        m2_02 = get_mat_val(inv_view_matrix_64d, 34)
        m2_03 = get_mat_val(inv_view_matrix_64d, 35)
        m2_10 = get_mat_val(inv_view_matrix_64d, 36)
        m2_11 = get_mat_val(inv_view_matrix_64d, 37)
        m2_12 = get_mat_val(inv_view_matrix_64d, 38)
        m2_13 = get_mat_val(inv_view_matrix_64d, 39)

        # 2物体の判定を並列で行い、最大値で1空間に合成
        m2_20 = get_mat_val(inv_view_matrix_64d, 40)
        m2_21 = get_mat_val(inv_view_matrix_64d, 41)
        m2_22 = get_mat_val(inv_view_matrix_64d, 42)
        m2_23 = get_mat_val(inv_view_matrix_64d, 43)
        local1_px_all = m1_00 * px_all + m1_01 * py_all + m1_02 * pz_all + m1_03
        local1_py_all = m1_10 * px_all + m1_11 * py_all + m1_12 * pz_all + m1_13
        local1_pz_all = m1_20 * px_all + m1_21 * py_all + m1_22 * pz_all + m1_23

        local2_px_all = m2_00 * px_all + m2_01 * py_all + m2_02 * pz_all + m2_03
        local2_py_all = m2_10 * px_all + m2_11 * py_all + m2_12 * pz_all + m2_13
        local2_pz_all = m2_20 * px_all + m2_21 * py_all + m2_22 * pz_all + m2_23

        # 2物体の判定を並列で行い、最大値で1空間に合成
        hit1_all = self.check_multiview_hit(local1_px_all, local1_py_all, local1_pz_all)
        hit2_all = self.check_multiview_hit(local2_px_all, local2_py_all, local2_pz_all)
        object_hit_all = torch.max(hit1_all, hit2_all)

        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        # 🛠️ 1x1 Convによる累積和（フォールバックを完全阻止）
        cum_hit = self.ane_cumsum_conv(any_hit_all)

        prior_hit = torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        is_first_floor_all = not_hit_yet_all * (1.0 - object_hit_all) * floor_hit_all

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
        local1_px = m1_00 * px + m1_01 * py + m1_02 * pz + m1_03
        local1_py = m1_10 * px + m1_11 * py + m1_12 * pz + m1_13
        local1_pz = m1_20 * px + m1_21 * py + m1_22 * pz + m1_23

        local2_px = m2_00 * px + m2_01 * py + m2_02 * pz + m2_03
        local2_py = m2_10 * px + m2_11 * py + m2_12 * pz + m2_13
        local2_pz = m2_20 * px + m2_21 * py + m2_22 * pz + m2_23

        f1_c = self.check_multiview_hit(local1_px, local1_py, local1_pz)
        f2_c = self.check_multiview_hit(local2_px, local2_py, local2_pz)

        f1_x = self.check_multiview_hit(local1_px + self.eps, local1_py, local1_pz)
        f2_x = self.check_multiview_hit(local2_px + self.eps, local2_py, local2_pz)

        f1_y = self.check_multiview_hit(local1_px, local1_py + self.eps, local1_pz)
        f2_y = self.check_multiview_hit(local2_px, local2_py + self.eps, local2_pz)

        f1_z = self.check_multiview_hit(local1_px, local1_py, local1_pz + self.eps)
        f2_z = self.check_multiview_hit(local2_px, local2_py, local2_pz + self.eps)

        obj1_mask = torch.sum(is_first_object_all * hit1_all, dim=1, keepdim=True).clamp(0.0, 1.0)
        obj2_mask = torch.sum(is_first_object_all * (1.0 - hit1_all) * hit2_all, dim=1, keepdim=True).clamp(0.0, 1.0)

        raw_nx1 = f1_c - f1_x
        raw_ny1 = f1_c - f1_y
        raw_nz1 = f1_c - f1_z
        raw_nx2 = f2_c - f2_x
        raw_ny2 = f2_c - f2_y
        raw_nz2 = f2_c - f2_z

        # メモリコピーを挟まず積和だけで一撃ワールド法線合成（Reshapeゼロ）
        world_nx = obj1_mask * (m1_00 * raw_nx1 + m1_10 * raw_ny1 + m1_20 * raw_nz1) + obj2_mask * (m2_00 * raw_nx2 + m2_10 * raw_ny2 + m2_20 * raw_nz2)
        world_ny = obj1_mask * (m1_01 * raw_nx1 + m1_11 * raw_ny1 + m1_21 * raw_nz1) + obj2_mask * (m2_01 * raw_nx2 + m2_11 * raw_ny2 + m2_21 * raw_nz2)
        world_nz = obj1_mask * (m1_02 * raw_nx1 + m1_12 * raw_ny1 + m1_22 * raw_nz1) + obj2_mask * (m2_02 * raw_nx2 + m2_12 * raw_ny2 + m2_22 * raw_nz2)

        inv_true_n_len = self.fast_rsqrt(world_nx*world_nx + world_ny*world_ny + world_nz*world_nz + 1e-5)

        first_nx = hit_object_mask * (world_nx * inv_true_n_len)
        first_ny = hit_object_mask * (world_ny * inv_true_n_len) + hit_floor_mask * 1.0
        first_nz = hit_object_mask * (world_nz * inv_true_n_len)

    # ==========================================
    # 👤 4. シャドウレイ（🌟Reshapeゼロ並列判定）
    # ==========================================
        shadow_start_x = px + 0.04 * self.light_dx
        shadow_start_y = py + 0.04 * self.light_dy
        shadow_start_z = pz + 0.04 * self.light_dz

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
        s_hit2 = self.check_multiview_hit(l2_spx, l2_spy, l2_spz)
        accum_shadow = torch.sum(torch.max(s_hit1, s_hit2), dim=1, keepdim=True).clamp(0.0, 1.0) * hit_floor_mask

    # ==========================================
    # 🎨 4.5 マテリアル数値の取得とブレンド (🌟新設・固定レイアウト拡張)
    # ==========================================
    # 統合バッファの空きch（48〜53）から直接マテリアル数値を抽出
        mat1_ior = get_mat_val(inv_view_matrix_64d, 48)
        mat1_metallic = get_mat_val(inv_view_matrix_64d, 50)

        mat2_ior = get_mat_val(inv_view_matrix_64d, 51)
        mat2_metallic = get_mat_val(inv_view_matrix_64d, 53)

        # 衝突した画素に応じて IOR と Metallic をブレンド合成
        pixel_ior = obj1_mask * mat1_ior + obj2_mask * mat2_ior
        pixel_metallic = obj1_mask * mat1_metallic + obj2_mask * mat2_metallic

        # ==========================================
        # 🎨 5. 汎用アーティスト制御シェーディング（法線自動補正・ユニバーサル完全版）
        # ==========================================
        # 5.0 外からの直感的パラメーターを直接その場でバインド
        raw_p1_trans   = get_mat_val(inv_view_matrix_64d, 48) # 透明度 (0.0~1.0)
        raw_p1_reflect = get_mat_val(inv_view_matrix_64d, 49) # 反射度 (0.0~1.0)
        raw_p1_distort = get_mat_val(inv_view_matrix_64d, 50) # 歪み強度 (0.0~0.5)
        
        raw_p2_trans   = get_mat_val(inv_view_matrix_64d, 51) 
        raw_p2_reflect = get_mat_val(inv_view_matrix_64d, 52) 
        raw_p2_distort = get_mat_val(inv_view_matrix_64d, 53) 

        # 衝突した画素に応じたパラメーターの合成
        p_trans   = obj1_mask * raw_p1_trans   + obj2_mask * raw_p2_trans
        p_reflect = obj1_mask * raw_p1_reflect + obj2_mask * raw_p2_reflect
        p_distort = obj1_mask * raw_p1_distort + obj2_mask * raw_p2_distort

        # 5.1 視線ベクトルの正規化
        inv_I_len = self.fast_rsqrt(init_dx * init_dx + init_dy * init_dy + init_dz * init_dz + 1e-5)
        norm_idx = init_dx * inv_I_len
        norm_idy = init_dy * inv_I_len
        norm_idz = init_dz * inv_I_len

        # 🌟【重要：カメラ角度破綻を防ぐ法線方向の自動補正】
        # 視線ベクトルと生の法線（first_nx等）の内積を計算
        raw_dot_I_N = norm_idx * first_nx + norm_idy * first_ny + norm_idz * first_nz
        
        # 内積がプラス（法線がオブジェクトの裏側に突き抜けている画素）を検出するマスク
        is_flipped = torch.clamp(raw_dot_I_N * 1000.0, min=0.0, max=1.0)
        
        # 裏返っている画素だけ符号を反転（-1を乗算）し、常にカメラ側を向く安全な法線を作る
        safe_nx = first_nx * (1.0 - 2.0 * is_flipped)
        safe_ny = first_ny * (1.0 - 2.0 * is_flipped)
        safe_nz = first_nz * (1.0 - 2.0 * is_flipped)

        # 補正後の正しい法線を使って内積を再計算（0.0〜1.0）
        dot_I_N = torch.clamp(-(norm_idx * safe_nx + norm_idy * safe_ny + norm_idz * safe_nz), min=0.0, max=1.0)
        
        # 輪郭度 (0.0で正面、1.0で完全なフチ)
        edge_mask = torch.clamp(1.0 - dot_I_N, min=0.0, max=1.0)
        normal_sanity = torch.clamp(dot_I_N * 3.0, min=0.0, max=1.0)

        # 5.3 ディフューズ（陰影）とシャドウの統合計算 (補正後の safe_n を使用)
        dot_nl = torch.clamp(safe_nx * self.light_dx + safe_ny * self.light_dy + safe_nz * self.light_dz, min=0.0, max=1.0)
        shading = dot_nl + 0.15
        # 👤 影の処理：影の部分（accum_shadow=1.0）は強制的に暗く(0.15)沈める
        light_modifier = (self.ONES - accum_shadow) * shading + accum_shadow * 0.15

        # 5.4 パラメーター「歪み強度」を使った屈折方向の疑似計算 (補正後の safe_n を使用)
        refract_dx = norm_idx + safe_nx * p_distort * normal_sanity
        refract_dy = norm_idy + safe_ny * p_distort * normal_sanity
        refract_dz = norm_idz * safe_nz * p_distort * normal_sanity

        # 5.6 屈折レイの床衝突座標（斜め上アングル完全対応・絶対値ガード）
        is_heading_down = torch.clamp(-refract_dy * 1000.0, min=0.0, max=1.0) 
        
        # 🌟 修正：分母が小さくなりすぎるのを防ぎ、ゼロ除算によるシマシマを防止
        safe_denom = torch.clamp(torch.abs(refract_dy), min=0.05, max=1.0) * torch.sign(refract_dy)
        safe_denom = safe_denom * is_heading_down + (-1.0) * (1.0 - is_heading_down)
        
        dist_to_floor = torch.abs(self.floor_y - py) / torch.abs(safe_denom)
        
        # 🌟 修正：遠すぎる交差点をフェードアウトさせてアーティファクトを防ぐ
        fade_out = torch.clamp(1.0 - dist_to_floor / 10.0, min=0.0, max=1.0)
        
        r_floor_px = px + refract_dx * dist_to_floor
        r_floor_pz = pz + refract_dz * dist_to_floor

        # 5.7 チェッカー床の生成
        sign_x = torch.clamp(px * 3.0 * 100.0, min=-1.0, max=1.0)
        sign_z = torch.clamp(pz * 3.0 * 100.0, min=-1.0, max=1.0)
        checker = (sign_x * sign_z + 1.0) * 0.5
        
        r_sign_x = torch.clamp(r_floor_px * 3.0 * 100.0, min=-1.0, max=1.0)
        r_sign_z = torch.clamp(r_floor_pz * 3.0 * 100.0, min=-1.0, max=1.0)
        r_checker = (r_sign_x * r_sign_z + 1.0) * 0.5

        floor_raw_color = (0.3 + 0.2 * checker)
        rgb_floor_color = torch.cat([floor_raw_color, floor_raw_color, floor_raw_color], dim=1) * light_modifier

        r_floor_raw_color = (0.3 + 0.2 * r_checker)
        rgb_refracted_floor_color = torch.cat([r_floor_raw_color, r_floor_raw_color, r_floor_raw_color], dim=1) * light_modifier

        # 5.5 基本色の展開
        c1_r = get_mat_val(inv_view_matrix_64d, 54)
        c1_g = get_mat_val(inv_view_matrix_64d, 55)
        c1_b = get_mat_val(inv_view_matrix_64d, 56)
        c2_r = get_mat_val(inv_view_matrix_64d, 57)
        c2_g = get_mat_val(inv_view_matrix_64d, 58)
        c2_b = get_mat_val(inv_view_matrix_64d, 59)
        base_color = torch.cat([obj1_mask * c1_r + obj2_mask * c2_r,
                                obj1_mask * c1_g + obj2_mask * c2_g,
                                obj1_mask * c1_b + obj2_mask * c2_b], dim=1)

        # ① 透過光の計算
        is_refract_hit_floor = torch.clamp(r_floor_raw_color * 5.0, min=0.0, max=1.0) * is_heading_down
        transmitted_color = is_refract_hit_floor * rgb_refracted_floor_color + (1.0 - is_refract_hit_floor) * self.bg_color
        
        thickness_shadow = torch.clamp(dot_I_N, min=0.4, max=1.0) * normal_sanity + 0.4 * (1.0 - normal_sanity)
        transmitted_color = transmitted_color * (base_color * 0.5 + 0.5) * thickness_shadow

        # ② 反射光の計算
        reflected_color = self.bg_color * (1.0 - p_reflect) + base_color * p_reflect
        reflected_color = torch.clamp(reflected_color + edge_mask * 0.3, 0.0, 1.0)

        # ③ 透明度パラメーターによる最終マテリアルルックのブレンド (Schlickの近似式)
        R0 = p_reflect
        fresnel = R0 + (1.0 - R0) * torch.pow(edge_mask, 5.0)
        blend_mask = p_trans * (1.0 - fresnel)
        
        obj_color = (1.0 - blend_mask) * reflected_color + blend_mask * transmitted_color
        obj_color = obj_color * light_modifier

        # 5.9 鏡面ハイライト (光沢)
        specular = torch.pow(dot_nl, 64.0) * 0.8 * hit_object_mask

        # 5.10 最終シーン合成
        final_scene_color = hit_object_mask * (obj_color + specular) + hit_floor_mask * rgb_floor_color
        
        return final_scene_color + (1.0 - accum_hit) * self.bg_color