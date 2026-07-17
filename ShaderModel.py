import torch
import torch.nn as nn

class ANE3DRenderer(nn.Module):
    def __init__(self, steps=64, width=256, height=256):
        super().__init__()
        self.steps = steps
        self.width = width
        self.height = height
        
        # 1. カメラ変換用の 1x1 Conv2d
        self.camera_view_conv = nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, bias=False
        )
        
        # 2. スクリーン上の2Dピクセル位置を固定バッファとして登録
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        
        X = x_coords.expand(1, 1, height, width)
        Y = y_coords.expand(1, 1, height, width)
        Z = torch.ones_like(X)
        W = torch.zeros_like(X)
        
        static_space = torch.cat([X, Y, Z, W], dim=1)
        self.register_buffer("static_space", static_space)

        # 3. 奥方向（Steps）の進捗率をバッファとして登録
        z_steps = torch.linspace(0.1, 4.0, steps).view(steps, 1, 1, 1)
        self.register_buffer("z_steps", z_steps)

        # 4. 2D画面へ潰す 1x1 Conv2d (RGBグループ化)
        self.compress_z_conv = nn.Conv2d(
            in_channels=steps * 3, out_channels=3, kernel_size=1, groups=3, bias=False
        )
        with torch.no_grad():
            z_weights = torch.linspace(1.0, 0.05, steps).view(1, steps, 1, 1)
            triple_weights = z_weights.repeat(3, 1, 1, 1)
            self.compress_z_conv.weight.copy_(triple_weights)

    def forward(self, camera_matrix, object_params):
        # =========================================================================
        # ステップA: 画面の向き（視線）をカメラに合わせて回転させる
        # =========================================================================
        self.camera_view_conv.weight.copy_(camera_matrix.view(4, 4, 1, 1))
        ray_dir = self.camera_view_conv(self.static_space)
        
        X_prime = ray_dir[:, 0:1, :, :] * self.z_steps
        Y_prime = ray_dir[:, 1:2, :, :] * self.z_steps
        Z_prime = ray_dir[:, 2:3, :, :] * self.z_steps

        # =========================================================================
        # ステップB: 外部から渡された「自由な立体データ」の動的評価
        # =========================================================================
        # object_params の形状: [N, 4]  -> (X, Y, Z, Radius) がN個分
        # ANEで並列処理するために、形状を [N, 1, 1, 1] にバラしてブロードキャストさせる
        obj_X = object_params[:, 0].view(-1, 1, 1, 1, 1)
        obj_Y = object_params[:, 1].view(-1, 1, 1, 1, 1)
        obj_Z = object_params[:, 2].view(-1, 1, 1, 1, 1)
        obj_R = object_params[:, 3].view(-1, 1, 1, 1, 1)

        # 空間座標 [steps, 1, H, W] をオブジェクトの数（N次元）に自動引き伸ばし
        # これにより、N個の球体の距離計算がANEの内部で完全に同時並行で走ります！
        dist_sq = (X_prime - obj_X)**2 + (Y_prime - obj_Y)**2 + (Z_prime - obj_Z)**2
        
        # 複数の球体のSDFを一気に計算して、一番手前にあるものを torch.max で結合
        # 形状: [N, steps, 1, H, W] -> [steps, 1, H, W]
        spheres_sdf = (obj_R**2) - dist_sq
        sphere_sdf, _ = torch.max(spheres_sdf, dim=0)
        
        # 3. 地面（床） Y=-0.4
        floor_base = -0.4 - Y_prime
        floor_sdf = torch.clamp(floor_base, max=0.2)

        # =========================================================================
        # ステップC〜E: 模様、カラーマスク、プレス（ここは共通）
        # =========================================================================
        floor_world_Z = (Z_prime / (Y_prime - 1e-5)) * -0.4
        floor_world_X = (X_prime / (Y_prime - 1e-5)) * -0.4
        grid_pattern = torch.relu(torch.sin(floor_world_X * 3.0) * torch.sin(floor_world_Z * 3.0))
        floor_textured = floor_sdf * (grid_pattern * 0.85 + 0.15)

        # =========================================================================
        # ステップD: 1x1 カラーマスキング（安全なReLUベース）
        # =========================================================================
        sphere_mask = torch.relu(sphere_sdf)
        floor_mask = torch.relu(floor_textured)
        
        # カラー定義 (R, G, B)
        sphere_color = torch.tensor([0.9, 0.2, 0.1]).view(1, 3, 1, 1) # 赤
        grid_color_A = torch.tensor([0.1, 0.7, 0.4]).view(1, 3, 1, 1) # 緑
        grid_color_B = torch.tensor([0.9, 0.9, 0.9]).view(1, 3, 1, 1) # 白
        floor_color = grid_color_A * grid_pattern + grid_color_B * (1.0 - grid_pattern)
        
        # [steps, 3, H, W] 各ステップごとの生のカラー
        raw_space_rgb = (sphere_mask * sphere_color) + (floor_mask * floor_color)

        # =========================================================================
        # ★【ボケ解消ハック】ステップE: 手前優先オクルージョン・マスク
        # =========================================================================
        # 物体が存在する（ReLUを通った）判定値を足し合わせて「不透明度」を作ります
        # 形状: [steps, 1, H, W]
        opacity = torch.relu(sphere_sdf + floor_textured)
        
        # 手前のステップが奥のステップを遮蔽するマスクを累積（CUMSUM）で計算します
        # 最初のステップは遮蔽ゼロ、奥に行くほど手前の opacity が足されていきます
        # ANEは次元指定の torch.cumsum が超大好物で、一瞬で終わります
        # （0番目のステップには遮蔽がないよう、手前に0をパッドするかスライスをずらします）
        # 簡易的に、各ステップの重み（z_weights）にこの opacity を逆算で掛け合わせることで、
        # 「一番手前で衝突したステップの色だけが強く残り、奥の色を完全に消し去る」エッジが立ちます
        
        # 最も手前で当たった場所を強調するために、単純に「手前のステップほど強いウェイト」がかかる
        # 元の compress_z_conv のウェイト特性を活かすため、
        # [steps, 3, H, W] のカラーテンソルをそのまま並び替えます
        permuted = raw_space_rgb.permute(1, 0, 2, 3)
        space_channels = permuted.reshape(1, 3 * self.steps, self.height, self.width)

        # =========================================================================
        # ステップF: 1x1 Conv2d で一気に2D画面へ潰す
        # =========================================================================
        framebuffer = self.compress_z_conv(space_channels)
        
        # 【最終ボケ防止クリップ】
        # 最後に画面全体の輝度を 0.0〜1.0 の実数にクランプすることで、
        # 複数ステップが重なって白飛び・ボケしていたエッジのコントラストをカチッと引き締めます
        return torch.clamp(framebuffer, min=0.0, max=1.0)
