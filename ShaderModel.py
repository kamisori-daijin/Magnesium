import torch
import torch.nn as nn
import torch.nn.functional as F

class ANE3DRenderer(nn.Module):
    def __init__(self, max_triangles=2000, width=256, height=256):
        """
        max_triangles: ANE側に確保させる一度にラスタライズできる最大三角形数。
                       今回はトレース用（固定枠）として2000ポリゴンをデフォルトに設定。
        """
        super().__init__()
        self.max_triangles = max_triangles
        self.width = width
        self.height = height
        
        # 1. 画面のピクセル位置 [X, Y, 1] の固定グリッドバッファ (256x256)
        # 形状: [1, 3, height, width] の4次元画像として最初から綺麗に登録
        y_coords = torch.linspace(1.0, -1.0, height).view(1, 1, height, 1)
        x_coords = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        X = x_coords.expand(1, 1, height, width)
        Y = y_coords.expand(1, 1, height, width)
        One = torch.ones_like(X)
        
        pixel_grid = torch.cat([X, Y, One], dim=1) # 形状: [1, 3, H, W]
        self.register_buffer("pixel_grid", pixel_grid)

    def forward(self, transformed_vertices):
        """
        transformed_vertices: [1, 3, 1, MAX_VERTICES]
        """
        # =========================================================================
        # ステップA: 頂点バッファを三角形（3頂点ずつ）の並びへ綺麗に抽出 ★端数カット版
        # =========================================================================
        # max_triangles * 3 の長さだけを正確に切り出すことで、お尻のゴミ（余り）を完全に無視します
        valid_len = self.max_triangles * 3
        v_buffer = transformed_vertices[:, :, :, :valid_len]
        
        # これで p0, p1, p2 の長さが寸分の狂いもなく [1, 1, 1, max_triangles] でカチッと一致します！
        p0_X = v_buffer[:, 0:1, :, 0::3]
        p0_Y = v_buffer[:, 1:2, :, 0::3]
        p1_X = v_buffer[:, 0:1, :, 1::3]
        p1_Y = v_buffer[:, 1:2, :, 1::3]
        p2_X = v_buffer[:, 0:1, :, 2::3]
        p2_Y = v_buffer[:, 1:2, :, 2::3]

        # 深度Zも完全に [1, 1, 1, max_triangles] 同士の足し算になるので、1ミリもエラーになりません
        z0 = v_buffer[:, 2:3, :, 0::3]
        z1 = v_buffer[:, 2:3, :, 1::3]
        z2 = v_buffer[:, 2:3, :, 2::3]
        avg_z = (z0 + z1 + z2) / 3.0

        # =========================================================================
        # ステップB: 全ポリゴンの3辺の直線方程式（A, B, C）をANEの要素別演算で自動生成
        # =========================================================================
        # 1行のループも条件分岐もなし。形状はすべて [1, 1, 1, max_triangles]
        A0 = p0_Y - p1_Y
        B0 = p1_X - p0_X
        C0 = -(A0 * p0_X + B0 * p0_Y)

        A1 = p1_Y - p2_Y
        B1 = p2_X - p1_X
        C1 = -(A1 * p1_X + B1 * p1_Y)

        A2 = p2_Y - p0_Y
        B2 = p0_X - p2_X
        C2 = -(A2 * p0_X + B2 * p0_Y)

        # =========================================================================
        # ステップC: ピクセルグリッドへの一斉ラスタライズ（内外判定）
        # =========================================================================
        # ピクセルグリッド [1, 3, H, W] の X と Y に対して、直線方程式を直接評価！
        # ANEが大好きな要素ごとの掛け算と足し算だけで、全ピクセルの内外判定が並列で走ります
        # max_trianglesの次元を「幅」ではなく「チャンネル次元」に変形してConvに直撃させるため、
        # ここでは [1, max_triangles, H, W] の巨大な空間マスクを生成します。
        pixel_X = self.pixel_grid[:, 0:1, :, :] # [1, 1, H, W]
        pixel_Y = self.pixel_grid[:, 1:2, :, :]
        
        # 三角形のパラメータを [1, max_triangles, 1, 1] に見立ててブロードキャスト
        # ANEは「画像 [1, 1, H, W] と ウェイト [1, N, 1, 1] の掛け算」が一番の好物です
        A0_t = A0.permute(0, 3, 1, 2) # [1, max_triangles, 1, 1]
        B0_t = B0.permute(0, 3, 1, 2)
        C0_t = C0.permute(0, 3, 1, 2)
        edges0 = pixel_X * A0_t + pixel_Y * B0_t + C0_t

        A1_t = A1.permute(0, 3, 1, 2)
        B1_t = B1.permute(0, 3, 1, 2)
        C1_t = C1.permute(0, 3, 1, 2)
        edges1 = pixel_X * A1_t + pixel_Y * B1_t + C1_t

        A2_t = A2.permute(0, 3, 1, 2)
        B2_t = B2.permute(0, 3, 1, 2)
        C2_t = C2.permute(0, 3, 1, 2)
        edges2 = pixel_X * A2_t + pixel_Y * B2_t + C2_t

        # =========================================================================
        # ステップD: 判定は100%これまで通り `ReLU` で！
        # =========================================================================
        # 3つの辺すべてに対して内側（プラス）の場所をパキパキに現像
        valid_poly_mask = torch.relu((A0_t**2 + B0_t**2) * 100.0)
        valid_poly_mask = torch.clamp(valid_poly_mask, min=0.0, max=1.0)

        # ★【ここが究極ハック】
        # 3辺の符号が（プラス、プラス、プラス）または（マイナス、マイナス、マイナス）の時に
        # 「内側」にいることになるため、3つの判定値をそのまま掛け算した後に
        # torch.abs（絶対値）を取るか、あるいは符号のねじれを吸収させます。
        # ANEは大好物の ReLU だけの組み合わせで「両面の内側」を一撃抽出できます。
        
        # パターン1: 時計回りの内側
        inside_clockwise = torch.relu(edges0 * 100.0) * torch.relu(edges1 * 100.0) * torch.relu(edges2 * 100.0)
        
        # パターン2: 反時計回りの内側 (マイナスを掛けて反転させてからReLUに流す)
        inside_counter = torch.relu(-edges0 * 100.0) * torch.relu(-edges1 * 100.0) * torch.relu(-edges2 * 100.0)
        
        # 2つのパターンのどちらかに引っかかっていれば「内側」！
        # ANEが大好きな torch.maximum で一撃結合します
        raw_mask = torch.maximum(inside_clockwise, inside_counter) * valid_poly_mask
        all_triangles_mask = torch.clamp(raw_mask, min=0.0, max=1.0)

        # =========================================================================
        # ステップE: Zバッファ（深度隠面消去）＆一撃プレス
        # =========================================================================
        # 平均Z [1, 1, 1, max_triangles] を [1, max_triangles, 1, 1] に変換
        poly_normal_Z = torch.abs(A0 * B2 - B0 * A2).permute(0, 3, 1, 2) # [1, max_triangles, 1, 1]
        # 数値をマイルドな輝度グラデーション（0.3 〜 1.0）へ正規化
        shading = torch.clamp(poly_normal_Z * 5.0, min=0.3, max=1.0)

        # 深度テスト用のウェイト計算（これはそのまま）
        avg_z_t = avg_z.permute(0, 3, 1, 2)
        z_weight = torch.clamp(1.0 - (avg_z_t / 4.0), min=0.0, max=1.0)
        
        # ★マスクに対して「奥行き重み」と「面の明るさ（shading）」を同時に掛け算！
        weighted_triangles = all_triangles_mask * z_weight * shading
        
        rendered_space, _ = torch.max(weighted_triangles, dim=1, keepdim=True)
        framebuffer = rendered_space.repeat(1, 3, 1, 1)
        
        return framebuffer
