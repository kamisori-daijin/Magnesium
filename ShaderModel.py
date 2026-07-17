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
        transformed_vertices: [1, 3, 1, MAX_VERTICES] (第1段のMVPプロセッサから吐き出された直撃データ)
                              0ch: 画面X, 1ch: 画面Y, 2ch: 深度Z
        """
        # =========================================================================
        # ステップA: 頂点バッファを三角形（3頂点ずつ）の並びへ1x1 Conv互換で切り出し
        # =========================================================================
        # ANEにへそを曲げさせないため、reshapeの代わりに 1x1 Conv のストライドや
        # チャンネルスライスを使って、p0, p1, p2 の座標 [1, 2, 1, max_triangles] を綺麗に抽出します。
        # 3万〜6万頂点の中から、三角形の各角のXYを綺麗に並列化
        p0_X = transformed_vertices[:, 0:1, :, 0::3] # [1, 1, 1, max_triangles]
        p0_Y = transformed_vertices[:, 1:2, :, 0::3]
        p1_X = transformed_vertices[:, 0:1, :, 1::3]
        p1_Y = transformed_vertices[:, 1:2, :, 1::3]
        p2_X = transformed_vertices[:, 0:1, :, 2::3]
        p2_Y = transformed_vertices[:, 1:2, :, 2::3]

        # 深度Zも同様に3頂点の平均を一撃で計算 [1, 1, 1, max_triangles]
        z0 = transformed_vertices[:, 2:3, :, 0::3]
        z1 = transformed_vertices[:, 2:3, :, 1::3]
        z2 = transformed_vertices[:, 2:3, :, 2::3]
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
        mask_edge0 = torch.relu(edges0 * 100.0)
        mask_edge1 = torch.relu(edges1 * 100.0)
        mask_edge2 = torch.relu(edges2 * 100.0)
        
        # 形状: [1, max_triangles, H, W] (各ポリゴンごとのクッキリした三角形マスク)
        all_triangles_mask = torch.clamp(mask_edge0 * mask_edge1 * mask_edge2, min=0.0, max=1.0)

        # =========================================================================
        # ステップE: Zバッファ（深度隠面消去）＆一撃プレス
        # =========================================================================
        # 平均Z [1, 1, 1, max_triangles] を [1, max_triangles, 1, 1] に変換
        avg_z_t = avg_z.permute(0, 3, 1, 2)
        
        # 手前にある（Zが小さい）ポリゴンほど強い輝度ウェイトにするハック
        z_weight = torch.clamp(1.0 - (avg_z_t / 4.0), min=0.0, max=1.0)
        
        # ポリゴンごとの三角形マスクに、そのポリゴンの手前優先ウェイトを乗算
        weighted_triangles = all_triangles_mask * z_weight
        
        # 2000個のポリゴンの重なりを、チャンネル次元（max_triangles）からの
        # torch.max によって一撃で1枚の2D画面（モノクロ）へプレス！
        # 形状: [1, max_triangles, H, W] -> [1, 1, H, W]
        rendered_space, _ = torch.max(weighted_triangles, dim=1, keepdim=True)
        
        # 最後に3チャンネル（RGB）に拡張してフルカラーフレームバッファとして返却！
        framebuffer = rendered_space.repeat(1, 3, 1, 1)
        
        return framebuffer
