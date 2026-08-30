import torch
import matplotlib.pyplot as plt
from PreProcessor import ANE3DPreProcessor64

def debug_projection():
    model = ANE3DPreProcessor64().eval()
    
    # ダミーデータの作成
    expanded_vertices = torch.zeros((1, 64, 4, 3), dtype=torch.float16)
    mvp_weights = torch.eye(4, dtype=torch.float16).unsqueeze(0).unsqueeze(0).expand(1, 64, 4, 4)
    colors_r = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
    colors_g = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
    colors_b = torch.zeros((1, 64, 1, 1), dtype=torch.float16)
    
    # 簡単な三角形の頂点を設定
    expanded_vertices[0, 0, 0, :] = torch.tensor([0.0, 1.0, -2.0])
    expanded_vertices[0, 0, 1, :] = torch.tensor([-1.0, -1.0, -2.0])
    expanded_vertices[0, 0, 2, :] = torch.tensor([1.0, -1.0, -2.0])
    
    with torch.no_grad():
        # PreProcessorの内部計算を再現して座標を取得
        transformed = torch.matmul(mvp_weights, expanded_vertices)
        X_c = transformed[:, :, 0:1, :].transpose(2, 3)
        Y_c = transformed[:, :, 1:2, :].transpose(2, 3)
        W_c = transformed[:, :, 3:4, :].transpose(2, 3)
        
        safe_W = torch.relu(W_c) + torch.relu(-W_c) + 0.02
        screen_x = X_c / safe_W
        screen_y = Y_c / safe_W
        
        # プロット用のデータ抽出
        x = screen_x[0, 0, :3, 0].numpy()
        y = screen_y[0, 0, :3, 0].numpy()
        
        # プロットの作成
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'r-') # 三角形の辺
        plt.plot([x[2], x[0]], [y[2], y[0]], 'r-') # 閉じる
        plt.scatter(x, y, color='blue') # 頂点
        
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.grid(True)
        plt.title("Projected Triangle")
        
        # 画像として保存
        plt.savefig("debug_projection.png")
        print("📸 'debug_projection.png' を保存しました。")

if __name__ == "__main__":
    debug_projection()