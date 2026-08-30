import torch
import matplotlib.pyplot as plt
from PreProcessor import ANE3DPreProcessor64

def debug_projection():
    print("🚀 デバッグログを開始します...")
    model = ANE3DPreProcessor64().eval()
    mvp_weights = torch.eye(4, dtype=torch.float16).unsqueeze(0).unsqueeze(0).expand(1, 64, 4, 4)
    # 💡 W座標(1.0)を含めた4次元の頂点データにする
    expanded_vertices = torch.zeros((1, 64, 4, 4), dtype=torch.float16)
    
    # X, Y, Z, W(1.0) を設定
    expanded_vertices[0, 0, 0, :] = torch.tensor([0.0, 1.0, -2.0, 1.0])
    expanded_vertices[0, 0, 1, :] = torch.tensor([-1.0, -1.0, -2.0, 1.0])
    expanded_vertices[0, 0, 2, :] = torch.tensor([1.0, -1.0, -2.0, 1.0])
    
    with torch.no_grad():
        transformed = torch.matmul(mvp_weights, expanded_vertices)
        print(f"[1] Transformed Shape: {transformed.shape}")
        print(f"    Transformed Max: {transformed.max().item()}, Min: {transformed.min().item()}")
        
    # 💡 XYZWが最後の次元にある場合
    X_c = transformed[:, :, :, 0:1]
    Y_c = transformed[:, :, :, 1:2]
    W_c = transformed[:, :, :, 3:4]
    
    safe_W = torch.relu(W_c) + torch.relu(-W_c) + 0.02
    print(f"[2] safe_W Max: {safe_W.max().item()}, Min: {safe_W.min().item()}")
    
    screen_x = X_c / safe_W
    screen_y = Y_c / safe_W
    
    x = screen_x[0, 0, :3, 0].numpy()
    y = screen_y[0, 0, :3, 0].numpy()
    
    print(f"[3] Screen X: {x}")
    print(f"[4] Screen Y: {y}")
    
    if torch.all(screen_x == 0) and torch.all(screen_y == 0):
        print("⚠️ 警告: すべての座標がゼロです。W除算か行列の掛け算に問題があります。")
    
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'r-')
        plt.plot([x[2], x[0]], [y[2], y[0]], 'r-')
        plt.scatter(x, y, color='blue')
        
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.grid(True)
        plt.savefig("debug_projection.png")
        print("📸 'debug_projection.png' を保存しました。")

if __name__ == "__main__":
    debug_projection()