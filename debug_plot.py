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
        outputs = model(expanded_vertices, mvp_weights, colors_r, colors_g, colors_b)
        
    # A0, B0, C0 などのエッジ係数から座標を逆算するか、
    # PreProcessorの内部変数を出力するように一時的に変更してプロットします
    print("デバッグ実行が完了しました。")
    print(f"出力テンソル数: {len(outputs)}")

if __name__ == "__main__":
    debug_projection()