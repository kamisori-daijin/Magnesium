import torch
import torch.nn as nn

class ANEDynamicRayTracingCore(nn.Module):
    def __init__(self, width=256, height=256, max_steps=64, grid_size=8):
        super().__init__()
        self.w, self.h = width, height
        self.max_steps = max_steps
        self.dt = 0.05
        self.grid_size = grid_size
        
        # 基本バッファの初期化
        self.register_buffer("eps", torch.tensor([[[[0.02]]]]).half())
        self.register_buffer("floor_y", torch.tensor([[[[-0.8]]]]).half())
        
        y_grid = torch.linspace(1.0, -1.0, self.h).view(1, 1, self.h, 1)
        x_grid = torch.linspace(-1.0, 1.0, self.w).view(1, 1, 1, self.w)
        
        self.register_buffer("cam_dx", x_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dy", y_grid.expand(1, 1, self.h, self.w).half())
        self.register_buffer("cam_dz", torch.full((1, 1, self.h, self.w), -1.0).half())
        
        self.register_buffer("step_ratios", (torch.arange(self.max_steps).view(1, self.max_steps, 1, 1) * self.dt).half())

        # ボクセル座標グリッド
        coords = torch.linspace(-1.0, 1.0, self.grid_size).half()
        self.register_buffer("grid_x", coords.view(1, 1, 1, 1, 1, self.grid_size))
        self.register_buffer("grid_y", coords.view(1, 1, 1, 1, self.grid_size, 1))
        self.register_buffer("grid_z", coords.view(1, 1, 1, self.grid_size, 1, 1))

        # ANE用 累積和 Conv2d
        self.ane_cumsum_conv = nn.Conv2d(self.max_steps, self.max_steps, kernel_size=1, bias=False)
        weight_matrix = torch.tril(torch.ones(self.max_steps, self.max_steps))
        self.ane_cumsum_conv.weight.data = weight_matrix.view(self.max_steps, self.max_steps, 1, 1).half()
        self.ane_cumsum_conv.weight.requires_grad = False

    def check_voxel_hit(self, voxel_data, px, py, pz):
        eps = 2.0 / self.grid_size
        
        # px, py, pz の次元を [1, max_steps, H, W, 1, 1, 1] に拡張して、
        # grid_x, grid_y, grid_z と正しくブロードキャストされるようにします
        px_exp = px.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        py_exp = py.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pz_exp = pz.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # mask_x は [..., 1, 1, X], mask_y は [..., 1, Y, 1], mask_z は [..., Z, 1, 1] になります
        mask_x = torch.relu(1.0 - torch.abs(px_exp - self.grid_x) / eps)
        mask_y = torch.relu(1.0 - torch.abs(py_exp - self.grid_y) / eps)
        mask_z = torch.relu(1.0 - torch.abs(pz_exp - self.grid_z) / eps)
        
        # これで 8x8x8 (あるいは設定した解像度) の空間に正しく展開されます
        full_mask = mask_x * mask_y * mask_z

        # voxel_data をブロードキャスト用に次元拡張
        masked_data = voxel_data.unsqueeze(1) * full_mask
        
        # reduce_sum でボクセル抽出
        hit_result = torch.sum(masked_data, dim=(-3, -2, -1), keepdim=False)
        return hit_result

    def forward(self, voxel_data, inv_view_matrix_64d):
        def get_mat_val(mat, idx): return mat[:, idx:idx+1, :, :]

        # カメラ行列の展開
        r00, r01, r02 = get_mat_val(inv_view_matrix_64d, 0), get_mat_val(inv_view_matrix_64d, 1), get_mat_val(inv_view_matrix_64d, 2)
        r10, r11, r12 = get_mat_val(inv_view_matrix_64d, 4), get_mat_val(inv_view_matrix_64d, 5), get_mat_val(inv_view_matrix_64d, 6)
        r20, r21, r22 = get_mat_val(inv_view_matrix_64d, 8), get_mat_val(inv_view_matrix_64d, 9), get_mat_val(inv_view_matrix_64d, 10)

        dx = r00 * self.cam_dx + r01 * self.cam_dy + r02 * self.cam_dz
        dy = r10 * self.cam_dx + r11 * self.cam_dy + r12 * self.cam_dz
        dz = r20 * self.cam_dx + r21 * self.cam_dy + r22 * self.cam_dz

        inv_len = torch.rsqrt(dx*dx + dy*dy + dz*dz + 1e-5)
        init_dx, init_dy, init_dz = dx * inv_len, dy * inv_len, dz * inv_len
        init_px, init_py, init_pz = get_mat_val(inv_view_matrix_64d, 3), get_mat_val(inv_view_matrix_64d, 7), get_mat_val(inv_view_matrix_64d, 11)

        # レイの進行
        px_all = init_px + init_dx * self.step_ratios
        py_all = init_py + init_dy * self.step_ratios
        pz_all = init_pz + init_dz * self.step_ratios

        # 動的ボクセルとの交差判定
        object_hit_all = self.check_voxel_hit(voxel_data, px_all, py_all, pz_all)
        
        floor_hit_all = torch.clamp(torch.relu(self.floor_y - py_all) * 100.0, min=0.0, max=1.0)
        any_hit_all = torch.clamp(object_hit_all + floor_hit_all, min=0.0, max=1.0)

        # ANE Cumsum
        cum_hit = self.ane_cumsum_conv(any_hit_all)
        prior_hit = torch.cat([torch.zeros_like(cum_hit[:, :1, :, :]), cum_hit[:, :-1, :, :]], dim=1)
        not_hit_yet_all = torch.clamp(1.0 - prior_hit, min=0.0, max=1.0)

        is_first_object_all = not_hit_yet_all * object_hit_all
        hit_object_mask = torch.sum(is_first_object_all, dim=1, keepdim=True).clamp(0.0, 1.0)

        return hit_object_mask