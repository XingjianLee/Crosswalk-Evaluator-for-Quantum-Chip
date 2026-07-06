import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# ==========================================
# 核心组件：直通估计器 (STE)
# ==========================================
class ClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        return torch.clamp(x, lo, hi)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None



# ==========================================
# 单比特门评估网络 (Single Gate)
# ==========================================
class GBFCN2_single_t3_new_global(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, xw_in_dim, adj_in_dim, global_in_dim, output=1):
        super().__init__()

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 1. 物理 & 拓扑分支
        self.fc_w_in = nn.Linear(xw_in_dim, hidden_dim)
        self.w_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

        self.fc_adj_in = nn.Linear(adj_in_dim, hidden_dim)
        self.adj_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # 2. 全局背景分支 (Information Bottleneck)
        global_hidden = hidden_dim // 2
        self.fc_global_in = nn.Linear(global_in_dim, global_hidden)
        self.global_layers = nn.ModuleList(
            [nn.Linear(global_hidden, global_hidden) for _ in range(max(1, num_layers - 1))])

        # 3. FiLM 调制层
        local_dim = hidden_dim * 2
        self.film_gamma = nn.Linear(global_hidden, local_dim)
        self.film_beta = nn.Linear(global_hidden, local_dim)

        # 4. SE 门控与输出层
        self.fc_fuse = nn.Linear(local_dim, hidden_dim)
        self.se_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        self.fc_out = nn.Linear(hidden_dim, output)

        # FiLM 初始化 (保证初始局部特征不被破坏)
        nn.init.constant_(self.film_gamma.weight, 0)
        nn.init.constant_(self.film_gamma.bias, 1.0)
        nn.init.constant_(self.film_beta.weight, 0)
        nn.init.constant_(self.film_beta.bias, 0)

    def forward(self, x):
        raw_dim = 1
        path_dim = 1
        dist_phy_dim = 3

        xw = x[:, :4]
        x_local = x[:, 4:9]
        feat_local = x[:, 9:9 + raw_dim * 2]
        path_density = x[:, 9 + raw_dim * 2:9 + raw_dim * 2 + path_dim]
        x_global = x[:, 9 + raw_dim * 2 + path_dim + dist_phy_dim:]

        if self.training:
            noise_scale = x_global.std() * 0.05
            # noise_scale = 0.02
            x_global = x_global + torch.randn_like(x_global) * noise_scale

        fi = xw[:, 0:2]
        fj = xw[:, 2:4]
        feat_i = feat_local[:, :raw_dim]
        feat_j = feat_local[:, raw_dim:]

        xw_feat = torch.cat([xw, fi - fj, (fi - fj).abs(), feat_i, feat_j], dim=1)
        xadj_feat = torch.cat([x_local, path_density], dim=1)

        h_w = torch.tanh(self.fc_w_in(xw_feat))
        for layer in self.w_layers:
            h_w = torch.tanh(layer(h_w)) + h_w

        h_adj = self.act(self.fc_adj_in(xadj_feat))
        for layer in self.adj_layers:
            h_adj = self.act(layer(h_adj))
            h_adj = self.dropout(h_adj)

        h_g = self.fc_global_in(x_global)
        for layer in self.global_layers:
            h_g = self.act(layer(h_g))

        # 核心：FiLM 调制
        h_local = torch.cat([h_w, h_adj], dim=1)
        gamma = self.film_gamma(h_g)
        beta = self.film_beta(h_g)
        h_modulated = h_local * gamma + beta

        # 核心：SE 通道门控特征选择
        h = torch.tanh(self.fc_fuse(h_modulated))
        gate = self.se_gate(h)
        h = h * gate

        raw_out = self.fc_out(h)
        raw_out = (torch.exp(raw_out + 1) - 5 - math.exp(1))

        # 严格截断并使用 STE 保留梯度
        out = ClampSTE.apply(raw_out, -5.0, -1.0)
        return out, raw_out


# ==========================================
# 双比特门评估网络 (iSWAP Gate)
# ==========================================
class GBFCN2_iSWAP_simple_AiHao_v2_global(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, xw_in_dim, adj_in_dim, global_in_dim, output=1):
        super().__init__()

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 1. 物理 & 拓扑分支
        self.fc_w_in = nn.Linear(xw_in_dim, hidden_dim)
        self.w_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

        self.fc_adj_in = nn.Linear(adj_in_dim, hidden_dim)
        self.adj_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # 2. 全局背景分支 (Information Bottleneck)
        global_hidden = hidden_dim // 2
        self.fc_global_in = nn.Linear(global_in_dim, global_hidden)
        global_layers_count = max(1, num_layers - 1)
        self.global_layers = nn.ModuleList(
            [nn.Linear(global_hidden, global_hidden) for _ in range(global_layers_count)])

        # 3. FiLM 调制层
        local_dim = hidden_dim * 2
        self.film_gamma = nn.Linear(global_hidden, local_dim)
        self.film_beta = nn.Linear(global_hidden, local_dim)

        # 4. SE 门控与输出层
        self.fc3 = nn.Linear(local_dim, hidden_dim)
        self.se_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        self.fc_out = nn.Linear(hidden_dim, output)

        nn.init.constant_(self.film_gamma.weight, 0)
        nn.init.constant_(self.film_gamma.bias, 1.0)
        nn.init.constant_(self.film_beta.weight, 0)
        nn.init.constant_(self.film_beta.bias, 0.0)

    def forward(self, x):
        dist_dim = 5
        raw_dim = 1
        path_dim = 1
        dist_phy_dim = 3
        dist_dim_co = 8 + 2 * dist_dim

        xw = x[:, :8]
        x_local = x[:, 8:dist_dim_co].clone()
        rw_i = x[:, dist_dim_co:dist_dim_co + raw_dim]
        rw_j = x[:, dist_dim_co + raw_dim:dist_dim_co + raw_dim * 2]
        rw_k = x[:, dist_dim_co + raw_dim * 2:dist_dim_co + raw_dim * 3]
        path_ik = x[:, dist_dim_co + raw_dim * 3:dist_dim_co + raw_dim * 3 + path_dim]
        path_jk = x[:, dist_dim_co + raw_dim * 3 + path_dim:dist_dim_co + raw_dim * 3 + path_dim * 2]
        x_global = x[:, dist_dim_co + raw_dim * 3 + path_dim * 2 + dist_phy_dim * 2:]

        if self.training:
            noise_scale = x_global.std() * 0.05
            # noise_scale = 0.02
            x_global = x_global + torch.randn_like(x_global) * noise_scale

        f_E = xw[:, :2]
        f_k = xw[:, 2:4]
        f_i = xw[:, 4:6]
        f_j = xw[:, 6:8]
        xw_delta1 = torch.cat([(f_E - f_j), (f_k - f_j), (f_i - f_j)], dim=1)
        xw_delta2 = torch.cat([(f_E - f_i), (f_k - f_i)], dim=1)
        xw_delta3 = torch.cat([(f_E - f_k)], dim=1)

        xw = torch.cat([xw, xw_delta1, xw_delta2, xw_delta3,
                        xw_delta1.abs(), xw_delta2.abs(), xw_delta3.abs(), rw_i, rw_j, rw_k], dim=1)
        xadj_feat = torch.cat([x_local, path_ik, path_jk], dim=1)

        h_w = torch.tanh(self.fc_w_in(xw))
        for layer in self.w_layers:
            h_w = torch.tanh(layer(h_w)) + h_w

        h_adj = self.act(self.fc_adj_in(xadj_feat))
        for layer in self.adj_layers:
            h_adj = self.act(layer(h_adj))
            h_adj = self.dropout(h_adj)

        h_g = self.fc_global_in(x_global)
        for layer in self.global_layers:
            h_g = self.act(layer(h_g))

        h_local = torch.cat([h_w, h_adj], dim=1)
        gamma = self.film_gamma(h_g)
        beta = self.film_beta(h_g)
        h_modulated = h_local * gamma + beta

        h = torch.tanh(self.fc3(h_modulated))
        gate = self.se_gate(h)
        h = h * gate

        raw_out = self.fc_out(h)
        raw_out = (torch.exp(raw_out + 1) - 5 - math.exp(1))

        out = ClampSTE.apply(raw_out, -5.0, -1.0)
        return out, raw_out