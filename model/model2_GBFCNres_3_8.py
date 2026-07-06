import torch
import math

class bw_DataEmbedder:
    def __init__(self, adj_mat, row_col, dev=torch.device('cpu'), dtype=torch.float32) -> None:
        self.adj_mat = adj_mat.to(dev).to(dtype)
        self.dev = dev
        self.dtype = dtype

        # [新增] 1. 预计算结构特征 (Degree & Path Density)
        self.feat_i_mat, self.feat_j_mat, self.path_diffusivity, self.dist_phy, self.dist_diff = self._get_structural_feats(row_col)

        self.adj1, self.adj2, self.adj3, self.adj4, self.adj5 = self._get_adjs()

        # [修改] node_relation 维度将从 5 变为 8
        self.node_relation = self._get_node_relation()      # (node_num, node_num, 8)

        # other used variables
        self.Q_num = self.adj_mat.shape[0]
        self.adj_total = self.adj1 + self.adj2 + self.adj3 + self.adj4 + self.adj5
        self.adj_total[range(self.Q_num), range(self.Q_num)] = 0
        self.adj_total[self.adj_total > 0] = 1

        # for iSWAP gate
        self.E_num = round(self.adj_mat.sum().item() / 2)
        # [说明] ExN_relations 会自动包含新的结构特征，因为它是基于 self.node_relation 生成的
        self.ExN_relation_i, self.ExN_relation_j = self._get_ExN_relations()
        self.ExN_adj_total = self._get_ExN_adj_total()


    @torch.no_grad()
    def _get_structural_feats(self, row_col):
        """
        参数:
            rows, cols: 芯片布局的行数和列数 (例如 5, 5)
        返回:
            feat_i: 节点 i 的相对邻居度数 (N, N, 1)
            feat_j: 节点 j 的相对邻居度数 (N, N, 1)
            redundancy: 路径冗余度 (N, N, 1)
            dist_phy: 物理欧氏距离 (N, N, 1)
        """

        N = self.adj_mat.shape[0]
        adj = self.adj_mat

        # --- 1. 邻域度数特征 (原有：区分边缘/中心) ---
        degree = adj.sum(dim=1)
        neighbor_avg_deg = torch.mv(adj, degree) / (degree + 1e-6)
        neighbor_avg_deg = neighbor_avg_deg / (neighbor_avg_deg.mean() + 1e-6)

        feat_i = neighbor_avg_deg.reshape(N, 1, 1).repeat(1, N, 1)
        feat_j = neighbor_avg_deg.reshape(1, N, 1).repeat(N, 1, 1)

        # --- 2. 路径冗余度 (原有：描述拓扑连通强度) ---
        adj2 = torch.matmul(adj, adj)
        adj3 = torch.matmul(adj2, adj)
        redundancy = torch.log1p(adj2 + adj3).unsqueeze(-1)

        if row_col[0] == '6_n1t':
            # 对应第一张图：十字星型
            # 0是在2上方，5是在2下方，1-2-3-4是水平线
            coords = [
                [1, 0],  # 节点 0
                [0, 1],  # 节点 1
                [1, 1],  # 节点 2 (中心)
                [2, 1],  # 节点 3
                [3, 1],  # 节点 4
                [1, 2]  # 节点 5
            ]
            coords = torch.tensor(coords, device=self.dev, dtype=self.dtype)
        elif row_col[0] == '6_n1t_chain':
            # 对应第二张图：L型链
            # 0-1-2-3 水平, 3-4-5 垂直
            coords = [
                [0, 0],  # 节点 0
                [1, 0],  # 节点 1
                [2, 0],  # 节点 2
                [3, 0],  # 节点 3 (转折点)
                [3, 1],  # 节点 4
                [3, 2]  # 节点 5
            ]
            coords = torch.tensor(coords, device=self.dev, dtype=self.dtype)
        else:
            rows, cols = row_col[0], row_col[1]
            # --- 3. 新增：物理欧氏距离 (Physical Euclidean Distance) ---
            # 生成网格坐标 (x, y)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(rows, device=self.dev),
                torch.arange(cols, device=self.dev),
                indexing='ij'
            )
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(self.dtype)  # (N, 2)

        # 计算所有节点对之间的物理距离: sqrt((xi-xj)^2 + (yi-yj)^2)
        dist_diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 2)
        dist_phy = torch.norm(dist_diff, p=2, dim=-1)  # (N, N)

        # 归一化：物理距离如果不归一化，在大图（5x5）中数值会远超小图（2x3）
        # 建议除以当前图中最大的物理距离，使其保持在 [0, 1] 之间，增强迁移性
        dist_phy = dist_phy
        dist_phy = dist_phy.unsqueeze(-1)  # (N, N, 1)

        # 返回拼接后的特征 (N, N, 4)
        # 注意：在 forward 中拼接 xadj_feat 时需注意维度变化
        return feat_i, feat_j, redundancy, dist_phy, dist_diff



    @torch.no_grad()
    def _get_adjs(self):
        adj1 = self.adj_mat
        adj2 = torch.matmul(adj1, adj1)
        adj3 = torch.matmul(adj2, adj1)
        adj4 = torch.matmul(adj3, adj1)
        adj5 = torch.matmul(adj4, adj1)

        Q_num = adj1.shape[0]
        adj2[range(Q_num), range(Q_num)] = 0
        adj3[range(Q_num), range(Q_num)] = 0
        adj4[range(Q_num), range(Q_num)] = 0
        adj5[range(Q_num), range(Q_num)] = 0

        adj2[adj2 > 0] = 1
        adj3[adj3 > 0] = 1
        adj4[adj4 > 0] = 1
        adj5[adj5 > 0] = 1

        adj2[(adj1 - 1).abs() < 1e-5] = 0
        adj3[(adj1 - 1).abs() < 1e-5] = 0
        adj4[(adj1 - 1).abs() < 1e-5] = 0
        adj5[(adj1 - 1).abs() < 1e-5] = 0
        adj3[(adj2 - 1).abs() < 1e-5] = 0
        adj4[(adj2 - 1).abs() < 1e-5] = 0
        adj5[(adj2 - 1).abs() < 1e-5] = 0
        adj4[(adj3 - 1).abs() < 1e-5] = 0
        adj5[(adj3 - 1).abs() < 1e-5] = 0
        adj5[(adj4 - 1).abs() < 1e-5] = 0

        return adj1, adj2, adj3, adj4, adj5
    
    @torch.no_grad()
    def _get_node_relation(self):
        return torch.cat([
            torch.stack([self.adj1, self.adj2, self.adj3, self.adj4, self.adj5], dim=2),
            self.feat_i_mat,
            self.feat_j_mat,
            self.path_diffusivity,
            self.dist_phy,
            self.dist_diff,
        ], dim=2)
    
    ############################################################
    ################ single qubit gate #########################
    ############################################################

    def bw_datain_to_modelin_single_gate(self, data_in):
        assert data_in.dim() == 3
        B = data_in.shape[0]
        N = self.Q_num
        X = data_in.shape[2]
        node_feats = data_in
        REL_DIM = 11
        C = 2 * X + REL_DIM
        model_in_NxN = torch.zeros((B, N, N, C), dtype=self.dtype, device=self.dev)

        # fill node i features into channels [:X]
        model_in_NxN[:, :, :, :X] = node_feats.reshape(B, N, 1, X).repeat(1, 1, N, 1)
        # fill node j features into channels [X:2X]
        model_in_NxN[:, :, :, X:2 * X] = node_feats.reshape(B, 1, N, X).repeat(1, N, 1, 1)
        # fill relation channels [2X:2X+5]
        model_in_NxN[:, :, :, 2 * X:] = self.node_relation.reshape(1, N, N, REL_DIM).repeat(B, 1, 1, 1)

        # reshape to (B, N*N, C) and mask by adj_total
        model_in = model_in_NxN.reshape(B, N * N, C)
        model_in = model_in[:, (self.adj_total.reshape(-1) - 1).abs() < 1e-5]

        # flatten batches to (B*E, C)
        model_in = model_in.reshape(-1, C)
        return model_in

    @torch.no_grad()
    def bw_dataout_to_modelout_single_gate(self, data_out):
        assert data_out.dim() == 3, f"the dataout dim is {data_out.dim()}, while the required dim is 3"
        assert data_out.shape[1] == data_out.shape[2] == self.Q_num, f"the dataout shape is {data_out.shape}, while the Q_num is {self.Q_num}"

        bw_size = data_out.shape[0]
        model_out_NxN = data_out.reshape(bw_size, self.Q_num, self.Q_num, 1)
        model_out = model_out_NxN.reshape(bw_size, self.Q_num * self.Q_num, 1)
        model_out = model_out[:, (self.adj_total.reshape(-1) - 1).abs() < 1e-5]

        model_out = torch.log10(model_out.clamp(1e-5, 1e-1))

        model_out = model_out.reshape(-1, 1)
        return model_out
    
    @torch.no_grad()
    def bw_modelout_to_dataout_single_gate(self, model_out):
        assert model_out.dim() == 2, f"the modelout dim is {model_out.dim()}, while the required dim is 2"
        assert model_out.shape[1] == 1, f"the modelout shape is {model_out.shape}, while the required shape is (?, 1)"

        adj_total_edge_num = ((self.adj_total.reshape(-1) - 1).abs() < 1e-5).to(self.dtype).sum().round().int().item()
        model_out = model_out.reshape(-1, adj_total_edge_num)

        data_out = torch.zeros((model_out.shape[0], self.Q_num * self.Q_num), dtype=self.dtype, device=self.dev)
        data_out[:, (self.adj_total.reshape(-1) - 1).abs() < 1e-5] = 10 ** model_out
        data_out = data_out.reshape(-1, self.Q_num, self.Q_num)
        return data_out
    
    ############################################################
    ################ iSWAP gate ################################
    ############################################################

    @staticmethod
    def bwe_loader(adj_mat, bw_N_X):
        assert adj_mat.dim() == 2, f"the adj_mat dim is {adj_mat.dim()}, while the required dim is 2"
        assert adj_mat.shape[0] == adj_mat.shape[1], f"the adj_mat shape is {adj_mat.shape}, while the required shape is (N, N)"
        assert bw_N_X.dim() == 3, f"the bw_N_X dim is {bw_N_X.dim()}, while the required dim is 3"
        assert bw_N_X.shape[1] == adj_mat.shape[0], f"the bw_N_X shape is {bw_N_X.shape}, while the adj_mat shape is {adj_mat.shape}"

        bw_size = bw_N_X.shape[0]
        N = bw_N_X.shape[1]
        X = bw_N_X.shape[2]

        bw_adj = adj_mat.reshape(1, N, N).repeat(bw_size, 1, 1)
        bw_adj_triu = torch.triu(bw_adj, diagonal=1)            # (bw_size, N, N)

        bw_N_N_X_i = bw_N_X.reshape(bw_size, N, 1, X).repeat(1, 1, N, 1)
        bw_N_N_X_j = bw_N_X.reshape(bw_size, 1, N, X).repeat(1, N, 1, 1)

        bwe_X_i = bw_N_N_X_i.reshape(-1, X)
        bwe_X_j = bw_N_N_X_j.reshape(-1, X)
        bwe_X_i = bwe_X_i[(bw_adj_triu.reshape(-1) - 1).abs() < 1e-5]
        bwe_X_j = bwe_X_j[(bw_adj_triu.reshape(-1) - 1).abs() < 1e-5]

        bwe_X = torch.cat([bwe_X_i, bwe_X_j], dim=1)
        return bwe_X
    
    @torch.no_grad()
    def _get_ExN_relations(self):
        feat_dim = self.node_relation.shape[2]  # 8
        ExN_relations_list = [self.bwe_loader(self.adj_mat, self.node_relation[:, :, i].reshape(1, self.Q_num, self.Q_num))
                              for i in range(feat_dim)]     # [(E, 2N), ...]
        
        ExN_relations_i_list = [ExN_relations_list[i][:, :self.Q_num].reshape(-1, self.Q_num, 1) for i in range(feat_dim)]    # [(E, N, 1), ...]
        ExN_relations_j_list = [ExN_relations_list[i][:, self.Q_num:].reshape(-1, self.Q_num, 1) for i in range(feat_dim)]    # [(E, N, 1), ...]

        ExN_relation_i = torch.cat(ExN_relations_i_list, dim=2)     # (E, N, 8)
        ExN_relation_j = torch.cat(ExN_relations_j_list, dim=2)     # (E, N, 8)
        return ExN_relation_i, ExN_relation_j
    
    @torch.no_grad()
    def _get_ExN_adj_total(self):
        ExN_adj_total_ij = self.bwe_loader(self.adj_mat, self.adj_total.reshape(1, self.Q_num, self.Q_num))    # (E, 2N)

        ExN_adj_total_i = ExN_adj_total_ij[:, :self.Q_num]   # (E, N), 0 or 1
        ExN_adj_total_j = ExN_adj_total_ij[:, self.Q_num:]   # (E, N), 0 or 1

        ExN_adj_total = ExN_adj_total_i * ExN_adj_total_j   # (E, N), 0 or 1
        return ExN_adj_total


    def bw_datain_to_modelin_iSWAP(self, data_in_Q, data_in_E):
        """
        双比特门特征提取改进版：特征融合与去重
        输出维度: 从 40 降维到 29 (更紧凑，迁移性更好)
        """
        device = self.dev
        dtype = self.dtype
        B, N, _ = data_in_Q.shape

        # ... (前处理代码保持不变，计算 idx, Xi, Xj, edge_feat 等) ...
        # Copy start
        idx_i_all, idx_j_all = torch.triu_indices(N, N, offset=1, device=self.adj_mat.device)
        edge_mask = (self.adj_mat[idx_i_all, idx_j_all] != 0)
        idx_i = idx_i_all[edge_mask]
        idx_j = idx_j_all[edge_mask]
        E = idx_i.shape[0]

        node_feats = data_in_Q.to(device=device, dtype=dtype)
        node_scores = data_in_E.reshape(B, N).to(device, dtype)

        Xi = node_feats[:, idx_i, :]
        Xj = node_feats[:, idx_j, :]

        si = node_scores[:, idx_i]
        sj = node_scores[:, idx_j]

        edge_scores = torch.stack([si, sj], dim=-1)
        edge_probs = torch.softmax(edge_scores, dim=-1)

        p_i = edge_probs[..., 0].unsqueeze(-1)
        p_j = edge_probs[..., 1].unsqueeze(-1)

        # 1. 门的物理参数特征 (加权融合)
        edge_feat = p_i * Xi + p_j * Xj  # (B,E,2)

        Xk = node_feats.unsqueeze(1).expand(-1, E, -1, -1)  # (B,E,N,2)
        edge_feat_exp = edge_feat.unsqueeze(2).expand(-1, -1, N, -1)  # (B,E,N,2)

        # Xi, Xj 原始特征虽然有冗余，但保留作为参考
        Xi_exp = Xi.unsqueeze(2).expand(-1, -1, N, -1)
        Xj_exp = Xj.unsqueeze(2).expand(-1, -1, N, -1)

        # 2. 获取原始关系特征 (B, E, N, 16)
        # 假设 16维 = [Dist(5), RW_Src(5), RW_Tgt(5), Path(1)]
        rel_i = self.ExN_relation_i.to(device=device, dtype=dtype)
        rel_j = self.ExN_relation_j.to(device=device, dtype=dtype)

        rel_i_batch = rel_i.unsqueeze(0).expand(B, -1, -1, -1)
        rel_j_batch = rel_j.unsqueeze(0).expand(B, -1, -1, -1)

        # A. 拆分特征通道 (假设前5维是距离，中间5维是源RW，再5维是目标RW，最后1维是Path)
        dist_dim = 5
        rw_dim = 1

        # i -> k 的特征
        dist_i = rel_i_batch[..., :dist_dim]  # 距离
        rw_i = rel_i_batch[..., dist_dim: dist_dim + rw_dim]  # i 的结构 (Source)
        rw_k = rel_i_batch[..., dist_dim + rw_dim: dist_dim + rw_dim+rw_dim]  # k 的结构 (Target)
        path_ik = rel_i_batch[..., dist_dim + rw_dim+rw_dim:dist_dim + rw_dim+rw_dim+rw_dim]  # i-k 路径
        dist_phy_ik = rel_i_batch[..., dist_dim + rw_dim+rw_dim+rw_dim:]

        # j -> k 的特征
        dist_j = rel_j_batch[..., :dist_dim]
        rw_j = rel_j_batch[..., dist_dim: dist_dim + rw_dim]  # j 的结构 (Source)
        # rw_k_dup = rel_j_batch[..., dist_dim+rw_dim : -1]     # 重复的 k 结构，扔掉
        path_jk = rel_j_batch[..., dist_dim + rw_dim+rw_dim:dist_dim + rw_dim+rw_dim+rw_dim]
        dist_phy_jk = rel_j_batch[..., dist_dim + rw_dim+rw_dim+rw_dim:]

        model_in_ExN = torch.cat([
            edge_feat_exp,  # 2
            Xk,  # 2
            Xi_exp,  # 2
            Xj_exp,  # 2
            dist_i,  # 5 (保留几何形状)
            dist_j,  # 5 (保留几何形状)
            rw_i,
            rw_j,
            rw_k,  # 5 (受害点结构，只放一次)
            path_ik,
            path_jk,
            dist_phy_ik,
            dist_phy_jk,
        ], dim=-1)

        # ... (后续 Reshape 保持不变，注意调整 reshape 的维度为 29) ...
        # 如果你没改 Xi, Xj 那里，维度就是 29。
        # Copy end

        total_dim = model_in_ExN.shape[-1]  # 自动获取维度，防止算错
        model_in = model_in_ExN.reshape(B, self.E_num * self.Q_num, total_dim)
        model_in = model_in[:, (self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5]
        model_in = model_in.reshape(-1, total_dim)
        return model_in



    @torch.no_grad()
    def bw_dataout_to_modelout_iSWAP(self, data_out):
        assert data_out.dim() == 3, f"the dataout dim is {data_out.dim()}, while the required dim is 3"
        assert data_out.shape[1] == self.E_num, f"the dataout shape is {data_out.shape}, while the E_num is {self.E_num}"
        assert data_out.shape[2] == self.Q_num, f"the dataout shape is {data_out.shape}, while the Q_num is {self.Q_num}"

        bw_size = data_out.shape[0]
        model_out_ExN = data_out.reshape(bw_size, self.E_num, self.Q_num, 1)
        model_out = model_out_ExN.reshape(bw_size, self.E_num * self.Q_num, 1)
        model_out = model_out[:, (self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5]

        model_out = torch.log10(model_out.clamp(1e-5, 1e-1))

        model_out = model_out.reshape(-1, 1)
        return model_out
    
    @torch.no_grad()
    def bw_modelout_to_dataout_iSWAP(self, model_out):
        assert model_out.dim() == 2, f"the modelout dim is {model_out.dim()}, while the required dim is 2"
        assert model_out.shape[1] == 1, f"the modelout shape is {model_out.shape}, while the required shape is (?, 1)"

        ExN_adj_total_edge_num = ((self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5).to(self.dtype).sum().round().int().item()
        model_out = model_out.reshape(-1, ExN_adj_total_edge_num)

        data_out = torch.zeros((model_out.shape[0], self.E_num * self.Q_num), dtype=self.dtype, device=self.dev)
        data_out[:, (self.ExN_adj_total.reshape(-1) - 1).abs() < 1e-5] = 10 ** model_out - 1e-5
        data_out = data_out.reshape(-1, self.E_num, self.Q_num)
        return data_out
