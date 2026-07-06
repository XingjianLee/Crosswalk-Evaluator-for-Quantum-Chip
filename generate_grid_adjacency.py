import numpy as np


def generate_grid_adj(n, m, return_tensor=True, device='cpu'):
    """
    生成 n(行) * m(列) 网格的邻接矩阵 (Nearest-Neighbor, 4联通)
    参数:
        n (int): 行数
        m (int): 列数
        return_tensor (bool): 是否返回 PyTorch Tensor，否则返回 Numpy Array
        device (str): 如果返回 Tensor，指定设备 ('cpu' 或 'cuda')
    返回:
        adj_matrix: 形状为 (n*m, n*m) 的邻接矩阵
    """
    num_nodes = n * m
    # 初始化全 0 矩阵
    adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    for r in range(n):
        for c in range(m):
            # 将二维坐标 (r, c) 映射为一维索引 i
            current_idx = r * m + c
            # 1. 检查下方邻居 (Down)
            if r + 1 < n:
                down_idx = (r + 1) * m + c
                adj[current_idx, down_idx] = 1
                adj[down_idx, current_idx] = 1
            # 2. 检查右侧邻居 (Right)
            if c + 1 < m:
                right_idx = r * m + (c + 1)
                adj[current_idx, right_idx] = 1
                adj[right_idx, current_idx] = 1
    return adj.tolist()



# ================= 使用示例 =================
# if __name__ == '__main__':
#     rows = 1
#     cols = 4
#     # 1. 生成矩阵
#     adj = generate_grid_adjacency(rows, cols, device='cpu')
#     print(adj)