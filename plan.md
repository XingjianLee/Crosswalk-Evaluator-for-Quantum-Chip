# 提升大芯片串扰预测效果的改进计划

本文给出后续改进路线，目标是在只使用小芯片训练数据的前提下，提高模型在 `5x5`、`6x6`、`8x8` 等更大芯片上的迁移预测效果。

核心原则：

```text
不要直接把整个芯片映射成一个总串扰 scalar。
继续采用：
局部串扰预测 -> qubit 级聚合 -> chip 级 ScoreTotal 聚合
```

原因是局部规律更有可能从小图迁移到大图，而整芯片总分会强烈依赖 qubit 数、边数、中心节点比例和可发生串扰的 pair 数。

## 1. 当前问题判断

当前代码已经采用了局部预测路线：

```text
X/Y:   ct_X(i -> j), ct_Y(i -> j)
iSWAP: ct_iSWAP((i,j) -> k)
```

但在迁移到大芯片时仍然有误差，主要原因可能包括：

1. 当前模型不是 GNN，没有真正 message passing。
2. 训练拓扑太少，局部环境覆盖不足。
3. `C_matrix` 没有作为输入。
4. 物理距离构造了但没有充分使用，也没有真正归一化。
5. 局部误差在 chip-level 聚合后被放大。
6. `5x5` 样本数太少，评估不稳定。

因此改进不应只理解为“把 MLP 换成更大 MLP”。更合理的路线是：

```text
先明确聚合指标
-> 增强现有 MLP baseline
-> 加入物理启发残差学习
-> 再做 local GNN
```

## 2. 总体路线

推荐保留三层预测与评估结构：

```text
第 1 层：local prediction
  X/Y:   source i -> victim j
  iSWAP: active edge (i,j) -> victim k

第 2 层：qubit-level aggregation
  每个 qubit 作为 victim 受到多少串扰
  每个 qubit 作为 source/edge endpoint 造成多少串扰

第 3 层：chip-level aggregation
  每个候选芯片一个 ScoreTotal
```

直接预测整芯片总分可以作为 baseline，但不建议作为主方案。

## 3. 第一阶段：改进现有 MLP Baseline

第一阶段不急着换成 GNN，而是让当前 MLP baseline 更严谨、更可解释。

### 3.1 增加 qubit-level 指标

当前 transfer 中主要有两类指标：

- 展开后的 local 指标。
- 聚合后的 chip-level 指标。

建议新增 qubit-level 指标，明确每个 qubit 的串扰来源。

对 X/Y：

```text
ct_single: (B, N, N)
ct_single[b, i, k] = source i 对 victim k 的串扰

victim_score[b, k] = sum_i ct_single[b, i, k]
source_score[b, i] = sum_k ct_single[b, i, k]
```

对 iSWAP：

```text
ct_iswap: (B, E, N)
ct_iswap[b, e, k] = active edge e 对 victim k 的串扰

victim_score[b, k] = sum_e ct_iswap[b, e, k]
edge_score[b, e] = sum_k ct_iswap[b, e, k]
```

这样可以分别观察：

- 哪些 qubit 容易受害。
- 哪些 source 或 active edge 容易造成串扰。
- qubit 级误差是否比 chip 级误差更稳定。

新增指标：

```text
local MAE / Spearman
qubit-level MAE / Spearman
chip-level MAE / MAE_r / Spearman
```

### 3.2 修正和使用物理距离特征

当前 `model2_GBFCNres_3_8.py` 里构造了：

```text
dist_phy
dist_diff
```

但注释说要归一化，代码实际没有归一化。

建议改成：

```text
dist_phy_norm = dist_phy / max_dist_in_current_graph
dist_dx_norm = dx / max(cols - 1, 1)
dist_dy_norm = dy / max(rows - 1, 1)
```

并确保这些距离特征进入 MLP 的局部拓扑分支，而不是被切片跳过。

### 3.3 使用 `C_matrix`

`C_matrix_FIN.pkl` 当前没有作为模型输入。建议先不要直接把整个矩阵输入模型，而是抽取局部特征。

对 X/Y 的 `(i, j)`：

```text
C_ij
abs(C_ij)
log(abs(C_ij) + eps)
row_norm_C_ij
```

对 iSWAP 的 `((i,j), k)`：

```text
C_ik
C_jk
C_ij
abs(C_ik) + abs(C_jk)
max(abs(C_ik), abs(C_jk))
```

如果考虑多跳路径，可加入：

```text
sum over paths: product(abs(C_edge_on_path))
```

第一版不需要复杂物理公式，只要把局部耦合强度作为特征加入即可。

### 3.4 增强局部结构特征

建议加入：

```text
source degree
victim degree
is_source_boundary
is_victim_boundary
is_source_corner
is_victim_corner
hop_distance
number_of_shortest_paths
local_clustering_like_feature
```

对大图迁移尤其重要的是区分：

```text
角点 qubit
边界 qubit
内部 qubit
```

小图中内部节点少，大图中内部节点多。如果模型不能识别这种结构角色，迁移会不稳定。

### 3.5 改进采样方式

当前训练中使用：

```python
min_data_size = min(d[0].shape[0] for d in embedded_datas)
```

这会把每个拓扑都截断到同样数量，可能浪费数据。

建议改成：

```text
每个拓扑单独形成 dataset
训练时按拓扑均衡采样 batch
每个 batch 中不同拓扑比例可控
```

例如：

```text
1/3 来自 3x3
1/3 来自 2x4
1/3 来自 1x6
```

这样既保留数据，又避免某个拓扑主导训练。

## 4. 第二阶段：物理启发特征 + MLP Residual

这一阶段的核心思想是：不要让神经网络从零学习全部规律，而是先构造一个粗糙但方向正确的 baseline，再让 MLP 学残差。

### 4.1 预测形式

把直接预测：

```text
pred_log_ct = neural_model(features)
```

改成：

```text
pred_log_ct = physics_baseline_log_ct + neural_residual(features)
```

其中：

```text
neural_residual = MLP(features)
```

### 4.2 粗 baseline 的特征

不需要一开始就写复杂物理公式，可以用工程近似：

```text
距离越远，串扰通常越小
耦合越强，串扰通常越大
参数差异越小，某些相互影响可能越强
多条短路径存在时，间接影响可能更强
```

对 X/Y：

```text
baseline_features(i, j):
  normalized_distance(i, j)
  hop_distance(i, j)
  abs(C_ij)
  abs(param_i - param_j)
  shortest_path_count(i, j)
```

对 iSWAP：

```text
baseline_features((i,j), k):
  normalized_distance(i, k)
  normalized_distance(j, k)
  hop_distance(i, k)
  hop_distance(j, k)
  abs(C_ik)
  abs(C_jk)
  abs(C_ij)
  abs(param_i - param_k)
  abs(param_j - param_k)
```

第一版 baseline 可以是线性模型或小 MLP，但建议它的输出单独可观察。

### 4.3 为什么 residual 更适合迁移

纯神经网络可能在小图上学到偶然相关性。Residual 方式把任务拆成：

```text
可解释的主趋势：由 baseline 负责
复杂的非线性修正：由 MLP 负责
```

这样当芯片从 `2x3`、`1x6` 迁移到 `5x5`、`6x6`、`8x8` 时，模型至少会保留一些单调趋势，例如距离和耦合强度的影响。

## 5. 第三阶段：Local GNN 方案

如果增强 MLP 和 residual baseline 仍然不够，下一步建议做 local GNN。

关键原则：

```text
不要做整图 GCN 直接预测芯片总分。
要做局部子图上的 source-target / edge-target 预测。
```

也就是说，GNN 的预测对象仍然是：

```text
X/Y:   ct(i -> j)
iSWAP: ct((i,j) -> k)
```

只是用 GNN 自动学习局部子图中的信息传播，而不是完全依赖手工特征。

## 6. 为什么不推荐普通 GCN 作为主模型

普通 GCN 的典型更新方式可以简化理解为：

```text
h_v <- average(h_neighbors)
```

它适合作为弱 baseline，但不适合作为本项目主模型。

主要问题：

1. 边特征利用弱。

普通 GCN 通常只看邻接矩阵，不方便表达：

```text
C_matrix
物理距离
耦合强度
边是否为 active edge
```

2. 角色区分能力弱。

本项目必须区分：

```text
source qubit
victim qubit
active edge endpoint
normal qubit
```

普通 GCN 如果不额外设计 role embedding，很容易把这些角色混掉。

3. 多路径信息容易被平均掉。

串扰可能与多条路径、间接耦合有关。简单邻居平均会损失路径差异。

4. 输出对象不是普通节点分类。

本项目不是预测每个节点类别，而是预测：

```text
pair-level: (i, j)
edge-target-level: ((i,j), k)
```

普通 GCN 需要大量额外读出设计才能适配。

结论：

```text
GCN 可以做 baseline，但不建议作为主线。
```

## 7. MPNN 方案

MPNN 是 Message Passing Neural Network。它的核心思想是：节点通过边向邻居发送消息，消息内容可以依赖边特征。

简化公式：

```text
message(u -> v) = MLP(h_u, h_v, edge_attr_uv)
h_v = update(h_v, aggregate(messages))
```

### 7.1 为什么 MPNN 适合本项目

本项目有大量重要边特征：

```text
是否相邻
hop distance
physical distance
C_matrix coupling
路径数量
active edge 标记
```

MPNN 可以让这些边特征直接参与消息计算。

相比当前 MLP：

```text
当前 MLP：先手工展开关系，再一次性预测
MPNN：在局部子图中多轮传播，自动组合邻居和路径信息
```

### 7.2 X/Y 的 MPNN 输入

目标：

```text
预测 ct_X(i -> j) 或 ct_Y(i -> j)
```

对每个 `(source=i, victim=j)`，抽取局部子图：

```text
subgraph = nodes within R hops from i or j
```

默认：

```text
R = 2
```

节点特征：

```text
state 参数 2 维
degree
是否角点
是否边界
是否 source
是否 victim
```

边特征：

```text
是否物理连接
物理距离
C_matrix 派生耦合强度
是否在 source-victim 短路径上
```

读出方式：

```text
z = concat(h_i, h_j, h_i - h_j, abs(h_i - h_j), mean_pool(h_subgraph), max_pool(h_subgraph))
pred = MLP(z)
```

### 7.3 iSWAP 的 MPNN 输入

目标：

```text
预测 ct_iSWAP((i,j) -> k)
```

对每个 `(active edge=(i,j), victim=k)`，抽取局部子图：

```text
subgraph = nodes within R hops from i, j, or k
```

节点角色：

```text
edge_endpoint_i
edge_endpoint_j
victim
normal
```

边角色：

```text
active_edge
normal_edge
```

读出方式：

```text
edge_embed_ij = MLP(concat(h_i, h_j, edge_attr_ij))
z = concat(h_i, h_j, h_k, edge_embed_ij, mean_pool(h_subgraph), max_pool(h_subgraph))
pred = MLP(z)
```

### 7.4 MPNN 默认配置

推荐第一版配置：

```text
R = 2
num_layers = 3
hidden_dim = 64
dropout = 0.1
readout = role embeddings + mean/max pooling
loss = Huber + rank loss + bound loss
output = log10 crosstalk in [-5, -1]
```

后续 ablation：

```text
R = 1, 2, 3
hidden_dim = 32, 64, 128
num_layers = 2, 3, 4
```

## 8. GINE 方案

GINE 是 GIN 的 edge-feature 版本。它比普通 GCN 更适合当前任务，因为它能显式使用边特征。

简化更新公式：

```text
h_v = MLP((1 + eps) * h_v + sum_u ReLU(h_u + edge_encoder(edge_attr_uv)))
```

### 8.1 为什么推荐 GINE 作为第一版 GNN

GINE 是建议优先实现的 GNN baseline。

原因：

1. 能使用边特征。
2. 结构比 edge-conditioned GNN 简单。
3. 参数量可控。
4. 比 Graph Transformer 更不容易在小数据上过拟合。
5. 可以快速验证 message passing 是否真的改善 5x5 迁移。

### 8.2 GINE 的输入设计

节点特征：

```text
state_0
state_1
degree
is_corner
is_boundary
role_source
role_victim
role_edge_endpoint_i
role_edge_endpoint_j
```

边特征：

```text
adjacency
normalized_physical_distance
abs(C_uv)
log(abs(C_uv) + eps)
is_active_edge
is_on_shortest_path
```

不同任务使用不同 role 标记：

```text
X/Y: source, victim
iSWAP: endpoint_i, endpoint_j, victim, active_edge
```

### 8.3 GINE 的输出设计

X/Y：

```text
z_single = concat(h_source, h_victim, h_source - h_victim, abs(h_source - h_victim), pool)
pred_log_ct = MLP(z_single)
```

iSWAP：

```text
h_edge = MLP(concat(h_i, h_j, edge_attr_ij))
z_iswap = concat(h_i, h_j, h_victim, h_edge, pool)
pred_log_ct = MLP(z_iswap)
```

### 8.4 GINE 默认训练设置

```text
num_layers = 3
hidden_dim = 64
dropout = 0.1
batch_size = 512
loss_func = Huber
rank_max_weight = 0.3
rank_warmup = 20
bound_weight = 0.01
```

为了公平比较，先尽量沿用当前 MLP 的训练超参数。

### 8.5 GINE 的成功标准

GINE 不一定要显著降低所有局部 MAE，最关键是改善大图聚合后的稳定性。

重点观察：

```text
5x5 local Spearman 是否不下降
5x5 qubit-level MAE 是否下降
5x5 ScoreTotal MAE_r 是否下降
5x5 ScoreTotal Spearman 是否上升或更稳定
```

## 9. Edge-conditioned GNN 方案

Edge-conditioned GNN 的核心是让边特征决定消息变换。

简化公式：

```text
message(u -> v) = W(edge_attr_uv) * h_u
```

其中：

```text
W(edge_attr_uv)
```

不是固定矩阵，而是由边特征通过一个小网络生成。

### 9.1 为什么它可能有效

串扰很可能与不同边的物理耦合强度有关。Edge-conditioned GNN 可以表达：

```text
强耦合边传递更强消息
远距离边传递更弱消息
active edge 对周围节点影响更大
```

这比 GINE 更灵活。

### 9.2 适合加入的边特征

```text
edge_attr_uv:
  adjacency
  normalized_physical_distance
  abs(C_uv)
  log(abs(C_uv) + eps)
  hop_distance
  is_active_edge
  is_boundary_edge
```

### 9.3 风险

Edge-conditioned GNN 的主要风险是过拟合。

原因：

- 参数更多。
- 小图数据少。
- 训练拓扑少。
- 如果 `C_matrix` 噪声较大，模型可能学到不稳定关系。

因此不建议作为第一个 GNN 版本。推荐顺序是：

```text
先做 GINE
如果 GINE 有提升，再做 edge-conditioned GNN
```

### 9.4 默认设计

为了控制参数量，第一版 edge-conditioned GNN 不直接生成完整 `hidden_dim x hidden_dim` 矩阵，而是生成门控向量：

```text
gate_uv = sigmoid(MLP(edge_attr_uv))
message(u -> v) = gate_uv * linear(h_u)
```

这样既利用边特征，又避免参数爆炸。

## 10. Graph Transformer 方案

Graph Transformer 使用 attention 在局部子图内部建模节点间影响。

建议只做 local Graph Transformer：

```text
对每个 source-target 或 edge-target 样本抽取 R-hop 子图
只在这个局部子图内做 attention
不对整个 8x8 芯片做全局 attention
```

### 10.1 基本形式

节点 token：

```text
每个 qubit 是一个 token
```

attention bias：

```text
attention_bias(u, v) = f(hop_distance, physical_distance, C_uv, role_u, role_v)
```

模型输出仍然是：

```text
X/Y:   ct(i -> j)
iSWAP: ct((i,j) -> k)
```

### 10.2 优点

Graph Transformer 可能更擅长：

- 多路径影响。
- 多个邻近节点共同作用。
- 大图中心区域的复杂局部环境。
- 非局部但仍在 R-hop 范围内的相互影响。

### 10.3 风险

Graph Transformer 风险更高：

1. 数据少时容易过拟合。
2. 超参数更多。
3. 对 attention bias 设计敏感。
4. 训练成本高于 GINE/MPNN。
5. 5x5 样本少时，很难判断提升是否真实。

因此它不适合作为第一版主线。

### 10.4 推荐定位

推荐把 Graph Transformer 放在后期做 ablation：

```text
MLP baseline
MLP residual
GINE
edge-conditioned GNN
local Graph Transformer
```

如果 GINE 已经无法继续提升，再尝试 Graph Transformer。

## 11. 推荐实现顺序

### Step 1：补充评估指标

先不改模型，只新增评估逻辑：

```text
local MAE / Spearman
qubit-level MAE / Spearman
ScoreTotal MAE / MAE_r / Spearman
```

这样后续每次改模型都能定位问题：

```text
是局部预测差？
还是 qubit 聚合差？
还是 chip 聚合后误差放大？
```

### Step 2：增强 MLP 特征

加入：

```text
normalized distance
C_matrix derived features
degree / boundary / corner
shortest path count
```

并修复物理距离没有真正归一化的问题。

### Step 3：做 MLP residual

实现：

```text
pred_log_ct = baseline_log_ct + residual_mlp(features)
```

比较它与当前 MLP 的差异。

### Step 4：实现 local GINE

先用纯 PyTorch 实现，不立即引入 `torch_geometric`。

理由：

- 当前环境是 Mac + conda DL。
- 项目计算量不大。
- 避免因为 PyG 安装问题打断实验。

### Step 5：实现 edge-conditioned GNN

如果 GINE 有稳定提升，再尝试边条件消息传递。

第一版采用 gate 形式：

```text
message = sigmoid(MLP(edge_attr)) * Linear(h_source)
```

### Step 6：尝试 local Graph Transformer

只作为后期实验，不作为第一版核心改造。

## 12. 实验设计

训练拓扑保持小图：

```text
1x6
2x4
3x3
```

迁移测试：

```text
3x4
3x5
4x4
3x6
4x5
4x6
5x5
```

每个模型都报告：

```text
local MAE
local Spearman
qubit-level MAE
qubit-level Spearman
ScoreTotal MAE
ScoreTotal MAE_r
ScoreTotal Spearman
```

重点观察：

```text
5x5 ScoreTotal MAE_r
5x5 ScoreTotal Spearman
5x5 qubit-level MAE
```

同时必须注明：

```text
5x5 样本数少，chip-level Spearman 统计不稳定
```

## 13. 依赖策略

默认不新增依赖。

第一版 GINE/MPNN 建议用纯 PyTorch 实现。原因：

- 当前项目已经只依赖 PyTorch、NumPy、SciPy、Matplotlib 等基础库。
- 纯 PyTorch 更容易在 Mac CPU 环境复现。
- 避免安装 `torch_geometric` 及其扩展包带来的环境问题。

如果后续确实要使用 PyG，需要单独确认：

```text
torch 版本
Python 版本
Apple Silicon 兼容性
CPU wheel 是否可用
```

不能直接盲目安装。

## 14. 最终建议

建议优先级：

```text
1. qubit-level 指标与聚合分析
2. 修复距离特征并加入 C_matrix 特征
3. MLP residual baseline
4. local GINE
5. edge-conditioned GNN
6. local Graph Transformer
```

这条路线的好处是每一步都有明确对照实验，不会一开始就陷入复杂 GNN 调参。

如果目标是科研复现和论文式表达，最重要的是证明：

```text
模型学到的是可迁移的局部规律，而不是记住小图拓扑。
```

因此所有改进都应该围绕：

```text
local prediction 是否稳
qubit aggregation 是否合理
chip-level ranking 是否可靠
```

三层指标展开。
