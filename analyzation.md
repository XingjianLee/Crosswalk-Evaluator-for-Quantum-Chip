# 当前 Evaluator 代码解析

本文面向“复现、读懂、后续修改”三个目标，解释当前项目中四个核心文件的作用和数据流：

- `fit_SmallGraphs_single.py`
- `fit_SmallGraphs_iSWAP.py`
- `model/model2_GBFCNres_3_8.py`
- `model/MLP_NEW_F.py`

项目整体目标不是直接替代完整物理仿真，而是从小芯片仿真数据中学习局部串扰规律，再迁移到更大的芯片拓扑上，辅助快速评估候选芯片参数。

## 1. 总体结论

当前代码采用的是“局部串扰预测，再聚合”的路线，而不是直接输入整个芯片、输出一个总串扰值。

对 X/Y 单比特门：

```text
预测对象：ct_X(i -> j), ct_Y(i -> j)
含义：source qubit i 执行 X/Y 门时，对 victim qubit j 造成的局部串扰
```

对 iSWAP 双比特门：

```text
预测对象：ct_iSWAP((i,j) -> k)
含义：active edge (i,j) 执行 iSWAP 门时，对 victim qubit k 造成的局部串扰
```

所以当前代码并不是直接预测“每个 qubit 一个 X/Y/iSWAP 串扰值”。如果需要 qubit 级结果，应该从局部预测矩阵继续聚合得到，例如：

```text
某个 qubit k 作为 victim 受到的 X 串扰 = sum_i ct_X(i -> k)
某个 qubit k 作为 victim 受到的 iSWAP 串扰 = sum_edge ct_iSWAP(edge -> k)
```

这条路线适合迁移学习，因为局部关系比整芯片总分更容易从小图迁移到大图。

## 2. 四个核心文件职责

### 2.1 `fit_SmallGraphs_single.py`

这个文件负责训练和评估 X/Y 单比特门的局部串扰模型。默认参数里 `gate_type='X'`，所以直接运行时训练 X gate；Y gate 有单独脚本 `fit_SmallGraphs_single_Y.py`，结构基本一致。

主要流程：

1. 读取小图训练数据：
   - `Random_0_9_3x3bit_seed0_step02_560um`
   - `Random_0_8_2x4bit_seed0_step02_560um`
   - `Random_0_6_1x6bit_seed0_step02_560um`
2. 读取每个图的：
   - `state_FIN.pkl`
   - `cross_talk_X_FIN.pkl` 或 `cross_talk_Y_FIN.pkl`
3. 用 `bw_DataEmbedder` 把 `(B, N, N)` 标签展开为多个 `(source, victim)` 局部样本。
4. 拼接图谱特征 `get_spectral_features(adj)`。
5. 用 `GBFCN2_single_t3_new_global` 训练 MLP。
6. 在更大拓扑上做 zero-shot transfer，例如 `3x4`、`4x5`、`5x5`。

训练集只来自小图，但迁移评估会加载更大图数据。

### 2.2 `fit_SmallGraphs_iSWAP.py`

这个文件负责训练和评估 iSWAP 双比特门的局部串扰模型。

iSWAP 与 X/Y 的关键区别是：iSWAP 的 active object 不是单个 qubit，而是一条物理连接边 `(i,j)`。

主要流程：

1. 读取小图训练数据：
   - `state_FIN.pkl`
   - `score_node_FIN.pkl`
   - `cross_talk_iSWAP_FIN.pkl`
2. 用邻接矩阵找出所有物理边。
3. 对每条 active edge 和每个 victim qubit 生成训练样本。
4. 模型预测：

```text
ct_iSWAP((i,j) -> k)
```

其中 `(i,j)` 是执行 iSWAP 的边，`k` 是被影响的 qubit。

### 2.3 `model/model2_GBFCNres_3_8.py`

这个文件最关键的类是 `bw_DataEmbedder`。它不是神经网络，而是数据展开器。

它负责把原始芯片数据转换成 MLP 可吃的二维训练样本：

```text
原始芯片级 batch 数据
-> 局部 pair / edge-target 样本
-> 每个样本对应一个局部串扰标签
```

核心职责：

- 根据邻接矩阵构造 1 到 5 跳的拓扑关系。
- 构造节点对关系特征。
- 把 X/Y 的 `(B, N, N)` 数据展开成 `(B * 有效节点对数, feature_dim)`。
- 把 iSWAP 的 `(B, E, N)` 数据展开成 `(B * 有效 edge-target 数, feature_dim)`。
- 把模型输出从 log 空间还原回原始串扰矩阵。

### 2.4 `model/MLP_NEW_F.py`

这个文件定义真正的神经网络。

虽然类名里有 `GBFCN`，但当前实现不是 GNN，也没有图上的 message passing。它本质是：

```text
手工构造局部特征 + 多分支 MLP
```

它包含两个模型：

- `GBFCN2_single_t3_new_global`：用于 X/Y 单比特门。
- `GBFCN2_iSWAP_simple_AiHao_v2_global`：用于 iSWAP 双比特门。

两个模型都包含：

- 物理参数分支。
- 拓扑关系分支。
- 全局谱特征分支。
- FiLM 调制。
- SE gate 通道选择。
- `ClampSTE` 输出截断。

## 3. 数据文件与 shape

每个数据目录大致包含：

```text
state_FIN.pkl
score_node_FIN.pkl
cross_talk_X_FIN.pkl
cross_talk_Y_FIN.pkl
cross_talk_iSWAP_FIN.pkl
C_matrix_FIN.pkl
```

### 3.1 `state_FIN.pkl`

`state_FIN.pkl` 是每个候选芯片的 qubit 参数：

```text
shape: (B, N, 2)
```

含义：

- `B`：候选样本数量。
- `N`：qubit 数量。
- `2`：每个 qubit 的两个设计参数。

在训练前，代码会把这两个参数从 `[-8, 8]` 线性归一化到 `[0, 1]`：

```python
normalize_ops(data_in, [-8, -8], [8, 8])
```

### 3.2 `cross_talk_X_FIN.pkl` 和 `cross_talk_Y_FIN.pkl`

在 X/Y 脚本中，标签会被裁成：

```python
cross_talk_X_FIN[:, :, :N]
cross_talk_Y_FIN[:, :, :N]
```

然后被当成：

```text
shape: (B, N, N)
```

理解方式：

```text
data_out[b, i, j] = 第 b 个候选芯片中，source i 对 victim j 的 X/Y 串扰
```

训练时不会直接把整个 `(N, N)` 矩阵作为一个样本，而是展开成多个局部样本。

### 3.3 `cross_talk_iSWAP_FIN.pkl`

iSWAP 标签会被解释成：

```text
shape: (B, E, N)
```

其中：

- `B`：候选芯片数量。
- `E`：芯片中的物理连接边数量。
- `N`：victim qubit 数量。

理解方式：

```text
data_out[b, e, k] = 第 b 个候选芯片中，第 e 条 active edge 执行 iSWAP 时，对 qubit k 的串扰
```

### 3.4 `score_node_FIN.pkl`

`score_node_FIN.pkl` 只在 iSWAP 脚本中使用：

```text
shape: (B, N, 1)
```

代码用它对 active edge 两端 qubit 的参数做加权融合：

```text
edge_feat = p_i * Xi + p_j * Xj
```

可以把它理解为 iSWAP 中“边两端 qubit 对当前边门贡献大小”的辅助权重。

### 3.5 `C_matrix_FIN.pkl`

`C_matrix_FIN.pkl` 是电容矩阵相关数据。当前 README 也说明它本轮训练脚本中不直接作为模型输入。

这是一项重要局限：串扰和耦合强度通常有关，而 `C_matrix` 很可能包含有用的物理耦合信息。后续改进时应优先考虑使用它或其派生特征。

## 4. X/Y 单比特门的数据展开

X/Y 的核心函数是：

```python
bw_datain_to_modelin_single_gate(data_in)
bw_dataout_to_modelout_single_gate(data_out)
```

### 4.1 输入展开

原始输入：

```text
data_in: (B, N, 2)
```

代码构造所有节点对 `(i, j)` 的特征：

```text
model_in_NxN[b, i, j, :]
```

特征包括：

- qubit `i` 的参数。
- qubit `j` 的参数。
- `i` 和 `j` 的拓扑关系。
- 局部结构特征。

然后用 `adj_total` 过滤，只保留 1 到 5 跳内的有效节点对。

最终输出：

```text
embedded_in: (B * 有效节点对数量, feature_dim)
```

每一行就是一个局部训练样本：

```text
(source i, victim j)
```

### 4.2 标签展开

原始标签：

```text
data_out: (B, N, N)
```

标签同样按有效节点对过滤，得到：

```text
embedded_out: (B * 有效节点对数量, 1)
```

并转成 log 空间：

```python
model_out = torch.log10(model_out.clamp(1e-5, 1e-1))
```

所以模型训练目标不是原始线性串扰，而是：

```text
log10(crosstalk)
```

数值范围大致被压到：

```text
[-5, -1]
```

### 4.3 X/Y 的语义

X/Y 当前训练目标可以写成：

```text
f(state_i, state_j, relation_i_j, graph_feature) -> log10(ct_X(i -> j))
```

或：

```text
f(state_i, state_j, relation_i_j, graph_feature) -> log10(ct_Y(i -> j))
```

因此它确实是在学习：

```text
每个 qubit 发起 X/Y 操作时，对其他 qubit 造成的局部串扰
```

不过它不是直接输出每个 qubit 的总串扰，而是输出更细粒度的 pair-level 串扰。

## 5. iSWAP 双比特门的数据展开

iSWAP 的核心函数是：

```python
bw_datain_to_modelin_iSWAP(data_in_Q, data_in_E)
bw_dataout_to_modelout_iSWAP(data_out)
```

### 5.1 active edge

iSWAP 是双比特门，所以 active object 是一条物理边：

```text
edge = (i, j)
```

代码通过邻接矩阵上三角部分找出所有物理边：

```python
idx_i_all, idx_j_all = torch.triu_indices(N, N, offset=1)
edge_mask = (self.adj_mat[idx_i_all, idx_j_all] != 0)
```

每条边对应一个可执行 iSWAP 的 qubit pair。

### 5.2 输入展开

对每个候选芯片、每条 active edge、每个 victim qubit，代码构造：

```text
(active edge=(i,j), victim=k)
```

输入特征包括：

- active edge 两端 qubit 的参数 `Xi`, `Xj`。
- victim qubit 的参数 `Xk`。
- 由 `score_node` 加权得到的 `edge_feat`。
- `i -> k` 的拓扑关系。
- `j -> k` 的拓扑关系。
- 距离、路径、节点结构等手工特征。

最终得到：

```text
embedded_in: (B * 有效 edge-target 数量, feature_dim)
```

### 5.3 标签展开

原始标签被解释为：

```text
data_out: (B, E, N)
```

展开后得到：

```text
embedded_out: (B * 有效 edge-target 数量, 1)
```

同样转成：

```text
log10(crosstalk)
```

### 5.4 iSWAP 的语义

iSWAP 当前训练目标可以写成：

```text
f(state_i, state_j, state_k, relation_i_k, relation_j_k, graph_feature)
-> log10(ct_iSWAP((i,j) -> k))
```

这不是“某个 qubit 发起 iSWAP”，而是：

```text
某条物理连接边执行 iSWAP，对某个 victim qubit 造成的串扰
```

这个定义比“每个 qubit 参与 iSWAP”更精确，因为 iSWAP 是边级 gate。

## 6. 当前 MLP 模型结构

`MLP_NEW_F.py` 中的模型不是普通单层 MLP，而是多个分支组合。

### 6.1 `ClampSTE`

`ClampSTE` 的作用是前向传播时截断输出，反向传播时保留梯度：

```python
out = ClampSTE.apply(raw_out, -5.0, -1.0)
```

含义：

```text
预测的 log10(crosstalk) 被限制在 [-5, -1]
```

这与标签预处理中的：

```python
torch.log10(model_out.clamp(1e-5, 1e-1))
```

是对应的。

优点是防止模型输出离谱值。风险是如果真实串扰超过这个范围，会被截断，模型无法表达更大的变化。

### 6.2 X/Y 模型

`GBFCN2_single_t3_new_global` 的输入被拆成：

- `xw`：qubit i 和 qubit j 的参数。
- `x_local`：1 到 5 跳拓扑关系。
- `feat_local`：局部结构特征。
- `path_density`：路径冗余度。
- `x_global`：整图 spectral feature。

模型分支：

```text
物理参数分支 h_w
拓扑关系分支 h_adj
全局图特征分支 h_g
```

融合方式：

1. `h_w` 和 `h_adj` 拼接成局部表示。
2. 用 `h_g` 生成 FiLM 的 `gamma` 和 `beta`。
3. 用 FiLM 调制局部表示。
4. 用 SE gate 做通道选择。
5. 输出 `raw_out`。
6. Clamp 到 `[-5, -1]`。

### 6.3 iSWAP 模型

`GBFCN2_iSWAP_simple_AiHao_v2_global` 与 X/Y 模型类似，但输入语义更复杂：

- `f_E`：active edge 融合特征。
- `f_i`、`f_j`：active edge 两端 qubit 参数。
- `f_k`：victim qubit 参数。
- `path_ik`、`path_jk`：edge 两端到 victim 的路径特征。
- `x_global`：整图 spectral feature。

它会构造多组差分特征：

```text
f_E - f_j
f_k - f_j
f_i - f_j
f_E - f_i
f_k - f_i
f_E - f_k
```

这些差分特征可以理解为让模型显式看到不同 qubit 参数之间的相对关系。

## 7. 训练损失和指标

### 7.1 基础损失

代码支持：

```text
HuberLoss
MSELoss
```

默认通常使用 HuberLoss。HuberLoss 对离群值比 MSE 稍稳健。

### 7.2 Ranking loss

`rank_loss` 是 pairwise ranking loss。它不只要求数值接近，也希望预测结果的相对顺序与真实结果一致。

这对芯片设计很重要，因为 Evaluator 的实际用途往往是排序候选方案：

```text
哪个候选芯片串扰更低？
哪个候选参数更值得进一步仿真？
```

### 7.3 Bound loss

`loss_bound` 惩罚 `raw_out` 超出 `[-5, -1]` 的部分：

```python
F.relu(raw_out - (-1.0)).mean() + F.relu(-5.0 - raw_out).mean()
```

它配合 `ClampSTE`，让模型在训练时也感知输出范围约束。

### 7.4 局部指标

训练和 transfer 中的：

```text
MSE
MAE
Spearman
Pearson
```

默认是对展开后的局部样本计算，也就是 pair-level 或 edge-target-level。

例如 X/Y 中：

```text
embedded_out vs pred
```

这衡量的是：

```text
局部 ct(i -> j) 是否预测准确
```

### 7.5 聚合指标

在 transfer 中，代码会把模型输出还原为原始矩阵，再做：

```python
Single_ct.sum(-1).mean(-1)
iSWAP_ct.sum(-1).mean(-1)
```

对 X/Y：

```text
Single_ct: (B, N, N)
Single_ct.sum(-1): 对 victim 维度求和
Single_ct.sum(-1).mean(-1): 再对 source 维度平均
```

对 iSWAP：

```text
iSWAP_ct: (B, E, N)
iSWAP_ct.sum(-1): 对 victim 维度求和
iSWAP_ct.sum(-1).mean(-1): 再对 active edge 维度平均
```

这会得到每个候选芯片一个聚合分数，用来衡量 chip-level 排序效果。

需要注意：当前聚合方式更像整体平均分，不是显式 qubit-level victim 聚合。如果后续要做“每个 qubit 的串扰值”，建议新增更清楚的聚合定义。

## 8. 当前代码的主要局限

### 8.1 当前不是 GNN

虽然代码处理了图结构，也构造了多跳邻接关系，但模型本身没有 message passing。

也就是说，模型不是这样工作：

```text
节点之间迭代传递信息 -> 得到节点 embedding -> 预测串扰
```

而是：

```text
提前把局部关系手工展开成特征 -> MLP 预测
```

这对小数据友好，但对大图泛化能力有限。

### 8.2 物理距离特征使用不充分

`model2_GBFCNres_3_8.py` 中构造了：

```text
dist_phy
dist_diff
```

但在 `MLP_NEW_F.py` 的 forward 中，这部分特征没有作为主要局部分支使用，部分维度被跳过后直接进入 `x_global` 之后的切片。

此外，代码注释里提到物理距离应该归一化，但实际写的是：

```python
dist_phy = dist_phy
```

这意味着 5x5、6x6、8x8 上距离数值范围会比小图更大，可能带来分布偏移。

### 8.3 `C_matrix` 没有使用

`C_matrix_FIN.pkl` 当前没有进入训练输入。

如果它确实代表电容矩阵或耦合相关信息，那么它可能直接影响串扰强度。后续模型应考虑：

```text
C_ij
|C_ij|
归一化 C_ij
多跳路径上的 C 乘积或和
```

### 8.4 训练拓扑覆盖不足

当前 X/Y 和 iSWAP 训练默认只用：

```text
3x3, 2x4, 1x6
```

这些小图很难覆盖 5x5 中大量出现的中心节点和更复杂局部路径结构。

虽然模型希望学习局部规律，但训练数据中的局部环境仍然太少。

### 8.5 `min_data_size` 截断可能浪费数据

训练时会取：

```python
min_data_size = min(d[0].shape[0] for d in embedded_datas)
```

然后每个图都只保留前 `min_data_size` 条局部样本。

这样做可以平衡不同拓扑，但会丢掉大量数据，也可能引入顺序偏差。更好的做法是按图构建 dataset，再在 DataLoader 中做均衡采样。

### 8.6 聚合误差会放大

局部预测的 MAE 和 Spearman 可能看起来还可以，但当大量局部预测加总成 chip-level 分数时，小误差会累积。

附件运行结果中，X gate 在 5x5 上局部 Spearman 仍较高，但聚合后的 `all_Sp` 下降，`all_MAE_r` 明显增大。这说明主要问题不是局部模型完全失效，而是：

```text
局部误差在大图聚合时被放大
```

### 8.7 5x5 样本数少

当前 5x5 数据样本很少，chip-level Spearman 和 MAE_r 的统计稳定性不足。

因此评价 5x5 效果时应同时报告：

- local 指标。
- qubit-level 指标。
- chip-level 指标。
- 样本数。

不能只看一个总 Spearman。

## 9. 给后续修改者的建议

如果后续要改模型，建议按以下顺序理解和动手：

1. 先跑通 X gate 的 `fit_SmallGraphs_single.py`，确认训练和 transfer 输出。
2. 理解 `bw_DataEmbedder` 如何把 `(B, N, N)` 展开成 pair-level 样本。
3. 再理解 iSWAP 的 `(B, E, N)` 展开。
4. 先新增 qubit-level 聚合指标，不急着改网络。
5. 再修复和增强特征，例如物理距离、`C_matrix`、边界/中心标识。
6. 最后再考虑 GNN 或 Graph Transformer。

最重要的是保持任务定义清晰：

```text
不要直接混淆 local、qubit-level、chip-level 三种指标。
```

当前代码的主线应该保留：

```text
局部 source-target / edge-target 串扰预测
-> qubit 级聚合
-> chip 级聚合
```
