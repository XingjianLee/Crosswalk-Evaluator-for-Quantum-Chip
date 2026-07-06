# Transferable Crosstalk Evaluator for Superconducting-Chip Design

本项目用于训练和评估一个面向超导芯片参数设计的 **Evaluator**。Evaluator 的目标不是替代完整电磁仿真或系统动力学仿真，而是在已有仿真数据上学习局部串扰规律，并在未见过的大拓扑上快速预测串扰排序，为后续 Designer 搜索低串扰芯片参数提供代理评估器。

项目核心验证目标有三层：

1. **Local source-target / edge-target 指标收敛**：X/Y 单比特门的 source-target 局部串扰，以及 iSWAP 双比特门的 edge-target 局部串扰在未见拓扑上仍能保持排序相关性，说明模型学习到的是可迁移的局部物理规律，而不是只记住训练拓扑。这里主要看 **Spearman** 是否稳定/收敛，同时必须报告 **MAE** 来确认局部数值误差没有失控。
2. **ScoreTotal 指标收敛**：将 X、Y、iSWAP 的局部预测值从 log 空间还原到线性串扰空间，并在候选样本级别大量聚合后，总串扰分数仍能保持较好的排序相关性，说明局部误差在全局聚合后没有破坏设计排序。这里仍以 **ScoreTotal Spearman** 作为候选排序主指标，但 **ScoreTotal MAE / MAE_r** 同样重要，用于衡量总分绝对误差和相对误差规模。
3. **Designer_sample 只做 external holdout**：Designer 生成的样本只作为最终外部测试集，不参与训练、validation、checkpoint 选择或模型选择，用于验证 Evaluator 对真实设计器候选的泛化能力。Designer_sample 上也需要同时报告 Spearman 和 MAE，不能只报告排序相关性。

---

## 1. 项目结构

```text
evaluator/
├── Datasets_FIN/
│   ├── Random_0_6_1x6bit_seed0_step02_560um/
│   ├── Random_0_6_2x3bit_seed0_step02_560um/
│   ├── Random_0_8_2x4bit_seed0_step02_560um/
│   ├── Random_0_9_3x3bit_seed0_step02_560um/
│   ├── Random_0_12_3x4bit_seed0_step02_560um/
│   ├── Random_0_15_3x5bit_seed0_step02_560um/
│   ├── Random_0_16_4x4bit_seed0_step02_560um/
│   ├── Random_0_18_3x6bit_seed0_step02_560um/
│   ├── Random_0_20_4x5bit_seed0_step02_560um/
│   ├── Random_0_24_4x6bit_seed0_step02_560um/
│   └── Random_0_25_5x5bit_seed0_step02_560um/
│
├── model/
│   ├── MLP_NEW_F.py
│   └── model2_GBFCNres_3_8.py
│
├── run_X/                 # 已训练好的 X gate checkpoint
├── run_Y/                 # 已训练好的 Y gate checkpoint
├── run_iSWAP/             # 已训练好的 iSWAP checkpoint
│
├── fit_SmallGraphs_single.py      # X gate 局部评估器训练 + zero-shot transfer
├── fit_SmallGraphs_single_Y.py    # Y gate 局部评估器训练 + zero-shot transfer
├── fit_SmallGraphs_iSWAP.py       # iSWAP gate 局部评估器训练 + zero-shot transfer
├── fit_SmallGraphs_total.py       # 加载 X/Y/iSWAP checkpoint，评估 ScoreTotal transfer
├── generate_grid_adjacency.py     # 生成规则网格邻接矩阵
├── utils.py                       # pkl 读写工具
└── requirements.txt
```

### 每个数据目录包含

```text
state_FIN.pkl              # 候选芯片节点参数，shape ≈ (sample_num, qubit_num, 2)
score_node_FIN.pkl         # iSWAP 源端相关节点权重，shape ≈ (sample_num, qubit_num, 1)
cross_talk_X_FIN.pkl       # X gate 串扰标签
cross_talk_Y_FIN.pkl       # Y gate 串扰标签
cross_talk_iSWAP_FIN.pkl   # iSWAP gate 串扰标签
C_matrix_FIN.pkl           # 电容矩阵相关数据，本轮训练脚本中不直接作为模型输入
```

---

## 2. 推荐运行环境

代码默认使用 GPU：

```python
dev = torch.device('cuda:0')
```

因此推荐使用带 CUDA 的 PyTorch 环境。建议使用 Python 3.10 或 3.11，并在项目根目录 `evaluator/` 下运行所有命令。

### 2.1 创建环境

```bash
conda create -n chip_evaluator python=3.10 -y
conda activate chip_evaluator
```

### 2.2 安装依赖

项目中提供了 `requirements.txt`，但其中 PyTorch 的 CUDA wheel 可能需要按本机 CUDA 版本单独安装。推荐先安装 PyTorch，再安装其余依赖。

```bash
# 1) 根据本机 CUDA 版本安装 torch；如果使用 CUDA 11.8，可参考如下形式
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2) 安装常用科学计算依赖
pip install numpy pandas scipy matplotlib seaborn
```

如果严格复现实验环境，也可以尝试：

```bash
pip install -r requirements.txt
```

如果 `requirements.txt` 中某些版本在当前机器上不可用，可以使用兼容版本。该项目核心依赖是：

```text
python
numpy
pandas
scipy
matplotlib
seaborn
torch
```

### 2.3 没有 GPU 时如何运行

四个主脚本中都写了：

```python
dev = torch.device('cuda:0')
```

如果只想在 CPU 上调试，需要手动改为：

```python
dev = torch.device('cpu')
```

需要修改的文件包括：

```text
fit_SmallGraphs_single.py
fit_SmallGraphs_single_Y.py
fit_SmallGraphs_iSWAP.py
fit_SmallGraphs_total.py
```

CPU 可以用于小规模调试，但完整评估可能较慢。

---

## 3. 最快跑通方式：直接使用随包 checkpoint 评估 ScoreTotal

压缩包中已经包含训练好的 checkpoint：

```text
run_X/.../model_X_150.pt
run_Y/.../model_Y_150.pt
run_iSWAP/.../model_iSWAP_150.pt
```

因此不重新训练也可以直接跑总分迁移评估：

```bash
cd evaluator
python fit_SmallGraphs_total.py
```

默认会加载：

```text
run_iSWAP/bs_512_lr0.001_hid32_lyr2_dp0.1_loss_huber_rank0.3_warm20_bnd0.01/model_iSWAP_150.pt
run_X/bs_512_lr0.001_hid32_lyr2_dp0.2_loss_huber_rank0.3_warm20_bnd0.01/model_X_150.pt
run_Y/bs_512_lr0.001_hid32_lyr2_dp0.2_loss_huber_rank0.3_warm20_bnd0.01/model_Y_150.pt
```

默认评估的 zero-shot transfer 拓扑为：

```text
3x4, 3x5, 4x4, 3x6, 4x5, 4x6, 5x5
```

运行结束后会打印每个拓扑的总分指标，并默认保存：

```text
transfer_total_score_results.pkl
```

也可以指定保存路径：

```bash
python fit_SmallGraphs_total.py --save_path results/transfer_total_score_results.pkl
```

如果保存目录不存在，需要先创建：

```bash
mkdir -p results
```

---

## 4. 从头训练三个局部 Evaluator

如果需要重新训练 X、Y、iSWAP 三个局部模型，可以依次运行：

```bash
cd evaluator

# X gate local evaluator
python fit_SmallGraphs_single.py \
  --gate_type X \
  --epochs 151 \
  --bw_size 512 \
  --lr 0.001 \
  --hidden_dim 32 \
  --num_layers 2 \
  --dropout 0.2 \
  --loss_func huber \
  --rank_max_weight 0.3 \
  --rank_warmup 20 \
  --bound_weight 0.01

# Y gate local evaluator
python fit_SmallGraphs_single_Y.py \
  --gate_type Y \
  --epochs 151 \
  --bw_size 512 \
  --lr 0.001 \
  --hidden_dim 32 \
  --num_layers 2 \
  --dropout 0.2 \
  --loss_func huber \
  --rank_max_weight 0.3 \
  --rank_warmup 20 \
  --bound_weight 0.01

# iSWAP gate local evaluator
python fit_SmallGraphs_iSWAP.py \
  --gate_type iSWAP \
  --epochs 151 \
  --bw_size 512 \
  --lr 0.001 \
  --hidden_dim 32 \
  --num_layers 2 \
  --dropout 0.1 \
  --loss_func huber \
  --rank_max_weight 0.3 \
  --rank_warmup 20 \
  --bound_weight 0.01
```

训练完成后，再运行总分评估：

```bash
python fit_SmallGraphs_total.py --epochs 151
```

### 重要：`epochs` 与 checkpoint 后缀的关系

训练循环是：

```python
for epoch in range(arg.epochs):
```

并且每 10 个 epoch 保存一次模型：

```python
if epoch % 10 == 0:
    torch.save(... model_{gate_type}_{epoch}.pt)
```

默认 `--epochs 151` 时，最后一个 epoch 是 `150`，因此会保存：

```text
model_X_150.pt
model_Y_150.pt
model_iSWAP_150.pt
```

`fit_SmallGraphs_total.py` 默认会寻找：

```text
model_*_{epochs-1}.pt
```

所以推荐使用 `151 / 201 / 251 / 301` 这类 `10k+1` 的 epoch 设置。不要直接设置 `--epochs 150`，否则脚本会寻找 `model_*_149.pt`，但训练脚本并不会保存这个文件。

---

## 5. 训练/测试拓扑划分

当前代码中的局部训练脚本使用三个小拓扑训练：

```text
3x3: Random_0_9_3x3bit_seed0_step02_560um
2x4: Random_0_8_2x4bit_seed0_step02_560um
1x6: Random_0_6_1x6bit_seed0_step02_560um
```

每个训练拓扑内部按样本顺序划分：

```text
train : valid : test = 0.8 : 0.1 : 0.1
```

用于 zero-shot transfer 的未见拓扑为：

```text
3x4, 3x5, 4x4, 3x6, 4x5, 4x6, 5x5
```

`Datasets_FIN` 中还包含 `2x3` 数据，但当前训练脚本没有使用它。若后续要加入训练或测试，需要手动修改脚本中的 `graph_names`、`graph_adj_mats` 和 `graph_row_col`。

---

## 6. 模型输入与输出逻辑

### 6.1 输入变量

每个候选芯片的核心设计变量来自：

```text
state_FIN.pkl      -> ops-like node features, shape: (B, N, 2)
score_node_FIN.pkl -> iSWAP source-node score, shape: (B, N, 1)
```

代码中会把 `state_FIN.pkl` 从原始范围 `[-8, 8]` 归一化到 `[0, 1]`：

```python
normalize_ops(data_in, [-8, -8], [8, 8])
```

### 6.2 局部关系特征

`model/model2_GBFCNres_3_8.py` 中的 `bw_DataEmbedder` 会把每个候选芯片展开成局部预测样本。

单比特门 X/Y：

```text
source qubit i  -> target qubit j
```

iSWAP 双比特门：

```text
source edge (i, j) -> target qubit k
```

局部关系特征包括：

```text
1-hop 到 5-hop 邻接关系
源/目标节点结构特征
路径冗余度
物理欧氏距离
坐标差分
图谱全局特征，即归一化拉普拉斯谱直方图和 heat kernel 特征
```

这些结构特征的目的，是让模型不仅看局部参数，还能感知目标点在芯片拓扑中的相对位置，从而增强跨拓扑迁移能力。

### 6.3 输出空间

串扰标签会先被裁剪到 `[1e-5, 1e-1]`，再变成 log10 空间：

```python
model_out = torch.log10(data_out.clamp(1e-5, 1e-1))
```

模型输出也被限制在：

```text
[-5, -1]
```

这对应线性空间中的：

```text
[1e-5, 1e-1]
```

训练损失由三部分构成：

```text
base loss: HuberLoss 或 MSELoss
rank loss: pairwise ranking loss，用于提升排序能力
bound loss: 约束 raw output 不偏离合理 log10 串扰范围
```

---

## 7. 指标说明与评价目标

本项目的主观察指标是 **Spearman 排序相关系数**，因为 Evaluator 最终服务于 Designer：Designer 首先需要知道“哪个候选设计更好”，也就是候选之间的排序是否可靠。

但是，**Spearman 不是唯一指标，MAE 也很重要**。Spearman 回答的是“排序是否正确”，MAE 回答的是“预测数值离真实值有多远”。因此推荐评价口径是：

```text
Spearman = 主指标，用于判断 local / ScoreTotal 排序是否收敛、是否可用于候选筛选。
MAE      = 重要辅助指标，用于判断误差规模是否可接受，防止只有排序但数值偏差过大。
MAE_r    = 相对误差指标，尤其适合不同拓扑、不同总分量级之间的对比。
Pearson  = 辅助观察线性相关性，不作为最核心结论。
```

### 7.1 局部指标：local source-target / edge-target

局部指标在 log10 串扰空间计算。

单比特门 X/Y：

```text
每条样本 = 一个 source qubit -> target qubit 的局部串扰预测
```

iSWAP：

```text
每条样本 = 一个 source edge -> target qubit 的局部串扰预测
```

脚本输出中对应：

```text
MSE
MAE
Sp
pearsonr
```

其中：

```text
Sp  = local Transfer_Sp，主要观察 local 排序是否可迁移。
MAE = local log10-space MAE，观察局部串扰数值误差是否可接受。
```

评价目标：

```text
local source-target / edge-target 的 Spearman 应该在未见拓扑上保持收敛或稳定；
同时 local MAE 不应明显失控，尤其不能出现 Spearman 看起来较高但 MAE 大幅恶化的情况。
```

这说明 Evaluator 学到了可迁移的局部物理基础，例如局部参数差异、hop 距离、物理距离、路径冗余与串扰之间的关系。这里的结论不应该只写“Spearman 收敛”，更完整的说法是：**Spearman 证明局部排序可迁移，MAE 证明局部数值误差仍在可控范围内**。

### 7.2 单门聚合指标：all_Transfer_*

局部预测完成后，代码会把 log10 预测值还原到线性串扰空间：

```python
Single_ct_pre = embedder.bw_modelout_to_dataout_single_gate(pred)
iSWAP_ct_pre = embedder.bw_modelout_to_dataout_iSWAP(pred)
```

然后对每个候选样本聚合：

```python
ct.sum(-1).mean(-1)
```

脚本输出中对应：

```text
all MAE
all MAE r
all Sp
all pearsonr
```

评价目标：

```text
all Sp 应该保持较高，说明局部预测还原到线性空间并聚合后，仍能对候选样本进行正确排序；
all MAE / all MAE_r 也要同步报告，说明聚合后的候选级误差规模是否可接受。
```

这里需要特别注意：local MAE 是在 log10 空间看局部误差，而 all MAE 是还原到线性空间并聚合后的候选级误差。两者含义不同，不能互相替代。

### 7.3 总分指标：ScoreTotal

`fit_SmallGraphs_total.py` 会加载三个已经训练好的局部模型：

```text
X evaluator
Y evaluator
iSWAP evaluator
```

然后分别预测局部串扰，恢复到线性空间，并计算：

```python
ScoreTotal = (
    X_ct.sum(-1).mean(-1)
    + Y_ct.sum(-1).mean(-1)
    + iSWAP_ct.sum(-1).mean(-1)
)
```

预测总分同理：

```python
ScoreTotal_pred = (
    X_ct_pre.sum(-1).mean(-1)
    + Y_ct_pre.sum(-1).mean(-1)
    + iSWAP_ct_pre.sum(-1).mean(-1)
)
```

输出指标包括：

```text
Transfer_MSE
Transfer_MAE
Transfer_MAE_r
Transfer_Sp
Transfer_Pearson
```

评价目标：

```text
ScoreTotal Spearman 在未见拓扑上应该保持收敛或稳定，这是 Designer 候选排序的核心指标；
ScoreTotal MAE / MAE_r 也必须保持在合理范围内，这是总分数值可信度的重要证据。
```

这一步比 local 指标更严格，因为它验证的是：大量局部串扰预测从 log 空间还原到线性空间并进行全局聚合后，整体设计排序是否仍然可靠。更准确地说，**ScoreTotal Spearman 说明能不能排序筛选候选，ScoreTotal MAE / MAE_r 说明总分预测偏差有多大**。

### 7.4 Designer_sample external holdout

如果后续加入 Designer 生成的样本，推荐将其命名为：

```text
Designer_sample/
```

或单独放在：

```text
External_Holdout/Designer_sample/
```

它的使用原则必须是：

```text
Designer_sample 只做 external holdout。
不参与训练。
不参与 validation。
不参与 checkpoint 选择。
不参与模型结构或超参数选择。
只在最终模型固定后评估一次。
```

推荐在报告中单独列出 Designer_sample 的结果，避免和训练拓扑、validation 拓扑、zero-shot transfer 拓扑混在一起。Designer_sample 的结论同样不能只看 Spearman：如果 Spearman 较高但 MAE / MAE_r 很差，只能说明候选排序有一定价值，不能说明 Evaluator 的绝对总分预测已经准确。

---

## 8. 推荐报告格式

建议最终汇报时至少包含两张表。

### 8.1 Local transfer 排序表

| Topology | X local Sp | X all Sp | Y local Sp | Y all Sp | iSWAP local Sp | iSWAP all Sp |
|---|---:|---:|---:|---:|---:|---:|
| 3x4 |  |  |  |  |  |  |
| 3x5 |  |  |  |  |  |  |
| 4x4 |  |  |  |  |  |  |
| 3x6 |  |  |  |  |  |  |
| 4x5 |  |  |  |  |  |  |
| 4x6 |  |  |  |  |  |  |
| 5x5 |  |  |  |  |  |  |

解释重点：

```text
local Sp 证明局部物理机制可迁移。
all Sp 证明局部预测聚合到候选级别后仍能排序。
Spearman 是主要收敛观察指标，但不是唯一指标。
```

### 8.2 Local transfer 误差表

| Topology | X local MAE | X all MAE_r | Y local MAE | Y all MAE_r | iSWAP local MAE | iSWAP all MAE_r |
|---|---:|---:|---:|---:|---:|---:|
| 3x4 |  |  |  |  |  |  |
| 3x5 |  |  |  |  |  |  |
| 4x4 |  |  |  |  |  |  |
| 3x6 |  |  |  |  |  |  |
| 4x5 |  |  |  |  |  |  |
| 4x6 |  |  |  |  |  |  |
| 5x5 |  |  |  |  |  |  |

解释重点：

```text
local MAE 用于检查 log10-space 局部预测误差。
all MAE_r 用于检查线性空间聚合后的相对误差。
如果 Spearman 收敛但 MAE / MAE_r 明显变差，需要在报告中说明模型主要可靠在排序，而非绝对数值。
```

### 8.3 ScoreTotal transfer 表

| Topology | ScoreTotal MAE | ScoreTotal MAE_r | ScoreTotal Sp | ScoreTotal Pearson |
|---|---:|---:|---:|---:|
| 3x4 |  |  |  |  |
| 3x5 |  |  |  |  |
| 4x4 |  |  |  |  |
| 3x6 |  |  |  |  |
| 4x5 |  |  |  |  |
| 4x6 |  |  |  |  |
| 5x5 |  |  |  |  |

解释重点：

```text
ScoreTotal 是 Designer 真正关心的候选级目标。
ScoreTotal Sp 是主要结论：它说明 Evaluator 能否用于跨规模候选筛选。
ScoreTotal MAE / MAE_r 是重要补充：它说明总分预测的数值偏差是否可接受。
```

### 8.4 Designer_sample external holdout 表

| Dataset | ScoreTotal MAE | ScoreTotal MAE_r | ScoreTotal Sp | ScoreTotal Pearson | Note |
|---|---:|---:|---:|---:|---|
| Designer_sample |  |  |  |  | external holdout only |

解释重点：

```text
Designer_sample 不用于任何模型训练或选择，因此它反映最终 Evaluator 对 Designer 生成候选的真实泛化能力。
这里也要同时报告 Spearman 和 MAE / MAE_r：前者说明排序能力，后者说明数值误差规模。
```

---

## 11. 一句话总结

本项目的核心不是单纯追求某个拓扑上的最低误差，而是验证：

```text
Evaluator 是否能从小拓扑学习可迁移的局部串扰物理规律，
并在更大拓扑和 Designer 生成候选上保持可靠的候选级 ScoreTotal 排序能力；
其中 Spearman 是主要收敛指标，MAE / MAE_r 是必须同步报告的误差规模指标。
```
