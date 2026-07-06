import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
import math
import pandas as pd

from model.model2_GBFCNres_3_8 import bw_DataEmbedder
from model.MLP_NEW_F import (
    GBFCN2_iSWAP_simple_AiHao_v2_global,
    GBFCN2_single_t3_new_global,
)
from utils import load_pkl, save_pkl
from generate_grid_adjacency import generate_grid_adj


# dev = torch.device("cuda:0")
dev = torch.device('cpu')
dtype = torch.float32


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0, 1.0

    res = spearmanr(y_true, y_pred)
    corr = res.correlation
    p_val = res.pvalue

    if corr is None or np.isnan(corr):
        return 0.0, 1.0

    return float(corr), float(p_val)


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0, 1.0

    corr, p_val = pearsonr(y_true, y_pred)

    if corr is None or np.isnan(corr):
        return 0.0, 1.0

    return float(corr), float(p_val)


def normalize_ops(ops, lower, upper):
    lower = torch.tensor(lower, dtype=ops.dtype, device=ops.device).reshape(1, 1, -1)
    upper = torch.tensor(upper, dtype=ops.dtype, device=ops.device).reshape(1, 1, -1)
    return (ops - lower) / (upper - lower)


def get_spectral_features(adj_matrix):
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device=dev)

    adj_matrix = adj_matrix.to(dtype=torch.float32, device=dev)
    n = adj_matrix.shape[0]

    degree = adj_matrix.sum(dim=1)
    d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
    d_inv_sqrt_mat = torch.diag(d_inv_sqrt)

    identity = torch.eye(n, device=adj_matrix.device)
    l_norm = identity - d_inv_sqrt_mat @ adj_matrix @ d_inv_sqrt_mat

    eigvals = torch.linalg.eigvalsh(l_norm)

    bins = [0.0, 0.4, 0.8, 1.2, 1.6, 2.01]
    hist_feats = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        count = ((eigvals >= low) & (eigvals < high)).float().sum()
        hist_feats.append(count / n)

    hist_tensor = torch.stack(hist_feats)

    time_scales = [1, 5, 10]
    heat_feats = []
    for t in time_scales:
        heat_feats.append(torch.exp(-t * eigvals).sum() / n)

    heat_tensor = torch.stack(heat_feats)

    return torch.cat([hist_tensor, heat_tensor])


def transfer_smallgraphs(pretrained_path_iSWAP=None,
                         pretrained_path_X=None,
                         pretrained_path_Y=None,
                         arg=None,
                         graph_names=None,
                         graph_adj_mats=None,
                         graph_row_col=None):
    """
    只评估总串扰 score，不分别评估 X / Y / iSWAP 的局部指标。

    总分定义：
        score = X_ct.sum(-1).mean(-1)
              + Y_ct.sum(-1).mean(-1)
              + iSWAP_ct.sum(-1).mean(-1)
    """

    if arg.loss_func == "huber":
        base_criterion = nn.HuberLoss(delta=1, reduction="mean")
    else:
        base_criterion = nn.MSELoss(reduction="mean")

    model_iSWAP = GBFCN2_iSWAP_simple_AiHao_v2_global(
        arg.hidden_dim_iSWAP,
        arg.num_layers_iSWAP,
        arg.dropout_iSWAP,
        arg.xw_in_dim_iSWAP,
        arg.adj_in_dim_iSWAP,
        arg.global_in_dim_iSWAP,
    ).to(dev)

    model_X = GBFCN2_single_t3_new_global(
        arg.hidden_dim_Single,
        arg.num_layers_Single,
        arg.dropout_Single,
        arg.xw_in_dim_Single,
        arg.adj_in_dim_Single,
        arg.global_in_dim_Single,
    ).to(dev)

    model_Y = GBFCN2_single_t3_new_global(
        arg.hidden_dim_Single,
        arg.num_layers_Single,
        arg.dropout_Single,
        arg.xw_in_dim_Single,
        arg.adj_in_dim_Single,
        arg.global_in_dim_Single,
    ).to(dev)

    model_iSWAP.load_state_dict(torch.load(pretrained_path_iSWAP, map_location=dev))
    model_X.load_state_dict(torch.load(pretrained_path_X, map_location=dev))
    model_Y.load_state_dict(torch.load(pretrained_path_Y, map_location=dev))

    model_iSWAP.eval()
    model_X.eval()
    model_Y.eval()

    all_results = {}

    for graph_id in range(len(graph_names)):
        graph_name = graph_names[graph_id]
        adj_mat = graph_adj_mats[graph_id].to(dtype).to(dev)
        row_col = graph_row_col[graph_id]
        qubit_num = adj_mat.shape[0]

        embedder = bw_DataEmbedder(adj_mat, row_col, dev=dev, dtype=dtype)
        spec_feat = get_spectral_features(adj_mat).to(dev)

        data_in = torch.as_tensor(
            load_pkl(f"Datasets_FIN/{graph_name}/state_FIN.pkl"),
            dtype=dtype,
            device=dev,
        )
        data_in = normalize_ops(data_in, [-8, -8], [8, 8])
        sample_num = data_in.shape[0]

        data_out_X = torch.as_tensor(
            load_pkl(f"Datasets_FIN/{graph_name}/cross_talk_X_FIN.pkl"),
            dtype=dtype,
            device=dev,
        )[:, :, :qubit_num]

        data_out_Y = torch.as_tensor(
            load_pkl(f"Datasets_FIN/{graph_name}/cross_talk_Y_FIN.pkl"),
            dtype=dtype,
            device=dev,
        )[:, :, :qubit_num]

        data_out_iSWAP = torch.as_tensor(
            load_pkl(f"Datasets_FIN/{graph_name}/cross_talk_iSWAP_FIN.pkl"),
            dtype=dtype,
            device=dev,
        )[:, :, :qubit_num]

        score_node = torch.as_tensor(
            load_pkl(f"Datasets_FIN/{graph_name}/score_node_FIN.pkl"),
            dtype=dtype,
            device=dev,
        )
        score_node = score_node[:sample_num]

        # ========================================================
        # X gate
        # 注意：embedded_in_X 的第一维不一定等于 sample_num
        # 例如 1x6 中 sample_num=1200，但 embedded_in_X=36000
        # 所以 spec_feat_batch_X 必须按 embedded_in_X.shape[0] 扩展
        # ========================================================
        embedded_in_X = embedder.bw_datain_to_modelin_single_gate(data_in)
        embedded_out_X = embedder.bw_dataout_to_modelout_single_gate(data_out_X)

        spec_feat_batch_X = spec_feat.unsqueeze(0).expand(embedded_in_X.shape[0], -1)
        embedded_in_X = torch.cat([embedded_in_X, spec_feat_batch_X], dim=1)

        # ========================================================
        # Y gate
        # ========================================================
        embedded_in_Y = embedder.bw_datain_to_modelin_single_gate(data_in)
        embedded_out_Y = embedder.bw_dataout_to_modelout_single_gate(data_out_Y)

        spec_feat_batch_Y = spec_feat.unsqueeze(0).expand(embedded_in_Y.shape[0], -1)
        embedded_in_Y = torch.cat([embedded_in_Y, spec_feat_batch_Y], dim=1)

        # ========================================================
        # iSWAP gate
        # iSWAP 的 embedded_in_iSWAP 第一维也可能不是 sample_num
        # 因此也按 embedded_in_iSWAP.shape[0] 扩展
        # ========================================================
        embedded_in_iSWAP = embedder.bw_datain_to_modelin_iSWAP(data_in, score_node)
        embedded_out_iSWAP = embedder.bw_dataout_to_modelout_iSWAP(data_out_iSWAP)

        spec_feat_batch_iSWAP = spec_feat.unsqueeze(0).expand(embedded_in_iSWAP.shape[0], -1)
        embedded_in_iSWAP = torch.cat([embedded_in_iSWAP, spec_feat_batch_iSWAP], dim=1)

        with torch.no_grad():
            pred_X, _ = model_X(embedded_in_X)
            pred_Y, _ = model_Y(embedded_in_Y)
            pred_iSWAP, _ = model_iSWAP(embedded_in_iSWAP)

        X_ct = embedder.bw_modelout_to_dataout_single_gate(embedded_out_X)
        Y_ct = embedder.bw_modelout_to_dataout_single_gate(embedded_out_Y)
        iSWAP_ct = embedder.bw_modelout_to_dataout_iSWAP(embedded_out_iSWAP)

        X_ct_pre = embedder.bw_modelout_to_dataout_single_gate(pred_X)
        Y_ct_pre = embedder.bw_modelout_to_dataout_single_gate(pred_Y)
        iSWAP_ct_pre = embedder.bw_modelout_to_dataout_iSWAP(pred_iSWAP)

        score = (
            X_ct.sum(-1).mean(-1)
            + Y_ct.sum(-1).mean(-1)
            + iSWAP_ct.sum(-1).mean(-1)
        )

        score_pre = (
            X_ct_pre.sum(-1).mean(-1)
            + Y_ct_pre.sum(-1).mean(-1)
            + iSWAP_ct_pre.sum(-1).mean(-1)
        )

        mse = base_criterion(score_pre, score).item()
        mae = torch.mean(torch.abs(score_pre - score)).item()

        eps = 1e-12
        mae_r = (
            torch.abs(score_pre - score)
            / torch.clamp(torch.abs(score), min=eps)
        ).mean().item()

        score_np = score.detach().cpu().numpy()
        score_pre_np = score_pre.detach().cpu().numpy()

        sp, sp_p = safe_spearman(score_np, score_pre_np)
        pearson, pearson_p = safe_pearson(score_np, score_pre_np)

        metrics = {
            "Transfer_MSE": mse,
            "Transfer_MAE": mae,
            "Transfer_MAE_r": mae_r,
            "Transfer_Sp": sp,
            "Transfer_p_val": sp_p,
            "Transfer_Pearson": pearson,
            "Transfer_Pearson_p_val": pearson_p,
        }

        all_results[graph_name] = metrics

        print(f"\n==== Zero-Shot Total Score Transfer → {graph_name} / {row_col} ====")
        print(f"sample_num: {sample_num}")
        print(f"embedded_in_X shape     : {tuple(embedded_in_X.shape)}")
        print(f"embedded_in_Y shape     : {tuple(embedded_in_Y.shape)}")
        print(f"embedded_in_iSWAP shape : {tuple(embedded_in_iSWAP.shape)}")
        print("score = X_ct.sum(-1).mean(-1) + Y_ct.sum(-1).mean(-1) + iSWAP_ct.sum(-1).mean(-1)")
        print(f"MSE      = {mse:.6f}")
        print(f"MAE      = {mae:.6f}")
        print(f"MAE r    = {mae_r:.6f}")
        print(f"Sp       = {sp:.4f} (p-value: {sp_p:.2e})")
        print(f"pearsonr = {pearson:.4f} (p-value: {pearson_p:.2e})")

    return all_results


def main(arg):
    seed_everything(arg.seed)

    folder_name_iSWAP = (
        f"run_iSWAP/"
        f"bs_{arg.bw_size}_lr{arg.lr}_hid{arg.hidden_dim_iSWAP}_lyr{arg.num_layers_iSWAP}"
        f"_dp{arg.dropout_iSWAP}_loss_{arg.loss_func}_rank{arg.rank_max_weight}"
        f"_warm{arg.rank_warmup}_bnd{arg.bound_weight}"
    )

    folder_name_X = (
        f"run_X/"
        f"bs_{arg.bw_size}_lr{arg.lr}_hid{arg.hidden_dim_Single}_lyr{arg.num_layers_Single}"
        f"_dp{arg.dropout_Single}_loss_{arg.loss_func}_rank{arg.rank_max_weight}"
        f"_warm{arg.rank_warmup}_bnd{arg.bound_weight}"
    )

    folder_name_Y = (
        f"run_Y/"
        f"bs_{arg.bw_size}_lr{arg.lr}_hid{arg.hidden_dim_Single}_lyr{arg.num_layers_Single}"
        f"_dp{arg.dropout_Single}_loss_{arg.loss_func}_rank{arg.rank_max_weight}"
        f"_warm{arg.rank_warmup}_bnd{arg.bound_weight}"
    )

    pretrained_path_iSWAP = os.path.join(
        folder_name_iSWAP,
        f"model_iSWAP_{arg.epochs - 1}.pt",
    )

    pretrained_path_X = os.path.join(
        folder_name_X,
        f"model_X_{arg.epochs - 1}.pt",
    )

    pretrained_path_Y = os.path.join(
        folder_name_Y,
        f"model_Y_{arg.epochs - 1}.pt",
    )

    print("\n==== Pretrained model paths ====")
    print(f"iSWAP: {pretrained_path_iSWAP}")
    print(f"X    : {pretrained_path_X}")
    print(f"Y    : {pretrained_path_Y}")

    topology_configs = [
        {
            "topology": "3x4",
            "row_col": (3, 4),
            "graph_name": "Random_0_12_3x4bit_seed0_step02_560um",
        },
        {
            "topology": "3x5",
            "row_col": (3, 5),
            "graph_name": "Random_0_15_3x5bit_seed0_step02_560um",
        },
        {
            "topology": "4x4",
            "row_col": (4, 4),
            "graph_name": "Random_0_16_4x4bit_seed0_step02_560um",
        },
        {
            "topology": "3x6",
            "row_col": (3, 6),
            "graph_name": "Random_0_18_3x6bit_seed0_step02_560um",
        },
        {
            "topology": "4x5",
            "row_col": (4, 5),
            "graph_name": "Random_0_20_4x5bit_seed0_step02_560um",
        },
        {
            "topology": "4x6",
            "row_col": (4, 6),
            "graph_name": "Random_0_24_4x6bit_seed0_step02_560um",
        },
        {
            "topology": "5x5",
            "row_col": (5, 5),
            "graph_name": "Random_0_25_5x5bit_seed0_step02_560um",
        },
    ]

    graph_names = []
    graph_adj_mats = []
    graph_row_col = []

    for cfg in topology_configs:
        row, col = cfg["row_col"]
        graph_names.append(cfg["graph_name"])
        graph_adj_mats.append(
            torch.tensor(
                generate_grid_adj(row, col),
                dtype=dtype,
                device=dev,
            )
        )
        graph_row_col.append(cfg["row_col"])

    raw_results = transfer_smallgraphs(
        pretrained_path_iSWAP=pretrained_path_iSWAP,
        pretrained_path_X=pretrained_path_X,
        pretrained_path_Y=pretrained_path_Y,
        arg=arg,
        graph_names=graph_names,
        graph_adj_mats=graph_adj_mats,
        graph_row_col=graph_row_col,
    )

    transfer_results_list = []

    for cfg in topology_configs:
        graph_name = cfg["graph_name"]
        transfer_results_list.append({
            "topology": cfg["topology"],
            "graph_name": graph_name,
            "metrics": raw_results[graph_name],
        })

    print("\n==== Summary: Total Score Transfer ====")

    for item in transfer_results_list:
        m = item["metrics"]
        print(
            f"{item['topology']:>4s} | "
            f"MAE={m['Transfer_MAE']:.6f} | "
            f"MAE_r={m['Transfer_MAE_r']:.6f} | "
            f"Sp={m['Transfer_Sp']:.4f} | "
            f"Pearson={m['Transfer_Pearson']:.4f}"
        )

    if arg.save_path is not None and arg.save_path != "":
        save_pkl(transfer_results_list, arg.save_path)
        print(f"\nSaved transfer results to: {arg.save_path}")

    return transfer_results_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=151)
    parser.add_argument("--bw_size", type=int, default=512)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_func", type=str, default="huber", choices=["huber", "mse"])

    parser.add_argument("--hidden_dim_iSWAP", type=int, default=32)
    parser.add_argument("--dropout_iSWAP", type=float, default=0.1)
    parser.add_argument("--num_layers_iSWAP", type=int, default=2)
    parser.add_argument("--xw_in_dim_iSWAP", type=int, default=35)
    parser.add_argument("--adj_in_dim_iSWAP", type=int, default=12)
    parser.add_argument("--global_in_dim_iSWAP", type=int, default=8)

    parser.add_argument("--hidden_dim_Single", type=int, default=32)
    parser.add_argument("--dropout_Single", type=float, default=0.2)
    parser.add_argument("--num_layers_Single", type=int, default=2)
    parser.add_argument("--xw_in_dim_Single", type=int, default=10)
    parser.add_argument("--adj_in_dim_Single", type=int, default=6)
    parser.add_argument("--global_in_dim_Single", type=int, default=8)

    parser.add_argument("--rank_max_weight", type=float, default=0.3)
    parser.add_argument("--rank_warmup", type=int, default=20)
    parser.add_argument("--bound_weight", type=float, default=0.01)

    parser.add_argument(
        "--save_path",
        type=str,
        default="transfer_total_score_results.pkl",
    )

    args = parser.parse_args()
    main(args)