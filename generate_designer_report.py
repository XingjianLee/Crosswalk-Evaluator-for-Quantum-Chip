import datetime as dt
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from generate_grid_adjacency import generate_grid_adj
from model.MLP_NEW_F import (
    GBFCN2_iSWAP_simple_AiHao_v2_global,
    GBFCN2_single_t3_new_global,
)
from model.model2_GBFCNres_3_8 import bw_DataEmbedder


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
RUN_ID = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUT_DIR, f"report_{RUN_ID}")

DEV = torch.device("cpu")
DTYPE = torch.float32


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0, 1.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0, 1.0
    result = spearmanr(y_true, y_pred)
    corr = result.correlation
    if corr is None or np.isnan(corr):
        return 0.0, 1.0
    return float(corr), float(result.pvalue)


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0, 1.0
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
    adj_matrix = adj_matrix.to(dtype=torch.float32, device=DEV)
    n = adj_matrix.shape[0]

    degree = adj_matrix.sum(dim=1)
    d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
    d_inv_sqrt_mat = torch.diag(d_inv_sqrt)

    identity = torch.eye(n, device=adj_matrix.device)
    l_norm = identity - d_inv_sqrt_mat @ adj_matrix @ d_inv_sqrt_mat
    eigvals = torch.linalg.eigvalsh(l_norm)

    bins = [0.0, 0.4, 0.8, 1.2, 1.6, 2.01]
    hist_feats = []
    for low, high in zip(bins[:-1], bins[1:]):
        hist_feats.append(((eigvals >= low) & (eigvals < high)).float().sum() / n)

    heat_feats = [torch.exp(-t * eigvals).sum() / n for t in [1, 5, 10]]
    return torch.cat([torch.stack(hist_feats), torch.stack(heat_feats)])


def load_models():
    paths = {
        "X": os.path.join(
            ROOT,
            "run_X/bs_512_lr0.001_hid32_lyr2_dp0.2_loss_huber_rank0.3_warm20_bnd0.01/model_X_150.pt",
        ),
        "Y": os.path.join(
            ROOT,
            "run_Y/bs_512_lr0.001_hid32_lyr2_dp0.2_loss_huber_rank0.3_warm20_bnd0.01/model_Y_150.pt",
        ),
        "iSWAP": os.path.join(
            ROOT,
            "run_iSWAP/bs_512_lr0.001_hid32_lyr2_dp0.1_loss_huber_rank0.3_warm20_bnd0.01/model_iSWAP_150.pt",
        ),
    }

    model_x = GBFCN2_single_t3_new_global(32, 2, 0.2, 10, 6, 8).to(DEV)
    model_y = GBFCN2_single_t3_new_global(32, 2, 0.2, 10, 6, 8).to(DEV)
    model_iswap = GBFCN2_iSWAP_simple_AiHao_v2_global(32, 2, 0.1, 35, 12, 8).to(DEV)

    model_x.load_state_dict(torch.load(paths["X"], map_location=DEV))
    model_y.load_state_dict(torch.load(paths["Y"], map_location=DEV))
    model_iswap.load_state_dict(torch.load(paths["iSWAP"], map_location=DEV))

    for model in [model_x, model_y, model_iswap]:
        model.eval()

    return {"X": model_x, "Y": model_y, "iSWAP": model_iswap}, paths


def evaluate_dataset(name, row_col, data_dir, models, criterion):
    row, col = row_col
    adj = torch.tensor(generate_grid_adj(row, col), dtype=DTYPE, device=DEV)
    qubit_num = adj.shape[0]
    embedder = bw_DataEmbedder(adj, row_col, dev=DEV, dtype=DTYPE)
    spec_feat = get_spectral_features(adj).to(DEV)

    data_in = torch.as_tensor(load_pkl(os.path.join(data_dir, "state_FIN.pkl")), dtype=DTYPE, device=DEV)
    data_in = normalize_ops(data_in, [-8, -8], [8, 8])
    sample_num = int(data_in.shape[0])

    score_node = torch.as_tensor(
        load_pkl(os.path.join(data_dir, "score_node_FIN.pkl")),
        dtype=DTYPE,
        device=DEV,
    )[:sample_num]

    local = {}
    linear_outputs = {}

    for gate in ["X", "Y"]:
        data_out = torch.as_tensor(
            load_pkl(os.path.join(data_dir, f"cross_talk_{gate}_FIN.pkl")),
            dtype=DTYPE,
            device=DEV,
        )[:, :, :qubit_num]

        embedded_in = embedder.bw_datain_to_modelin_single_gate(data_in)
        embedded_out = embedder.bw_dataout_to_modelout_single_gate(data_out)
        embedded_in = torch.cat(
            [embedded_in, spec_feat.unsqueeze(0).expand(embedded_in.shape[0], -1)],
            dim=1,
        )

        with torch.no_grad():
            pred, _ = models[gate](embedded_in)

        ct_true = embedder.bw_modelout_to_dataout_single_gate(embedded_out)
        ct_pred = embedder.bw_modelout_to_dataout_single_gate(pred)
        linear_outputs[gate] = (ct_true, ct_pred)

        all_true = ct_true.sum(-1).mean(-1)
        all_pred = ct_pred.sum(-1).mean(-1)
        local_sp, local_sp_p = safe_spearman(embedded_out.detach().cpu().numpy(), pred.detach().cpu().numpy())
        local_pearson, local_pearson_p = safe_pearson(
            embedded_out.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
        all_sp, all_sp_p = safe_spearman(all_true.detach().cpu().numpy(), all_pred.detach().cpu().numpy())
        all_pearson, all_pearson_p = safe_pearson(all_true.detach().cpu().numpy(), all_pred.detach().cpu().numpy())

        local[gate] = {
            "local_MSE": float(criterion(pred, embedded_out).item()),
            "local_MAE": float(torch.mean(torch.abs(pred - embedded_out)).item()),
            "local_Sp": local_sp,
            "local_Sp_p": local_sp_p,
            "local_Pearson": local_pearson,
            "local_Pearson_p": local_pearson_p,
            "all_MAE": float(torch.mean(torch.abs(all_pred - all_true)).item()),
            "all_MAE_r": float(
                (torch.abs(all_pred - all_true) / torch.clamp(torch.abs(all_true), min=1e-12)).mean().item()
            ),
            "all_Sp": all_sp,
            "all_Sp_p": all_sp_p,
            "all_Pearson": all_pearson,
            "all_Pearson_p": all_pearson_p,
        }

    data_out_iswap = torch.as_tensor(
        load_pkl(os.path.join(data_dir, "cross_talk_iSWAP_FIN.pkl")),
        dtype=DTYPE,
        device=DEV,
    )[:, :, :qubit_num]

    embedded_in_iswap = embedder.bw_datain_to_modelin_iSWAP(data_in, score_node)
    embedded_out_iswap = embedder.bw_dataout_to_modelout_iSWAP(data_out_iswap)
    embedded_in_iswap = torch.cat(
        [embedded_in_iswap, spec_feat.unsqueeze(0).expand(embedded_in_iswap.shape[0], -1)],
        dim=1,
    )

    with torch.no_grad():
        pred_iswap, _ = models["iSWAP"](embedded_in_iswap)

    ct_true_iswap = embedder.bw_modelout_to_dataout_iSWAP(embedded_out_iswap)
    ct_pred_iswap = embedder.bw_modelout_to_dataout_iSWAP(pred_iswap)
    linear_outputs["iSWAP"] = (ct_true_iswap, ct_pred_iswap)

    all_true = ct_true_iswap.sum(-1).mean(-1)
    all_pred = ct_pred_iswap.sum(-1).mean(-1)
    local_sp, local_sp_p = safe_spearman(
        embedded_out_iswap.detach().cpu().numpy(),
        pred_iswap.detach().cpu().numpy(),
    )
    local_pearson, local_pearson_p = safe_pearson(
        embedded_out_iswap.detach().cpu().numpy(),
        pred_iswap.detach().cpu().numpy(),
    )
    all_sp, all_sp_p = safe_spearman(all_true.detach().cpu().numpy(), all_pred.detach().cpu().numpy())
    all_pearson, all_pearson_p = safe_pearson(all_true.detach().cpu().numpy(), all_pred.detach().cpu().numpy())

    local["iSWAP"] = {
        "local_MSE": float(criterion(pred_iswap, embedded_out_iswap).item()),
        "local_MAE": float(torch.mean(torch.abs(pred_iswap - embedded_out_iswap)).item()),
        "local_Sp": local_sp,
        "local_Sp_p": local_sp_p,
        "local_Pearson": local_pearson,
        "local_Pearson_p": local_pearson_p,
        "all_MAE": float(torch.mean(torch.abs(all_pred - all_true)).item()),
        "all_MAE_r": float(
            (torch.abs(all_pred - all_true) / torch.clamp(torch.abs(all_true), min=1e-12)).mean().item()
        ),
        "all_Sp": all_sp,
        "all_Sp_p": all_sp_p,
        "all_Pearson": all_pearson,
        "all_Pearson_p": all_pearson_p,
    }

    score_true = (
        linear_outputs["X"][0].sum(-1).mean(-1)
        + linear_outputs["Y"][0].sum(-1).mean(-1)
        + linear_outputs["iSWAP"][0].sum(-1).mean(-1)
    )
    score_pred = (
        linear_outputs["X"][1].sum(-1).mean(-1)
        + linear_outputs["Y"][1].sum(-1).mean(-1)
        + linear_outputs["iSWAP"][1].sum(-1).mean(-1)
    )

    score_sp, score_sp_p = safe_spearman(score_true.detach().cpu().numpy(), score_pred.detach().cpu().numpy())
    score_pearson, score_pearson_p = safe_pearson(score_true.detach().cpu().numpy(), score_pred.detach().cpu().numpy())

    score_total = {
        "ScoreTotal_MSE": float(criterion(score_pred, score_true).item()),
        "ScoreTotal_MAE": float(torch.mean(torch.abs(score_pred - score_true)).item()),
        "ScoreTotal_MAE_r": float(
            (torch.abs(score_pred - score_true) / torch.clamp(torch.abs(score_true), min=1e-12)).mean().item()
        ),
        "ScoreTotal_Sp": score_sp,
        "ScoreTotal_Sp_p": score_sp_p,
        "ScoreTotal_Pearson": score_pearson,
        "ScoreTotal_Pearson_p": score_pearson_p,
        "ScoreTotal_true_mean": float(score_true.mean().item()),
        "ScoreTotal_pred_mean": float(score_pred.mean().item()),
    }

    return {
        "dataset": name,
        "row_col": f"{row}x{col}",
        "sample_num": sample_num,
        "local": local,
        "score_total": score_total,
    }


def table_to_markdown(df):
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append(values)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---" for _ in headers]) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_df(df, name):
    path = os.path.join(RUN_DIR, f"{name}_{RUN_ID}.csv")
    df.to_csv(path, index=False)
    return path


def main():
    os.makedirs(RUN_DIR, exist_ok=False)
    models, checkpoint_paths = load_models()
    criterion = nn.HuberLoss(delta=1, reduction="mean")

    topology_configs = [
        ("3x4", (3, 4), os.path.join(ROOT, "Datasets_FIN/Random_0_12_3x4bit_seed0_step02_560um")),
        ("3x5", (3, 5), os.path.join(ROOT, "Datasets_FIN/Random_0_15_3x5bit_seed0_step02_560um")),
        ("4x4", (4, 4), os.path.join(ROOT, "Datasets_FIN/Random_0_16_4x4bit_seed0_step02_560um")),
        ("3x6", (3, 6), os.path.join(ROOT, "Datasets_FIN/Random_0_18_3x6bit_seed0_step02_560um")),
        ("4x5", (4, 5), os.path.join(ROOT, "Datasets_FIN/Random_0_20_4x5bit_seed0_step02_560um")),
        ("4x6", (4, 6), os.path.join(ROOT, "Datasets_FIN/Random_0_24_4x6bit_seed0_step02_560um")),
        ("5x5", (5, 5), os.path.join(ROOT, "Datasets_FIN/Random_0_25_5x5bit_seed0_step02_560um")),
    ]

    results = []
    for config in topology_configs:
        print(f"Evaluating {config[0]}", flush=True)
        results.append(evaluate_dataset(*config, models=models, criterion=criterion))

    designer_config = ("Designer_sample", (5, 5), os.path.join(ROOT, "Designer_sample"))
    print("Evaluating Designer_sample", flush=True)
    designer_result = evaluate_dataset(*designer_config, models=models, criterion=criterion)

    local_sp_rows = []
    local_err_rows = []
    score_rows = []
    for result in results:
        local_sp_rows.append(
            {
                "Topology": result["dataset"],
                "Samples": result["sample_num"],
                "X local Sp": result["local"]["X"]["local_Sp"],
                "X all Sp": result["local"]["X"]["all_Sp"],
                "Y local Sp": result["local"]["Y"]["local_Sp"],
                "Y all Sp": result["local"]["Y"]["all_Sp"],
                "iSWAP local Sp": result["local"]["iSWAP"]["local_Sp"],
                "iSWAP all Sp": result["local"]["iSWAP"]["all_Sp"],
            }
        )
        local_err_rows.append(
            {
                "Topology": result["dataset"],
                "Samples": result["sample_num"],
                "X local MAE": result["local"]["X"]["local_MAE"],
                "X all MAE_r": result["local"]["X"]["all_MAE_r"],
                "Y local MAE": result["local"]["Y"]["local_MAE"],
                "Y all MAE_r": result["local"]["Y"]["all_MAE_r"],
                "iSWAP local MAE": result["local"]["iSWAP"]["local_MAE"],
                "iSWAP all MAE_r": result["local"]["iSWAP"]["all_MAE_r"],
            }
        )
        score_rows.append(
            {
                "Topology": result["dataset"],
                "Samples": result["sample_num"],
                "ScoreTotal MAE": result["score_total"]["ScoreTotal_MAE"],
                "ScoreTotal MAE_r": result["score_total"]["ScoreTotal_MAE_r"],
                "ScoreTotal Sp": result["score_total"]["ScoreTotal_Sp"],
                "ScoreTotal Pearson": result["score_total"]["ScoreTotal_Pearson"],
            }
        )

    designer_rows = [
        {
            "Dataset": designer_result["dataset"],
            "Topology inferred": designer_result["row_col"],
            "Samples": designer_result["sample_num"],
            "ScoreTotal MAE": designer_result["score_total"]["ScoreTotal_MAE"],
            "ScoreTotal MAE_r": designer_result["score_total"]["ScoreTotal_MAE_r"],
            "ScoreTotal Sp": designer_result["score_total"]["ScoreTotal_Sp"],
            "ScoreTotal Pearson": designer_result["score_total"]["ScoreTotal_Pearson"],
            "Note": "external holdout only; not used for training/model selection; only 5 samples",
        }
    ]

    local_sp_df = pd.DataFrame(local_sp_rows)
    local_err_df = pd.DataFrame(local_err_rows)
    score_df = pd.DataFrame(score_rows)
    designer_df = pd.DataFrame(designer_rows)

    output_paths = {
        "local_transfer_spearman": save_df(local_sp_df, "local_transfer_spearman"),
        "local_transfer_error": save_df(local_err_df, "local_transfer_error"),
        "scoretotal_transfer": save_df(score_df, "scoretotal_transfer"),
        "designer_external_holdout": save_df(designer_df, "designer_external_holdout"),
    }

    raw = {
        "run_id": RUN_ID,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "device": str(DEV),
        "checkpoint_paths": checkpoint_paths,
        "transfer_results": results,
        "designer_result": designer_result,
    }

    raw_path = os.path.join(RUN_DIR, f"raw_metrics_{RUN_ID}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    output_paths["raw_metrics"] = raw_path

    report_path = os.path.join(RUN_DIR, f"report_{RUN_ID}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluator Transfer Report {RUN_ID}\n\n")
        f.write("## Run Info\n\n")
        f.write(f"- Generated at: `{raw['generated_at']}`\n")
        f.write(f"- Device: `{DEV}`\n")
        f.write("- Environment: `conda DL`, CPU evaluation\n")
        f.write("- Checkpoints:\n")
        for gate, path in checkpoint_paths.items():
            f.write(f"  - {gate}: `{path}`\n")

        f.write("\n## 8.1 Local Transfer 排序表\n\n")
        f.write(table_to_markdown(local_sp_df))
        f.write("\n\n## 8.2 Local Transfer 误差表\n\n")
        f.write(table_to_markdown(local_err_df))
        f.write("\n\n## 8.3 ScoreTotal Transfer 表\n\n")
        f.write(table_to_markdown(score_df))
        f.write("\n\n## 8.4 Designer_sample External Holdout 表\n\n")
        f.write(table_to_markdown(designer_df))
        f.write("\n\n## Notes\n\n")
        f.write("- Local MAE 在 log10-space 计算。\n")
        f.write("- all MAE_r 和 ScoreTotal MAE_r 在线性串扰空间计算。\n")
        f.write("- Designer_sample 推断为 5x5，因为 `state_FIN.pkl` shape 为 `(5, 25, 2)`，iSWAP 边数为 40。\n")
        f.write("- Designer_sample 只有 5 个候选样本，Spearman/Pearson 统计稳定性较弱，应与 MAE/MAE_r 一起解释。\n")
    output_paths["markdown_report"] = report_path

    print("\nOutput directory:", RUN_DIR)
    for key, value in output_paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
