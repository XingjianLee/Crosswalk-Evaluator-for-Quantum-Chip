import argparse
import sys, os, importlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import matplotlib.pyplot as plt
import datetime
import math
import pandas as pd
from model.model2_GBFCNres_3_8 import bw_DataEmbedder
from model.MLP_NEW_F import GBFCN2_single_t3_new_global
from utils import load_pkl, save_pkl
from generate_grid_adjacency import generate_grid_adj
from scipy.stats import spearmanr, pearsonr

# dev = torch.device('cuda:0')
dev = torch.device('cpu')
dtype = torch.float32

CT_THRESHOLD = -5.0


def seed_everything(seed=0):
    """
    固定所有的随机种子，确保实验 100% 可复现
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 5. 固定 PyTorch GPU 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def rank_loss(pred, target):
    """
    Pairwise ranking loss，直接约束相对顺序。
    pred/target: (B,1) 或 (B,)
    """
    pred = pred.view(-1)
    target = target.view(-1)

    if pred.numel() < 2:
        return pred.new_tensor(0.0)

    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_true = target.unsqueeze(1) - target.unsqueeze(0)

    mask = diff_true != 0
    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    sign = torch.sign(diff_true[mask])
    # softplus 比 log1p(exp()) 更稳定
    return F.softplus(-sign * diff_pred[mask]).mean()


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0, 1.0

    corr = spearmanr(y_true, y_pred).correlation
    p_val = spearmanr(y_true, y_pred).pvalue
    if corr is None or np.isnan(corr):
        return 0.0, 1.0
    return float(corr), float(p_val)


def get_spectral_features(adj_matrix):
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

    device = adj_matrix.device
    N = adj_matrix.shape[0]

    degree = adj_matrix.sum(dim=1)
    d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
    d_inv_sqrt_mat = torch.diag(d_inv_sqrt)

    identity = torch.eye(N, device=device)
    adj_float = adj_matrix.to(dtype=torch.float32)
    L_norm = identity - d_inv_sqrt_mat @ adj_float @ d_inv_sqrt_mat

    eigvals = torch.linalg.eigvalsh(L_norm)

    bins = [0.0, 0.4, 0.8, 1.2, 1.6, 2.01]
    hist_feats = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        count = ((eigvals >= low) & (eigvals < high)).float().sum()
        hist_feats.append(count / N)

    hist_tensor = torch.stack(hist_feats)

    time_scales = [1, 5, 10]
    heat_feats = []
    for t in time_scales:
        heat_val = torch.exp(-t * eigvals).sum() / N
        heat_feats.append(heat_val)
    heat_tensor = torch.stack(heat_feats)

    return torch.cat([hist_tensor, heat_tensor])


def normalize_ops(ops, lower, upper):
    lower = torch.tensor(lower, dtype=ops.dtype, device=ops.device)
    upper = torch.tensor(upper, dtype=ops.dtype, device=ops.device)
    lower = lower.reshape(1, 1, -1)
    upper = upper.reshape(1, 1, -1)
    return (ops - lower) / (upper - lower)


def fit_smallgraphs_single(print_or_plot_or_savemodel=True, arg=None, saving_dir=None):
    gate_type = arg.gate_type
    bw_size = arg.bw_size
    lr_decay = arg.lr_decay
    lr = arg.lr

    if arg.loss_func == 'huber':
        base_criterion = nn.HuberLoss(delta=1, reduction='mean')
    else:
        base_criterion = nn.MSELoss(reduction='mean')
    model = GBFCN2_single_t3_new_global(arg.hidden_dim, arg.num_layers, arg.dropout,
                                        arg.xw_in_dim, arg.adj_in_dim, arg.global_in_dim)
    model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0e-4)

    graph_1x6t_adj_mat = torch.tensor(generate_grid_adj(1, 6)).to(dtype).to(dev)
    graph_2x4t_adj_mat = torch.tensor(generate_grid_adj(2, 4)).to(dtype).to(dev)
    graph_3x3t_adj_mat = torch.tensor(generate_grid_adj(3, 3)).to(dtype).to(dev)

    graph_adj_mats = [graph_3x3t_adj_mat, graph_2x4t_adj_mat, graph_1x6t_adj_mat]
    graph_names = ['Random_0_9_3x3bit_seed0_step02_560um',
                   'Random_0_8_2x4bit_seed0_step02_560um',
                   'Random_0_6_1x6bit_seed0_step02_560um']
    graph_row_col = [(3, 3), (2, 4), (1, 6)]

    graph_data_embedders = [
        bw_DataEmbedder(adj_mat, row_col, dev=dev, dtype=dtype)
        for adj_mat, row_col in zip(graph_adj_mats, graph_row_col)
    ]

    loaded_datas = [(
            torch.tensor(load_pkl(f'Datasets_FIN/{graph_names[graph_adj_mat_id]}/state_FIN.pkl')),
            torch.tensor(load_pkl(f'Datasets_FIN/{graph_names[graph_adj_mat_id]}/cross_talk_{gate_type}_FIN.pkl')))
        for graph_adj_mat_id in range(len(graph_adj_mats))]

    loaded_datas = [(
            torch.tensor(normalize_ops(data[0], [-8, -8], [8, 8]), dtype=dtype, device=dev),
            torch.tensor(data[1])[:, :, :adj_mat.shape[0]].to(dtype).to(dev))
        for data, adj_mat in zip(loaded_datas, graph_adj_mats)]

    print("data loaded.", datetime.datetime.now())
    sys.stdout.flush()

    graph_spectral_feats = []
    for adj in graph_adj_mats:
        spec_feat = get_spectral_features(adj).to(dev)
        graph_spectral_feats.append(spec_feat)

    embedded_datas = []
    for graph_adj_mat_id in range(len(graph_adj_mats)):
        graph_data_embedder = graph_data_embedders[graph_adj_mat_id]
        loaded_data = loaded_datas[graph_adj_mat_id]

        embedded_data_in = graph_data_embedder.bw_datain_to_modelin_single_gate(loaded_data[0])
        embedded_data_out = graph_data_embedder.bw_dataout_to_modelout_single_gate(loaded_data[1])

        spec_feat = graph_spectral_feats[graph_adj_mat_id]
        batch_size = embedded_data_in.shape[0]
        spec_feat_batch = spec_feat.unsqueeze(0).expand(batch_size, -1)

        embedded_data_in = torch.cat([embedded_data_in, spec_feat_batch], dim=1)
        embedded_datas.append((embedded_data_in, embedded_data_out))

    print("data embedded.", datetime.datetime.now())
    sys.stdout.flush()

    min_data_size = min(d[0].shape[0] for d in embedded_datas)

    inputs_per_graph = [d[0][:min_data_size] for d in embedded_datas]
    targets_per_graph = [d[1][:min_data_size] for d in embedded_datas]

    train_x_list, valid_x_list, test_x_list = [], [], []
    train_y_list, valid_y_list, test_y_list = [], [], []

    train_ratio = 0.8
    valid_ratio = 0.1

    for curr_x, curr_y in zip(inputs_per_graph, targets_per_graph):
        total_samples = curr_x.shape[0]
        train_sz = int(total_samples * train_ratio)
        valid_sz = int(total_samples * valid_ratio)
        test_sz = total_samples - train_sz - valid_sz

        tx, vx, tex = torch.split(curr_x, [train_sz, valid_sz, test_sz])
        ty, vy, tey = torch.split(curr_y, [train_sz, valid_sz, test_sz])

        train_x_list.append(tx)
        valid_x_list.append(vx)
        test_x_list.append(tex)

        train_y_list.append(ty)
        valid_y_list.append(vy)
        test_y_list.append(tey)

    train_x = torch.cat(train_x_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    valid_x = torch.cat(valid_x_list, dim=0)
    valid_y = torch.cat(valid_y_list, dim=0)

    test_x = torch.cat(test_x_list, dim=0)
    test_y = torch.cat(test_y_list, dim=0)

    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=bw_size, shuffle=True, drop_last=True)

    MSE_train_list, MSE_valid_list = [], []
    MAE_train_list, MAE_valid_list = [], []
    Sp_train_list, Sp_valid_list = [], []


    rank_max_weight = getattr(arg, 'rank_max_weight', 0.30)
    rank_warmup = max(1, getattr(arg, 'rank_warmup', 10))
    bound_weight = getattr(arg, 'bound_weight', 0.01)

    for epoch in range(arg.epochs):
        model.train()
        MSE_train_accum = 0.0
        MAE_train_accum, Sp_train_accum = 0.0, 0.0
        train_batches = 0


        lambda_rank = min(rank_max_weight, (epoch / rank_warmup) * rank_max_weight)

        for batch_in, batch_out in train_loader:
            model_out_pred, raw_out = model(batch_in)

            # loss_base = criterion(base_criterion, model_out_pred, batch_out)
            loss_base = base_criterion(model_out_pred, batch_out)
            loss_rank = rank_loss(raw_out, batch_out)
            loss_bound = F.relu(raw_out - (-1.0)).mean() + F.relu(-5.0 - raw_out).mean()

            loss = loss_base + lambda_rank * loss_rank + bound_weight * loss_bound

            batch_out_np = batch_out.detach().cpu().numpy().reshape(-1)
            pred_np = model_out_pred.detach().cpu().numpy().reshape(-1)

            MSE_train_accum += loss.item()
            MAE_train_accum += np.mean(np.abs(pred_np - batch_out_np))
            Sp_train_accum += safe_spearman(batch_out_np, pred_np)[0]

            train_batches += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if train_batches > 0:
            MSE_train_list.append(MSE_train_accum / train_batches)
            MAE_train_list.append(MAE_train_accum / train_batches)
            Sp_train_list.append(Sp_train_accum / train_batches)
        else:
            MSE_train_list.append(0)
            MAE_train_list.append(0)
            Sp_train_list.append(0)

        model.eval()
        with torch.no_grad():
            valid_in, valid_out = valid_dataset.tensors
            model_out_pred, raw_out = model(valid_in)

            # loss_base = criterion(base_criterion, model_out_pred, valid_out)
            loss_base = base_criterion( model_out_pred, valid_out)
            loss_rank = rank_loss(raw_out, valid_out)
            loss_bound = F.relu(raw_out - (-1.0)).mean() + F.relu(-5.0 - raw_out).mean()
            loss = loss_base + lambda_rank * loss_rank + bound_weight * loss_bound

            valid_out_np = valid_out.detach().cpu().numpy().reshape(-1)
            pred_np = model_out_pred.detach().cpu().numpy().reshape(-1)

            MSE_valid = loss.item()
            MAE_valid = np.mean(np.abs(pred_np - valid_out_np))
            Sp_valid = safe_spearman(valid_out_np, pred_np)[0]

        MSE_valid_list.append(MSE_valid)
        MAE_valid_list.append(MAE_valid)
        Sp_valid_list.append(Sp_valid)


        if epoch % 100 == 0 and epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_decay

        if print_or_plot_or_savemodel:
            print(
                "epoch: {}, loss: {:.4f}, {:.4f}, MAE: {:.4f}, {:.4f}, Sp: {:.3f}, {:.3f}.".format(
                    epoch,
                    MSE_train_list[-1], MSE_valid_list[-1],
                    MAE_train_list[-1], MAE_valid_list[-1],
                    Sp_train_list[-1], Sp_valid_list[-1]
                ),
                datetime.datetime.now()
            )


            sys.stdout.flush()

            if epoch % 10 == 0:
                plt.figure(figsize=(7, 8))
                plt.clf()
                plt.subplot(2, 1, 1)
                plt.plot(MSE_train_list, 'b-', label='train')
                plt.plot(MSE_valid_list, 'r-', label='valid')
                plt.legend()
                plt.ylabel('MSE')
                plt.yscale('log')
                plt.ylim(0.01, 1)
                plt.title(r'MSE and Spearman')
                plt.grid()

                plt.subplot(2, 1, 2)
                plt.plot(Sp_train_list, 'b--', label='train')
                plt.plot(Sp_valid_list, 'r--', label='valid')
                plt.legend()
                plt.ylabel('Spearman (Sp)')
                plt.xlabel('Epochs')
                plt.ylim(0.6, 1)
                plt.grid()

                plt.tight_layout()
                plt.savefig(os.path.join(saving_dir, f"delta_sp_{gate_type}.png"), bbox_inches='tight', dpi=600)
                plt.close()
                torch.save(model.state_dict(), os.path.join(saving_dir, f"model_{gate_type}_{epoch}.pt"))  # [NEW]

    # --- Test ---
    model.eval()
    with torch.no_grad():
        test_in, test_out = test_dataset.tensors
        model_out_pred, raw_out = model(test_in)

        # loss_base = criterion(base_criterion, model_out_pred, test_out)
        loss_base = base_criterion(model_out_pred, test_out)
        loss_rank = rank_loss(raw_out, test_out)
        loss_bound = F.relu(raw_out - (-1.0)).mean() + F.relu(-5.0 - raw_out).mean()
        loss = loss_base + rank_max_weight * loss_rank + bound_weight * loss_bound

        MSE_test = loss.item()
        MAE_test = torch.mean(torch.abs(model_out_pred - test_out)).item()
        Sp_test, p_val_test = safe_spearman(
            test_out.detach().cpu().numpy().reshape(-1),
            model_out_pred.detach().cpu().numpy().reshape(-1)
        )

    if print_or_plot_or_savemodel:
        # [NEW] 打印输出加入 P 值
        print("test: MSE: {:.4f}, MAE: {:.4f}, Sp: {:.3f} (p-value: {:.2e}).".format(
            MSE_test, MAE_test, Sp_test, p_val_test), datetime.datetime.now())

    base_metrics = {
        'Train_MAE': MAE_train_list[-1], 'Train_Sp': Sp_train_list[-1],
        'Valid_MAE': MAE_valid_list[-1], 'Valid_Sp': Sp_valid_list[-1],
        'Test_MAE': MAE_test, 'Test_Sp': Sp_test
    }
    save_pkl(base_metrics, os.path.join(saving_dir, "base_metrics.pkl"))
    return 0



def transfer_smallgraphs(gate_type='X', pretrained_path=None, arg=None,
                         graph_names=None, graph_adj_mats=None, graph_row_col=None):
    if arg.loss_func == 'huber':
        base_criterion = nn.HuberLoss(delta=1, reduction='mean')
    else:
        base_criterion = nn.MSELoss(reduction='mean')

    # print(f"Loading pretrained model from {pretrained_path}")
    model = GBFCN2_single_t3_new_global(arg.hidden_dim, arg.num_layers, arg.dropout,
                                        arg.xw_in_dim, arg.adj_in_dim, arg.global_in_dim)
    model.load_state_dict(torch.load(f'{pretrained_path}', map_location=dev))
    model.to(dev)
    model.eval()

    qubit_num = graph_adj_mats[0].shape[0]

    graph_data_embedders = [
        bw_DataEmbedder(adj_mat, row_col, dev=dev, dtype=dtype)
        for adj_mat, row_col in zip(graph_adj_mats, graph_row_col)
    ]

    loaded_datas = [(
            torch.tensor(load_pkl(f'Datasets_FIN/{graph_names[graph_adj_mat_id]}/state_FIN.pkl')),
            torch.tensor(load_pkl(f'Datasets_FIN/{graph_names[graph_adj_mat_id]}/cross_talk_{gate_type}_FIN.pkl')))
            for graph_adj_mat_id in range(len(graph_adj_mats))]

    loaded_datas = [(
            torch.tensor(normalize_ops(data[0], [-8, -8], [8, 8])).to(dtype).to(dev),
            torch.tensor(data[1])[:, :, :qubit_num].to(dtype).to(dev)) for data in loaded_datas]

    sample_num = loaded_datas[0][0].shape[0]
    graph_spectral_feats = []
    for adj in graph_adj_mats:
        spec_feat = get_spectral_features(adj).to(dev)
        graph_spectral_feats.append(spec_feat)

    data_in, data_out = loaded_datas[0]
    embedder = graph_data_embedders[0]

    embedded_in = embedder.bw_datain_to_modelin_single_gate(data_in)
    embedded_out = embedder.bw_dataout_to_modelout_single_gate(data_out)

    spec_feat = graph_spectral_feats[0]
    batch_size = embedded_in.shape[0]
    spec_feat_batch = spec_feat.unsqueeze(0).expand(batch_size, -1)

    embedded_in = torch.cat([embedded_in, spec_feat_batch], dim=1)

    with torch.no_grad():
        pred, _ = model(embedded_in)

    # mse = criterion(base_criterion, pred, embedded_out).item()
    mse = base_criterion(pred, embedded_out).item()
    mae = torch.mean(torch.abs(pred - embedded_out)).item()
    Sp, p_val = safe_spearman(
        embedded_out.detach().cpu().numpy().reshape(-1),
        pred.detach().cpu().numpy().reshape(-1)
    )
    corr, p_val_c = pearsonr(embedded_out.detach().cpu().numpy().reshape(-1),
                             pred.detach().cpu().numpy().reshape(-1))

    Single_ct_pre = embedder.bw_modelout_to_dataout_single_gate(pred)
    Single_ct = embedder.bw_modelout_to_dataout_single_gate(embedded_out)
    all_Sp, all_p = safe_spearman(Single_ct.sum(-1).mean(-1).detach().cpu().numpy(),
                                  Single_ct_pre.sum(-1).mean(-1).detach().cpu().numpy())
    all_corr, all_p_c = pearsonr(Single_ct.sum(-1).mean(-1).detach().cpu().numpy(),
                                 Single_ct_pre.sum(-1).mean(-1).detach().cpu().numpy())
    all_MAE = torch.abs(Single_ct.sum(-1).mean(-1) - Single_ct_pre.sum(-1).mean(-1)).mean().item()
    all_MAE_r = (torch.abs(Single_ct.sum(-1).mean(-1) - Single_ct_pre.sum(-1).mean(-1)) / Single_ct.sum(-1).mean(-1)).mean().item()

    print(f"\n====Gate type:{gate_type}  Zero-Shot → {graph_row_col} Transfer  ====")
    print(f'sample_num: {sample_num}')
    print(f"MSE = {mse:.6f}")
    print(f"MAE = {mae:.6f}")
    print(f"Sp = {Sp:.4f} (p-value: {p_val:.2e})")
    print(f"pearsonr = {corr:.4f} (p-value: {p_val_c:.2e})")

    print(f"all MAE = {all_MAE:.6f}")
    print(f"all MAE r= {all_MAE_r:.6f}")
    print(f"all Sp = {all_Sp:.4f} (p-value: {all_p:.2e})")
    print(f"all pearsonr = {all_corr:.4f} (p-value: {all_p_c:.2e})")

    return {'Transfer_MAE': mae, 'Transfer_Sp': Sp, 'Transfer_p_val': p_val,
            'all_Transfer_MAE': all_MAE, 'all_Transfer_MAE_r': all_MAE_r,
            'all_Transfer_Sp': all_Sp, 'all_Transfer_p_val': all_p}


def main(arg):
    seed_everything(arg.seed)
    folder_name = f"bs_{arg.bw_size}_lr{arg.lr}_hid{arg.hidden_dim}_lyr{arg.num_layers}_dp{arg.dropout}_loss_{arg.loss_func}_rank{arg.rank_max_weight}_warm{arg.rank_warmup}_bnd{arg.bound_weight}"
    saving_dir = os.path.join(f'run_{arg.gate_type}', folder_name)

    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    fit_smallgraphs_single(arg=arg, saving_dir=saving_dir)
    pretrained_path = os.path.join(saving_dir, f"model_{arg.gate_type}_{arg.epochs-1}.pt")
    # pretrained_path = os.path.join(f"run_{arg.gate_type}/model_{arg.gate_type}_{arg.epochs - 1}.pt")
    transfer_results_list = []

    # ---------------- 统一收集各个拓扑结构的 Zero-Shot 结果 ----------------
    graph_3x4_adj_mat = torch.tensor(generate_grid_adj(3, 4)).to(dtype).to(dev)
    res_3x4 = transfer_smallgraphs(gate_type=arg.gate_type, pretrained_path=pretrained_path, arg=arg,
                                   graph_names=['Random_0_12_3x4bit_seed0_step02_560um'],
                                   graph_adj_mats=[graph_3x4_adj_mat],
                                   graph_row_col=[(3, 4)])
    transfer_results_list.append({'topology': '3x4', 'metrics': res_3x4})

    graph_3x5_adj_mat = torch.tensor(generate_grid_adj(3, 5)).to(dtype).to(dev)
    res_3x5 = transfer_smallgraphs(gate_type=arg.gate_type, pretrained_path=pretrained_path, arg=arg,
                                   graph_names=['Random_0_15_3x5bit_seed0_step02_560um'],
                                   graph_adj_mats=[graph_3x5_adj_mat],
                                   graph_row_col=[(3, 5)])
    transfer_results_list.append({'topology': '3x5', 'metrics': res_3x5})

    graph_4x4_adj_mat = torch.tensor(generate_grid_adj(4, 4)).to(dtype).to(dev)
    res_4x4 = transfer_smallgraphs(gate_type=arg.gate_type, pretrained_path=pretrained_path, arg=arg,
                                   graph_names=['Random_0_16_4x4bit_seed0_step02_560um'],
                                   graph_adj_mats=[graph_4x4_adj_mat],
                                   graph_row_col=[(4, 4)])
    transfer_results_list.append({'topology': '4x4', 'metrics': res_4x4})

    graph_3x6_adj_mat = torch.tensor(generate_grid_adj(3, 6)).to(dtype).to(dev)
    res_3x6 = transfer_smallgraphs(gate_type=arg.gate_type, pretrained_path=pretrained_path, arg=arg,
                                   graph_names=['Random_0_18_3x6bit_seed0_step02_560um'],
                                   graph_adj_mats=[graph_3x6_adj_mat],
                                   graph_row_col=[(3, 6)])
    transfer_results_list.append({'topology': '3x6', 'metrics': res_3x6})

    graph_4x5_adj_mat = torch.tensor(generate_grid_adj(4, 5)).to(dtype).to(dev)
    res_4x5 = transfer_smallgraphs(gate_type=arg.gate_type, pretrained_path=pretrained_path, arg=arg,
                                   graph_names=['Random_0_20_4x5bit_seed0_step02_560um'],
                                   graph_adj_mats=[graph_4x5_adj_mat],
                                   graph_row_col=[(4, 5)])
    transfer_results_list.append({'topology': '4x5', 'metrics': res_4x5})

    graph_4x6_adj_mat = torch.tensor(generate_grid_adj(4, 6)).to(dtype).to(dev)
    res_4x6 = transfer_smallgraphs(gate_type=arg.gate_type, pretrained_path=pretrained_path, arg=arg,
                                   graph_names=['Random_0_24_4x6bit_seed0_step02_560um'],
                                   graph_adj_mats=[graph_4x6_adj_mat],
                                   graph_row_col=[(4, 6)])
    transfer_results_list.append({'topology': '4x6', 'metrics': res_4x6})

    graph_5x5_adj_mat = torch.tensor(generate_grid_adj(5, 5)).to(dtype).to(dev)
    res_5x5 = transfer_smallgraphs(gate_type=arg.gate_type, pretrained_path=pretrained_path, arg=arg,
                                   graph_names=['Random_0_25_5x5bit_seed0_step02_560um'],
                                   graph_adj_mats=[graph_5x5_adj_mat],
                                   graph_row_col=[(5, 5)])
    transfer_results_list.append({'topology': '5x5', 'metrics': res_5x5})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--gate_type', type=str, default='X', help='')

    """pretraining setting """
    parser.add_argument("--epochs", type=int, default=151, help="")
    parser.add_argument("--bw_size", type=int, default=512, help="")
    parser.add_argument('--lr_decay', type=float, default=0.98, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--dropout', type=float, default=0.2, help='')
    parser.add_argument('--loss_func', type=str, default='huber', help='')
    parser.add_argument('--num_layers', type=int, default=2, help='')
    parser.add_argument('--xw_in_dim', type=int, default=10, help='')
    parser.add_argument('--adj_in_dim', type=int, default=6, help='')
    parser.add_argument('--global_in_dim', type=int, default=8, help='')
    parser.add_argument('--nhead', type=int, default=4, help='')
    parser.add_argument('--rank_max_weight', type=float, default=0.3, help='')
    parser.add_argument('--rank_warmup', type=int, default=20, help='')
    parser.add_argument('--bound_weight', type=float, default=0.01, help='')
    args = parser.parse_args()
    main(args)
