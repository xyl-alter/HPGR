import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import torch
from torch_sparse import SparseTensor
import math
import numpy as np
import torch
import json
import os.path as osp
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from collections import defaultdict
import shutil
from utils.init_blink_protocol import get_pij as get_pij_init
from utils.modified_blink_protocol import get_pij as get_pij_modified
from sklearn.model_selection import train_test_split
from utils.dataloader import *
from utils.models import *
from utils.train import *
import gc
from torch_geometric.utils import to_dense_adj
from copy import copy
model_args = {'num_layers': 2, 'hidden': 16, 'dropout': 0.4}
train_args={'max_epoches': 500, 'lr': 0.001, 'weight_decay': 5e-4, 'eval_interval': 20}

def train_val_test_split(n, split_percentage=[0.5, 0.25], seed=42):
    if seed is None:
        seed = 42
    train, val_test = train_test_split(range(n), test_size=(1 - split_percentage[0]), random_state=seed)
    val, test = train_test_split(val_test, test_size=1 - (split_percentage[1] / (1 -  split_percentage[0])), random_state=seed)
    train_mask = torch.full([n], False)
    train_mask[torch.tensor(train)] = True
    val_mask = torch.full([n], False)
    val_mask[torch.tensor(val)] = True
    test_mask = torch.full([n], False)
    test_mask[torch.tensor(test)] = True
    return train_mask, val_mask, test_mask

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d
def nested_defaultdict():
    return defaultdict(nested_defaultdict)

def load_data_and_get_pseudo_label(dataset='cora', split_seed=42, device='cpu', model_args=None, train_args=None, mlp_seed=2024, split_percentage=[0.5, 0.25], 
                                   tau=0):
    data = load_data(dataset)
    train_mask, val_mask, test_mask = train_val_test_split(data.num_nodes, split_percentage, split_seed)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.to(device)
    model_args = copy(model_args)
    train_args = copy(train_args)
    if dataset == 'lastfm':
        model_args['hidden'] = 32
    device = torch.device(device)
    model = MLP(model_args['num_layers'], data.x.shape[1], model_args['hidden'], data.num_classes, model_args['dropout'])
    model.to(device)
    print('---------------------training pseudo label---------------------')
    mlp_acc = train_warpper(data.clone(), model, train_args['max_epoches'], train_args['lr'], eval_interval=train_args['eval_interval'], mode='edge_index', seed=mlp_seed)
    pred = model(data.x, data.edge_index).detach()
    data.pred = pred
    del model
    torch.cuda.empty_cache()
    if tau < 1e-6:
        pred = torch.argmax(pred, dim=1)
        pseudo_label = data.y * data.train_mask + data.y * data.val_mask + pred * data.test_mask
        data.pseudo_label = pseudo_label
    else:
        if tau < 1e6:
            pred = torch.nn.Softmax(dim=1)(pred / tau) # 
        else:
            pred = torch.ones((data.num_nodes, data.num_classes), dtype=torch.float32, device=device) / data.num_classes # 所有test节点全部等可能分到各个类别
        gt = F.one_hot(data.y * data.train_mask + data.y * data.val_mask, num_classes=data.num_classes).type(torch.float32)
        gt[data.test_mask] = 0
        pred[~data.test_mask] = 0
        onehot_pseudo_label = gt + pred
        assert torch.abs(torch.sum(onehot_pseudo_label) - onehot_pseudo_label.shape[0]) < 1e-2, 'error of pseudo label is {}!'.format(torch.abs(torch.sum(onehot_pseudo_label) - onehot_pseudo_label.shape[0]))
        data.onehot_pseudo_label = onehot_pseudo_label
    return data, mlp_acc

def reload_data(data, tau):
    pred = data.pred.clone()
    if tau < 1e-6:
        pred = torch.argmax(pred, dim=1)
        pseudo_label = data.y * data.train_mask + data.y * data.val_mask + pred * data.test_mask
        data.pseudo_label = pseudo_label
    else:
        if tau < 1e6:
            pred = torch.nn.Softmax(dim=1)(pred / tau) # 
        else:
            pred = torch.ones((data.num_nodes, data.num_classes), dtype=torch.float32, device=data.device) / data.num_classes # 所有test节点全部等可能分到各个类别
        gt = F.one_hot(data.y * data.train_mask + data.y * data.val_mask, num_classes=data.num_classes).type(torch.float32)
        gt[data.test_mask] = 0
        pred[~data.test_mask] = 0
        onehot_pseudo_label = gt + pred
        assert torch.abs(torch.sum(onehot_pseudo_label) - onehot_pseudo_label.shape[0]) < 1e-2, 'error of pseudo label is {}!'.format(torch.abs(torch.sum(onehot_pseudo_label) - onehot_pseudo_label.shape[0]))
        data.onehot_pseudo_label = onehot_pseudo_label
    return data


def load_data_only(dataset='cora', split_seed=42, device='cpu', split_percentage=[0.5, 0.25]):
    device = torch.device(device)
    data = load_data(dataset)
    train_mask, val_mask, test_mask = train_val_test_split(data.num_nodes, split_percentage, split_seed)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.to(device)
    return data

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, default=['cora'],help='datasets')
    # parser.add_argument('--gnn-models', nargs='+', type=str, default=['gcn'],help='GNNs')
    parser.add_argument('--eps-list', nargs='+', type=float, default=list(range(1, 9)), help='privacy budget')
    parser.add_argument('--methods', nargs='+', type=str, default=['init', 'modified'], help='blink protocol')
    # parser.add_argument('--variant', nargs='+', type=str, default=['hard', 'soft', 'hybrid'], help='variant')
    parser.add_argument('--delta-list', nargs='+', type=float, default=[0.1], help='delta for init blink')
    parser.add_argument('--sbm-ratio-list', nargs='+', type=float, default=[0.1], help='sbm_ratio for modified blink')
    parser.add_argument('--degree-ratio-list', nargs='+', type=float, default=[0.], help='degree-ratio for modified blink')
    parser.add_argument('--graph-seeds', nargs='+', type=int, default=list(range(2020, 2026)), help='seeds to disturb graphs')
    # parser.add_argument('--gnn-seeds', nargs='+', type=int, default=list(range(2020, 2026)), help='seeds to train gnns')
    # parser.add_argument('--best-hp-path', type=str, default=None, help='root for best delta, will disable delta-list and sbm-ratio-list')
    parser.add_argument('--split-seed', type=int, default=42, help='seed to split the dataset into train/val/test')
    parser.add_argument('--mlp-seed', type=int, default=2024, help='seed to train MLP')
    parser.add_argument('--tau', nargs='+', type=float, default=[0], help='tau for pseudo label of test nodes, 0 means using hard label, 1 means soft label, \inf means all set to 1/c')
    parser.add_argument('--split-percentage', nargs='+', type=float, default=[0.5, 0.25], help='dataset split percentage for val and test respectively')
    parser.add_argument('--save-root', type=str, default='./results', help='root to save results')
    parser.add_argument('--return-prior', action="store_true", help='if true, use prior, else use posterior')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    args = parser.parse_args()
    return args

def process(data, pij):
    # acc
    Aij = to_dense_adj(data.edge_index)[0]
    distance, TP, FP = torch.sum(torch.abs(Aij - pij)).item(), torch.sum(Aij * pij).item(), torch.sum((1 - Aij) * pij).item()
    FN = torch.sum(Aij).item() - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    # abs, true_positive, false_positive 
    acc = {'distance': distance, 'TP': TP, 'FP': FP, 'FN': FN, 'precision': precision, 'recall': recall, 'f1': f1}

    # sparsity
    spa = {'sparsity': torch.sum(pij).item()}

    # homophily
    label = data.y
    C = F.one_hot(label, num_classes=data.num_classes).to(pij.device).type(pij.dtype)
    M = C.T @ pij @ C
    cluster_num_nodes = torch.sum(C, dim=0)
    homo_linkx = 0.
    homo_mine = 0.
    for i in range(cluster_num_nodes.shape[0]):
        homo_mine += (M[i, i] / torch.sum(M[i])).item()
        homo_linkx += max(0, (M[i, i] / torch.sum(M[i]) - cluster_num_nodes[i] / data.num_nodes).item())
    homo_linkx = homo_linkx / (cluster_num_nodes.shape[0] - 1)
    homo_mine = homo_mine / (cluster_num_nodes.shape[0] - 1)
    homo = {'linkx': homo_linkx, 'mine': homo_mine}
    return [acc, spa, homo] 

def compose(lists):
    sum_dict = {}
    count_dict = {}
    for lst in lists:
        for entry in lst:
            for key, value in entry.items():
                if key not in sum_dict:
                    sum_dict[key] = 0.0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1
    average_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    return average_dict

def main(dataset='cora', model='gcn', dense=False, args=None, variant='hard'):
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d-%H-%M-%S')
    args = add_args()
    save_root = os.path.join(args.save_root, formatted_time)
    # save_root = './results/{}'.format(formatted_time)
    os.makedirs(save_root)

    if os.path.exists('./run.sh'):
        shutil.copyfile('./run.sh', os.path.join(save_root, 'run.sh'))

    return_prior = args.return_prior
    result = nested_defaultdict()
    if 'modified' in args.methods:
        method = 'modified'
        for dataset in args.datasets:
            for i in range(len(args.tau)):
                tau = args.tau[i]
                if tau < 1e-6:
                    use_soft = False
                else:
                    use_soft = True
                if i == 0:
                    data, mlp_acc = load_data_and_get_pseudo_label(dataset, args.split_seed, args.device, model_args, train_args, args.mlp_seed,    
                                                                    split_percentage=args.split_percentage, tau=tau)
                    # result['mlp'][dataset] = [mlp_acc, 0]
                else:
                    data = reload_data(data, tau)
                for eps in args.eps_list:
                    for sbm_ratio in args.sbm_ratio_list:
                        for degree_ratio in args.degree_ratio_list:
                            print('------------------{} {} {} {} {} {} {} {}------------------'.format(dataset, tau, model, eps, method, variant, sbm_ratio, degree_ratio))
                            blink_args = {'eps': eps, 'sbm_ratio': sbm_ratio, 'degree_ratio': degree_ratio}
                            lists = []
                            for graph_seed in args.graph_seeds:
                                pij = get_pij_modified(data=data.clone(), noise_seed=graph_seed, **blink_args, device=args.device, use_soft=use_soft, return_prior=return_prior)
                                lists.append(process(data, pij))
                        result[method][dataset][tau][eps][sbm_ratio][degree_ratio] = compose(lists)
                    result_ckp = defaultdict_to_dict(result)
                    with open(osp.join(save_root, 'result_ckp.json'), 'w') as f:
                        json.dump(result_ckp, f, indent=4)
    if 'init' in args.methods:
        method = 'init'
        for dataset in args.datasets:
            data = load_data_only(dataset, args.split_seed, args.device, split_percentage=args.split_percentage)
            for eps in args.eps_list:
                for delta in args.delta_list:
                    print('------------------{} {} {} {} {} {}------------------'.format(dataset, model, eps, method, variant, delta))
                    blink_args = {'eps': eps, 'delta': delta}
                    lists = []
                    for graph_seed in args.graph_seeds:
                        pij = get_pij_init(data=data.clone(), noise_seed=graph_seed, **blink_args, device=args.device, return_prior=return_prior)
                        lists.append(process(data, pij))
                    result[method][dataset][eps][delta] = compose(lists)
                result_ckp = defaultdict_to_dict(result)
                with open(osp.join(save_root, 'result_ckp.json'), 'w') as f:
                    json.dump(result_ckp, f, indent=4)
                                
    result = defaultdict_to_dict(result)
    with open(osp.join(save_root, 'result.json'), 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == '__main__':
    main()