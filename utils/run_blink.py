import torch
from torch_sparse import SparseTensor
import math
import numpy as np
import torch
from utils.dataloader import *
from utils.models import *
from utils.train import *
import os
import json
import os.path as osp
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import argparse
from utils.init_blink_protocol import get_pij as get_pij_init
from utils.modified_blink_protocol import get_pij as get_pij_modified
from utils.RR_protocol import get_pij as get_pij_rr
from utils.hagei_protocol import get_pij as get_pij_hagei
from torch_geometric.utils.sparse import dense_to_sparse

def sample_graph(pij, variant):
    adj = None
    est_edge_index = None
    assert variant in ['hard', 'soft', 'hybird', 'sample']
    num_edges = 0
    if variant == 'hard': # hard version has no edge weights and must be sparse GNNs
        est_edge_index = (pij > 0.5).float().to_sparse().coalesce().indices()
        num_edges = est_edge_index.shape[1]
    elif variant == 'soft': # soft version and model is dense version, we feed in a adjacency matrix
        adj = pij
        num_edges = adj.shape[0] * adj.shape[1]
    elif variant == 'sample':
        adj = torch.bernoulli(pij)
        est_edge_index = dense_to_sparse(adj)[0]
        num_edges = est_edge_index.shape[1]
    else:
        num_edges = pij.sum().long().item()
        kth_p = torch.topk(pij.flatten(), k=pij.sum().long().item()).values[-1].item()
        adj = pij * (pij >= kth_p).float()
    print('num_edges: {}'.format(num_edges))
    return est_edge_index, adj


def train_once(data, method='init', blink_args=None, model_args=None, model='gcn', variant='hard', 
               train_args={'max_epoches': 500, 'lr': 0.001, 'weight_decay': 5e-4, 'eval_interval': 20}, 
               graph_seed=2024, gnn_seeds=[2024], device=None, use_soft=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    data = data.clone()
    fix_seed(graph_seed)
    if method == 'init':
        pij = get_pij_init(data=data, noise_seed=graph_seed, **blink_args)
    elif method == 'modified':
        pij = get_pij_modified(data=data, noise_seed=graph_seed, **blink_args, device=device, use_soft=use_soft)
    elif method == 'rr':
        pij = get_pij_rr(data=data, noise_seed=graph_seed, **blink_args)
    elif method == 'hagei':
        pij = get_pij_hagei(data=data, noise_seed=graph_seed, **blink_args, device=device)
    else:
        raise NotImplementedError
    edge_index, adj = sample_graph(pij, variant)
    data.edge_index = edge_index
    data.adj = adj
    model_args['num_features'] = data.x.shape[1]
    model_args['num_classes'] = data.num_classes
    # model_args['hidden'] = 16 if data.num_classes < 16 else 32
    if variant in ['hard', 'sample']:
        dense = False
    else:
        dense = True
    model = get_model(model, dense, model_args)
    model.to(device)
    data.to(device)
    acc_list = []
    for train_seed in gnn_seeds:
        acc = train_warpper(data=data, model=model, mode='edge_index' if variant in ['hard', 'sample'] else 'adj', seed=train_seed, **train_args)
        acc_list.append(acc)
    print(acc_list)
    del data
    del model
    torch.cuda.empty_cache()
    return np.mean(acc_list)