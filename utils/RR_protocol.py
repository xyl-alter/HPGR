import torch
from torch_sparse import SparseTensor
import math
import numpy as np
import torch
from utils.dataloader import *
from utils.models import *
from utils.train import *
from sklearn.model_selection import train_test_split
import os
import json
import os.path as osp
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import argparse

class Client():
    def __init__(self, eps, data, device='cpu') -> None:
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
        self.data = data.to(device)
        self.eps_a = eps

    def send(self):
        
        n = self.data.num_nodes
        adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1], sparse_sizes=(n, n)).to(self.device).to_dense()

        def rr_adj() -> torch.Tensor:
            p = 1.0/(1.0+math.exp(self.eps_a))
            res = ((adj + torch.bernoulli(torch.full((n, n), p)).to(self.device)) % 2).float()
            res.fill_diagonal_(0)
            return res
        adj_rr = rr_adj()
        return adj_rr
        
class Server:
    def __init__(self, eps, data, device) -> None:
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
        self.data = data.to(device)
        self.eps_a = eps
        self.n = data.num_nodes

    def receive(self, priv_adj):
        
        self.priv_adj = priv_adj.to(self.device)

    def estimate(self, return_prior=False):

        def estimate_prior():
            return torch.ones((self.n, self.n), dtype=torch.float32, device=self.device) / 2
        
        def estimate_posterior(prior):
            p = 1.0/(1.0+np.exp(self.eps_a))
            priv_adj_t = self.priv_adj.transpose(0,1)
            x = self.priv_adj + priv_adj_t
            pr_y_edge = 0.5*(x-1)*(x-2)*p*p + 0.5*x*(x-1)*(1-p)*(1-p) - 1*x*(x-2)*p*(1-p)
            pr_y_no_edge = 0.5*(x-1)*(x-2)*(1-p)*(1-p) + 0.5*x*(x-1)*p*p - 1*x*(x-2)*p*(1-p)
            pij = pr_y_edge * prior / (pr_y_edge * prior + pr_y_no_edge * (1 - prior))
            return pij

        if return_prior:
            return estimate_prior()
        self.pij = estimate_posterior(estimate_prior())
        torch.cuda.empty_cache()
        return self.pij
    
def get_pij(data, eps, noise_seed=2024, device=None, return_prior=False, variant='hard'):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    linkless_graph = data.clone()
    linkless_graph.edge_index = None
    client = Client(eps=eps, data=data, device=device)
    server = Server(eps=eps, data=linkless_graph, device=device)
    fix_seed(noise_seed)
    priv_adj = client.send()
    server.receive(priv_adj)
    pij = server.estimate(return_prior)  
    del server
    del client
    torch.cuda.empty_cache()
    return pij