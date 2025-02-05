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
    def __init__(self, eps, delta, data, device='cpu') -> None:
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
        self.data = data.to(device)
        if delta == None:
            # do not privatize degree sequence
            self.priv_deg = False
            self.eps_a = eps
            self.eps_d = None
        else:
            self.priv_deg = True
            self.eps_d = eps * delta
            self.eps_a = eps * (1-delta)

    def AddLDP(self):
        
        n = self.data.num_nodes
        adj = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1], sparse_sizes=(n, n)).to(self.device).to_dense()
        deg = adj.sum(1).reshape(n, 1)

        def rr_adj() -> torch.Tensor:
            p = 1.0/(1.0+math.exp(self.eps_a))
            # return 1 with probability p, but does not flip diagonal edges since no self loop allowed
            res = ((adj + torch.bernoulli(torch.full((n, n), p)).to(self.device)) % 2).float()
            res.fill_diagonal_(0)
            return res

        def laplace_deg() -> torch.Tensor:
            return deg + torch.distributions.laplace.Laplace(loc=0, scale=1/self.eps_d).sample((n,1)).to(self.device)
        adj_rr = rr_adj()
        if self.priv_deg:
            return adj_rr, laplace_deg()
        else:
            return adj_rr, deg
        
class Server:
    def __init__(self, eps, delta, data, device) -> None:
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
        self.data = data.to(device)
        # no privacy
        if eps == None:
            self.priv = False
        else:
            self.priv = True
            if delta == None:
                # do not privatize degree sequence
                self.priv_deg = False
                self.eps_a = eps
                self.eps_d = None
            else:
                self.priv_deg = True
                self.eps_d = eps * delta
                self.eps_a = eps * (1-delta)
        self.n = data.num_nodes

    def receive(self, priv_adj, priv_deg):
        
        self.priv_adj = priv_adj.to(self.device)
        self.priv_deg = priv_deg.to(self.device)
        # project priv_deg to [1, n-2], otherwise resulting in useless prior = 0 or 1
        # This step is necessary for the MLE algorithm to run, but this makes the implementation to have higher MAE than the theoretical bound
        self.priv_deg[priv_deg < 1] = 1
        self.priv_deg[priv_deg > self.n - 2] = self.n - 2

    def estimate(self, return_prior=False):
        
        # store 1 vectors to save RAM
        ones_1xn = torch.ones(1,self.n).to(self.device)
        ones_nx1 = torch.ones(self.n,1).to(self.device)

        def estimate_prior():
            def phi(x):
                r = 1.0/(torch.exp(x).matmul(ones_1xn) + ones_nx1.matmul(torch.exp(-x).reshape(1,self.n)))
                return torch.log(self.priv_deg) - torch.log(r.sum(1).reshape(self.n,1) - r.diagonal().reshape(self.n,1))
            
            beta = torch.zeros(self.n, 1).to(self.device)

            # beta is a fixed point for phi
            for _ in range(200):
                beta = phi(beta)

            s = ones_nx1.matmul(beta.transpose(0,1)) + beta.matmul(ones_1xn)
            prior = torch.exp(s)/(1+torch.exp(s))
            prior.fill_diagonal_(0)
            return prior

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
        del ones_1xn
        del ones_nx1
        torch.cuda.empty_cache()
        return self.pij
    
def get_pij(data, eps, delta, noise_seed=2024, device=None, return_prior=False):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if eps == None:
        pij = None
        pass
    else:
        linkless_graph = data.clone()
        linkless_graph.edge_index = None
        client = Client(eps=eps, delta=delta, data=data, device=device)
        server = Server(eps=eps, delta=delta, data=linkless_graph, device=device)
        fix_seed(noise_seed)
        priv_adj, priv_deg = client.AddLDP()
        server.receive(priv_adj, priv_deg)
        pij = server.estimate(return_prior)  
    del server
    del client
    torch.cuda.empty_cache()
    return pij