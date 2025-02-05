import torch
import torch_geometric as pyg
from utils.dataloader import *
from utils.models import *
from utils.train import *
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
from tqdm import tqdm
import os
import math
from sklearn.model_selection import train_test_split
import os
import json
import os.path as osp
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import argparse

class Server():
    def __init__(self, data, eps, sbm_ratio, degree_ratio=0., mlp_seed=2024, device=None, use_soft=False):
        self.data = data
        self.use_soft = use_soft
        self.eps_sbm = (1 - degree_ratio) * sbm_ratio * eps
        self.eps_d = degree_ratio * sbm_ratio * eps
        self.eps_a = eps * (1 - sbm_ratio)
        self.mlp_seed = mlp_seed
        self.num_nodes = data.num_nodes
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.data.to(self.device)

    def get_pseudo_label(self, mlp_seed=2024):
        use_soft = self.use_soft
        try:
            if not use_soft:
                pseudo_label = self.data.pseudo_label
                self.pseudo_label = pseudo_label
                onehot_pseudo_label = F.one_hot(pseudo_label, num_classes=self.data.num_classes).type(torch.int32)
            else:
                onehot_pseudo_label = self.data.onehot_pseudo_label
        except Exception as e:
            print('data has no attribute pseudo label')
            raise AssertionError
        self.onehot_pseudo_label = onehot_pseudo_label
        self.group_nodes = torch.sum(onehot_pseudo_label, dim=0)
        return onehot_pseudo_label
    
    def send(self):
        return self.get_pseudo_label(self.mlp_seed)
    
    def receive(self, disturbed_sbm, disturbed_degree, adj_rr):
        self.disturbed_sbm = disturbed_sbm
        self.disturbed_degree = disturbed_degree
        self.adj_rr = adj_rr
        self.disturbed_group_edges = self.onehot_pseudo_label.T.type(disturbed_sbm.dtype) @ disturbed_sbm
        self.estimated_degree = torch.clamp(disturbed_degree, min=1, max=disturbed_degree.shape[0])

    def estimate_sbm_matrix(self):
        def gaussian_MLE(u1, u2, sigma1, sigma2):
            return (u1/sigma1**2 + u2/sigma2**2) / (1/sigma1**2 + 1/sigma2**2)
        estimated_group_edges = torch.zeros_like(self.disturbed_group_edges, device=self.device)
        for i in range(self.disturbed_group_edges.shape[0]):
            estimated_group_edges[i, i] = self.disturbed_group_edges[i, i]
            for j in range(i+1, self.disturbed_group_edges.shape[1]):
                sigma1 = torch.sqrt(self.group_nodes[i] * 2 * (1/self.eps_sbm) ** 2)
                sigma2 = torch.sqrt(self.group_nodes[j] * 2 * (1/self.eps_sbm) ** 2)
                estimated_group_edges[i, j] = estimated_group_edges[j, i] = gaussian_MLE(self.disturbed_group_edges[i, j], self.disturbed_group_edges[j, i], sigma1, sigma2)
        return estimated_group_edges
    
    def calculate_dc_sbm_parameters(self, estimated_group_edges, estimated_degree, node_groups):
        K = estimated_group_edges.size(0) 
        N = estimated_degree.size(0)      
        if not self.use_soft:
            group_total_degree = torch.zeros(K, dtype=torch.float32, device=self.device)
            for k in range(K):
                group_total_degree[k] = estimated_degree[node_groups == k].sum()
            
            theta = torch.zeros((K, K), dtype=torch.float32, device=self.device)
            for r in range(K):
                for s in range(K):
                    if group_total_degree[r] > 0 and group_total_degree[s] > 0:
                        theta[r, s] = estimated_group_edges[r, s] / (group_total_degree[r] * group_total_degree[s])
        else:
            normalize_mat = torch.linalg.inv(node_groups.T @ torch.diag(estimated_degree) @ node_groups)
            theta = normalize_mat @ estimated_group_edges @ normalize_mat
        return theta, estimated_degree
    
    def get_prior(self, theta, phi, pseudo_label):
        if not self.use_soft:
            labels_i = pseudo_label.unsqueeze(1) 
            labels_j = pseudo_label.unsqueeze(0) 
            phi_matrix = phi.unsqueeze(1) * phi.unsqueeze(0)
            theta_matrix = theta[labels_i, labels_j]
            prior = phi_matrix * theta_matrix
        else:
            n = phi.shape[0]
            onehot_pseudo_label = pseudo_label
            SBM = onehot_pseudo_label @ theta @ onehot_pseudo_label.T
            DC = phi.reshape(n, 1) @ phi.reshape(1, n)
            prior = DC * SBM
        return prior
    
    def check(self, prior, estimated_group_edges, estimated_degree, onehot_pseudo_label):
        if not self.use_soft:
            onehot_pseudo_label = F.one_hot(onehot_pseudo_label).type(torch.float32)
        print('degree:', torch.sum(torch.abs(torch.sum(prior, dim=1) - estimated_degree)))
        M_hat = onehot_pseudo_label.T @ prior @ onehot_pseudo_label
        print('homophily', torch.sum(torch.abs(M_hat - estimated_group_edges)))
        print('sum(prior), sum(degree), sum(M):', torch.sum(prior), torch.sum(estimated_degree), torch.sum(estimated_group_edges))

    def estimate_posterior(self, adj_rr, prior, eps):
        priv_adj = adj_rr
        p = 1.0/(1.0+np.exp(eps))
        priv_adj_t = priv_adj.transpose(0,1)
        x = priv_adj + priv_adj_t
        pr_y_edge = 0.5*(x-1)*(x-2)*p*p + 0.5*x*(x-1)*(1-p)*(1-p) - 1*x*(x-2)*p*(1-p)
        pr_y_no_edge = 0.5*(x-1)*(x-2)*(1-p)*(1-p) + 0.5*x*(x-1)*p*p - 1*x*(x-2)*p*(1-p)
        pij = pr_y_edge * prior / (pr_y_edge * prior + pr_y_no_edge * (1 - prior))
        return pij.clamp(min=0, max=1)
    
    def estimate(self, return_prior=False):
        # after receive()
        estimated_group_edges = self.estimate_sbm_matrix()
        theta, phi = self.calculate_dc_sbm_parameters(estimated_group_edges, self.estimated_degree, self.pseudo_label if not self.use_soft else self.onehot_pseudo_label)
        prior = self.get_prior(theta, phi, self.pseudo_label if not self.use_soft else self.onehot_pseudo_label)

        if return_prior:
            return prior.clamp(min=0)
        pij = self.estimate_posterior(self.adj_rr, prior, self.eps_a)
        return pij


class Client():
    def __init__(self, data, eps, sbm_ratio, degree_ratio=0., noise_seed=2024, device=None):
        self.data = data
        self.eps_sbm = (1 - degree_ratio) * sbm_ratio * eps
        print('sbm_ratio: {}, eps: {}, sbm_eps: {}'.format(sbm_ratio, eps, self.eps_sbm))
        self.eps_d = degree_ratio * sbm_ratio * eps
        self.noise_seed = noise_seed
        self.eps_a = eps * (1 - sbm_ratio)
        self.num_nodes = data.num_nodes
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
        self.data.to(self.device)

    def laplace_sbm_and_degree(self, onehot_pseudo_label):
        onehot_label = onehot_pseudo_label
        adj = to_dense_adj(self.data.edge_index)[0].type(torch.int32)
        sbm_mat = adj.type(onehot_label.dtype).cpu() @ onehot_label.cpu()
        disturbed_sbm = sbm_mat + torch.distributions.laplace.Laplace(loc=0, scale=1/self.eps_sbm).sample(sbm_mat.shape)
        disturbed_sbm = disturbed_sbm.to(self.device)
        if self.eps_d < 1e-6:
            disturbed_degree = torch.sum(disturbed_sbm, dim=1)
        else:
            degree = torch.sum(adj, dim=1)
            disturbed_degree = degree.to(self.device) + torch.distributions.laplace.Laplace(loc=0, scale=1/self.eps_d).sample(degree.shape).to(self.device)
            disturbed_degree = disturbed_degree
        return disturbed_sbm, disturbed_degree
    
    def rr_adj(self):
        adj = to_dense_adj(self.data.edge_index)[0]
        n = adj.shape[0]
        p = 1.0/(1.0+math.exp(self.eps_a))
        res = ((adj + torch.bernoulli(torch.full((n, n), p)).to(self.device)) % 2).float()
        res.fill_diagonal_(0)
        return res
    
    def receive(self, onehot_pseudo_label):
        self.onehot_pseudo_label = onehot_pseudo_label

    def send(self):
        pseudo_label = self.onehot_pseudo_label
        seed = self.noise_seed
        fix_seed(seed)
        disturbed_sbm, disturbed_degree = self.laplace_sbm_and_degree(pseudo_label)
        adj_rr = self.rr_adj()
        return disturbed_sbm, disturbed_degree, adj_rr

def get_pij(data, eps, sbm_ratio, degree_ratio, mlp_seed=2024, noise_seed=2024, device=None, return_prior=False, use_soft=False):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    server = Server(data, eps, sbm_ratio, degree_ratio, mlp_seed, device, use_soft)
    client = Client(data, eps, sbm_ratio, degree_ratio, noise_seed, device)
    onehot_pseudo_label = server.send()
    client.receive(onehot_pseudo_label)
    disturbed_sbm, disturbed_degree, adj_rr = client.send()
    server.receive(disturbed_sbm, disturbed_degree, adj_rr)
    pij = server.estimate(return_prior)
    del server
    del client
    torch.cuda.empty_cache()
    return pij