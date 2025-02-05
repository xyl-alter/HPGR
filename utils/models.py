import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import DenseGCNConv, GCNConv, DenseGraphConv, GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.utils import to_networkx
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn.dense.linear import Linear as pyg_Linear
import math
from typing import Any
from torch import Tensor
from torch.nn import Parameter

def get_model(model_name='gcn', dense=False, args=None):
    if model_name.lower() == 'gcn':
        if dense:
            model = DenseGCN(**args)
        else:
            model = GCN_Net(**args)
    elif model_name.lower() == 'graphsage':
        if dense:
            model = DenseGraphSAGE(**args)
        else:
            model = GraphSAGE(**args)
    elif model_name.lower() == 'gat':
        if dense:
            model = DenseGAT(**args)
        else:
            model = GAT(**args)
    elif model_name.lower() == 'mlp':
        model = MLP(**args)
    else:
        raise NotImplementedError
    return model

class GCN_Net(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(GCN_Net, self).__init__()

        self.num_layers = num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(GCNConv(num_features, hidden))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(GCNConv(hidden, hidden))
        if num_layers >= 2:
            self.conv_list.append(GCNConv(hidden, num_classes))
        
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        self.pred = F.log_softmax(x, dim=1)
        return self.pred
    
    def loss(self, label, mask):
        pred_loss = nn.NLLLoss(reduction='sum')(self.pred[mask], label[mask])
        return pred_loss

class DenseGCN(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(DenseGCN, self).__init__()

        self.num_layers = num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(DenseGCNConv(num_features, hidden))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(DenseGCNConv(hidden, hidden))
        if num_layers >= 2:
            self.conv_list.append(DenseGCNConv(hidden, num_classes))
        
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        self.pred = F.log_softmax(x, dim=1)[0]
        return self.pred
    
    def loss(self, label, mask):
        pred_loss = nn.NLLLoss(reduction='sum')(self.pred[mask], label[mask])
        return pred_loss
    
class MLP(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        conv_list = [Linear(num_features, hidden)]
        conv_list = conv_list + [Linear(hidden, hidden) for i in range(self.num_layers-2)] + [Linear(hidden, num_classes)]
        self.conv_list = torch.nn.ModuleList(conv_list)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers-1):
            x = F.relu(self.conv_list[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        self.emb = x
        x = self.conv_list[-1](x)
        self.pred_output = x
        return F.log_softmax(x, dim=1)
    
    def loss(self, label, mask):
        pred_loss = nn.CrossEntropyLoss(reduction='mean')(self.pred_output[mask], label[mask])
        return pred_loss
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(SAGEConv(num_features, hidden))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(SAGEConv(hidden, hidden))
        if num_layers >= 2:
            self.conv_list.append(SAGEConv(hidden, num_classes))
        
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        self.pred = F.log_softmax(x, dim=1)
        return self.pred
    
    def loss(self, label, mask):
        pred_loss = nn.NLLLoss(reduction='sum')(self.pred[mask], label[mask])
        return pred_loss
    
class DenseGraphSAGE(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(DenseGraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(DenseGraphConv(num_features, hidden, aggr='mean'))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(DenseGraphConv(hidden, hidden, aggr='mean'))
        if num_layers >= 2:
            self.conv_list.append(DenseGraphConv(hidden, num_classes, aggr='mean'))
        
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        self.pred = F.log_softmax(x, dim=1)[0]
        return self.pred
    
    def loss(self, label, mask):
        pred_loss = nn.NLLLoss(reduction='sum')(self.pred[mask], label[mask])
        return pred_loss
    
class GAT(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(GATConv(num_features, hidden))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(GATConv(hidden, hidden))
        if num_layers >= 2:
            self.conv_list.append(GATConv(hidden, num_classes))
        
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        self.pred = F.log_softmax(x, dim=1)
        return self.pred
    
    def loss(self, label, mask):
        pred_loss = nn.NLLLoss(reduction='sum')(self.pred[mask], label[mask])
        return pred_loss
    

def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

def constant(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)

class DenseGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2, dropout=0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = pyg_Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.att_src = Parameter(torch.Tensor(1, out_channels)) # C
        self.att_dst = Parameter(torch.Tensor(1, out_channels)) # C

        self.bias = Parameter(torch.Tensor(out_channels))    # C

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        constant(self.bias, 0.)
    
    def forward(self, x: Tensor, adj: Tensor, add_self_loops=True):
        N = len(adj)
        if add_self_loops:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[idx, idx] = 1

        out = self.lin(x) # [N, C]
        alpha_src = torch.sum(out * self.att_src, dim=-1) # [N]
        alpha_dst = torch.sum(out * self.att_dst, dim=-1) # [N]
        alpha = alpha_src.unsqueeze(1).repeat(1,N) + alpha_dst.unsqueeze(0).repeat(N,1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        alpha = adj * torch.exp(alpha)      # weighted/masked softmax, if adj=0, result would be zero
        alpha = alpha / alpha.sum(dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) # [N, N]

        out = torch.matmul((adj * alpha).transpose(0,1), out)

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
    
class DenseGAT(torch.nn.Module):
    def __init__(self, num_layers, num_features, hidden, num_classes, dropout):
        super(DenseGAT, self).__init__()

        self.num_layers = num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(DenseGATConv(num_features, hidden))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(DenseGATConv(hidden, hidden))
        if num_layers >= 2:
            self.conv_list.append(DenseGATConv(hidden, num_classes))
        
        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        self.pred = F.log_softmax(x, dim=1)
        return self.pred
    
    def loss(self, label, mask):
        pred_loss = nn.NLLLoss(reduction='sum')(self.pred[mask], label[mask])
        return pred_loss