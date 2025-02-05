import torch
import numpy as np
import random
import time
import copy
from tqdm import tqdm

def rand_train_val_test_split(data, num_train_per_class=20, num_val=500, num_test=1000):
    """ randomly splits label into train/valid/test splits """
    data.train_mask.fill_(False)
    for c in range(data.num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        # print(f"CLASS {c}/{data.num_classes-1}: Sample [{num_train_per_class}] Edges out of [{len(idx)}].")
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data.val_mask.fill_(False)
    data.val_mask[remaining[:num_val]] = True

    data.test_mask.fill_(False)
    if num_test is None or num_test <= 0:   # NOTE: test_set = node_set - train_set - val_set
        data.test_mask[remaining[num_val:]] = True
    else:   # NOTE: #(test_set) = num_test
        data.test_mask[remaining[num_val:num_val + num_test]] = True
        
def fix_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(data, model, opt, mode='edge_index'):
    model.train()
    opt.zero_grad()
    x, label, edge_index, train_mask = data.x, data.y, data.edge_index, data.train_mask
    if mode == 'edge_index':
        adj = edge_index
    elif mode == 'adj':
        adj = data.adj
    pred = model(x, adj)
    loss = model.loss(label, train_mask)
    # with torch.autograd.detect_anomaly():
    loss.backward()
    opt.step()
    return loss

def val(data, model, mode='edge_index'):
    model.eval()
    with torch.no_grad():
        x, label, edge_index, val_mask, train_mask, test_mask = data.x, data.y, data.edge_index, data.val_mask, data.train_mask, data.test_mask
        if mode == 'edge_index':
            adj = edge_index
        elif mode == 'adj':
            adj = data.adj
        pred = model(x, adj).detach()
        outcome = (torch.argmax(pred, dim=1) == label)
        train_acc = torch.sum(outcome[train_mask]) / len(outcome[train_mask])
        val_acc = torch.sum(outcome[val_mask]) / len(outcome[val_mask])
        test_acc = torch.sum(outcome[test_mask]) / len(outcome[test_mask])
    return {'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc}

def print_eval_result(eval_result, prefix=''):
    if prefix:
        prefix = prefix + ' '
    print(f"{prefix}"
        f"Train Acc:{eval_result['train_acc']*100:6.2f} | "
        f"Val Acc:{eval_result['val_acc']*100:6.2f} | "
        f"Test Acc:{eval_result['test_acc']*100:6.2f}")
    
def train_warpper(data, model, max_epoches, lr=0.001, weight_decay=5e-4, eval_interval=10, early_stopping=100, early_stopping_tolerance=1, seed=2024, mode='edge_index'):
    # torch.autograd.set_detect_anomaly(True)
    fix_seed(seed)
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    val_acc_history = []
    start_time = time.time()
    # for epoch in tqdm(range(1, max_epoches+1)):
    for epoch in range(1, max_epoches+1):
        loss = train(data, model, opt, mode=mode)
        if epoch % eval_interval == 0:
            eval_result = val(data, model, mode=mode)
            # print_eval_result(eval_result, prefix=f'[Epoch {epoch:3d}/{max_epoches:3d}]')
            if eval_result['val_acc'] > best_val_acc:
                best_val_acc = eval_result['val_acc']
                best_model_param = copy.deepcopy(model.state_dict())
            val_acc_history.append(eval_result['val_acc'])
            if early_stopping > 0 and len(val_acc_history) > early_stopping:
                    mean_val_acc = torch.tensor(
                        val_acc_history[-(early_stopping + 1):-1]).mean().item()
                    if (eval_result['val_acc'] - mean_val_acc) * 100 < - early_stopping_tolerance: # NOTE: in percentage
                        print('[Early Stop Info] Stop at Epoch: ', epoch)
                        break
    train_time = time.time() - start_time
    model.load_state_dict(best_model_param)
    eval_result = val(data, model, mode=mode)
    # print_eval_result(eval_result, prefix=f'[Final Result] Time: {train_time:.2f}s |')
    return eval_result['test_acc'].detach().cpu().item()