from torch_geometric.datasets import Planetoid
from .train import fix_seed
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import FacebookPagePage, LastFMAsia, Amazon, WikipediaNetwork
def load_cora(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = Planetoid(root='../data/Cora/', name='Cora')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_pubmed(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = Planetoid(root='../data/', name='pubmed')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_photo(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = Amazon(root='../data/', name='Photo')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_citeseer(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = Planetoid(root='../data/', name='citeseer')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_facebook(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = FacebookPagePage(root='../data/facebook')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_lastfm(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = LastFMAsia(root='../data/LastFMAsia')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_chameleon(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = WikipediaNetwork(root='../data/', name='chameleon')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_computers(split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    dataset = Amazon(root='../data/', name='Computers')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    if split:
        if split_seed is not None:
            fix_seed(split_seed)
        data = RandomNodeSplit('test_rest', num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)(data)
    return data

def load_data(dataset_name, split=False, split_seed=None, num_train_per_class=20, num_val=500, num_test=1000):
    if dataset_name.lower() == 'cora':
        return load_cora(split, split_seed, num_train_per_class, num_val, num_test)
    elif dataset_name.lower() == 'citeseer':
        return load_citeseer(split, split_seed, num_train_per_class, num_val, num_test)
    elif dataset_name.lower() == 'pubmed':
        return load_pubmed(split, split_seed, num_train_per_class, num_val, num_test)
    elif dataset_name.lower() == 'facebook':
        return load_facebook(split, split_seed, num_train_per_class, num_val, num_test)
    elif dataset_name.lower() == 'lastfm':
        return load_lastfm(split, split_seed, num_train_per_class, num_val, num_test)
    elif dataset_name.lower() == 'photo':
        return load_photo(split, split_seed, num_train_per_class, num_val, num_test)
    elif dataset_name.lower() == 'chameleon':
        return load_chameleon(split, split_seed, num_train_per_class, num_val, num_test)
    elif dataset_name.lower() == 'computers':
        return load_computers(split, split_seed, num_train_per_class, num_val, num_test)
    else:
        raise NotImplementedError('{} not implemented!'.format(dataset_name))