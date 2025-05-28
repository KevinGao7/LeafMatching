import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

# erdos renyi graph
SEED = 42
np.random.seed(SEED)
for i in range(10, 20):

    num_node = 2**i
    test_ratio = 0.1
    neg_ratio = 1
    average_degree = 5
    p = average_degree/num_node

    # positive edges
    G = nx.erdos_renyi_graph(num_node, p, seed=SEED)
    pos_edge_set = set(G.edges())
    pos_edge_array = np.array(list(pos_edge_set))

        # split into train/test
    perm_indices = np.random.permutation(len(pos_edge_array))
    train_pos_len = int(len(pos_edge_array) * (1 - test_ratio))
    train_pos = pos_edge_array[perm_indices[:train_pos_len]]
    test_pos = pos_edge_array[perm_indices[train_pos_len:]]

        # filter out unobserved edges
    train_nodes = set(train_pos.flatten())
    test_pos_filtered = [edge for edge in test_pos if edge[0] in train_nodes and edge[1] in train_nodes]
    print("node nums:", 2 ** i, "train_pos_len:", train_pos_len, "len(test_pos):", len(test_pos), "len(test_pos_filtered):", len(test_pos_filtered))
    test_pos = np.array(test_pos_filtered)

    # negative edges
    test_neg_len = int(len(test_pos) * neg_ratio)
    test_neg = set()
    while len(test_neg) < test_neg_len:
        new_edge = (np.random.randint(0, num_node), np.random.randint(0, num_node))
        if new_edge not in pos_edge_set and new_edge[0] in train_nodes and new_edge[1] in train_nodes:
            test_neg.add(new_edge)

    # save
    savedir = f'./datasets/synthetic/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    train = {'edge': []}
    test = {'edge': [], 'edge_neg': []}
    valid = {'edge': [], 'edge_neg': []}

    print("processing train data")
    # train data
    for u, v in zip(train_pos[:, 0], train_pos[:, 1]):
        train['edge'].append([u.item(), v.item()])
        
    print("processing test data")
    # test data
    test['edge'] = test_pos
    test['edge_neg'] = list(test_neg)

    print("processing valid data")
    # valid data
    valid['edge'] = [[train['edge'][0][0], train['edge'][0][1]]]
    valid['edge_neg'] = [[train['edge'][0][0], train['edge'][0][1]]]
    
    print("processing complete")
    
    train['edge'] = torch.tensor(train['edge'])
    test['edge'] = torch.tensor(test['edge'])
    test['edge_neg'] = torch.tensor(test['edge_neg'])
    valid['edge'] = torch.tensor(valid['edge'])
    valid['edge_neg'] = torch.tensor(valid['edge_neg'])
    
    torch.save({
        'train': train,
        'test': test,
        'valid': valid
    }, os.path.join(savedir, f'ER_{num_node}_{average_degree}.pt'))