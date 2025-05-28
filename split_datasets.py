import csv
import logging
import os
import os.path as osp
import sys
import random
sys.path.append("/home/gaochi/OurDataset/CogDL")

import numpy as np
import pandas as pd
import torch
from CogDL.cogdl.datasets.ogb import OGBLPpaDataset, OGBLCollabDataset
from cogdl.datasets.planetoid_data import (CiteSeerDataset, CoraDataset,
                                           PubMedDataset)
from CogDL.cogdl.wrappers.data_wrapper.link_prediction.embedding_link_prediction_dw import \
    EmbeddingLinkPredictionDataWrapper
from torch_geometric.datasets import ICEWS18
import argparse

np.random.seed(0)
random.seed(0)

def load_icews18(root):
    train_dataset = ICEWS18(root, split='train')
    val_dataset = ICEWS18(root, split='val')
    test_dataset = ICEWS18(root, split='test')
    
    dataset = train_dataset
    dataset.data.sub = torch.cat([train_dataset.data.sub, val_dataset.data.sub, test_dataset.data.sub], dim=0)
    dataset.data.obj = torch.cat([train_dataset.data.obj, val_dataset.data.obj, test_dataset.data.obj], dim=0)
    dataset.data.rel = torch.cat([train_dataset.data.rel, val_dataset.data.rel, test_dataset.data.rel], dim=0)
    dataset.data.t = torch.cat([train_dataset.data.t, val_dataset.data.t, test_dataset.data.t], dim=0)
        
    edge_index = torch.tensor(torch.zeros((dataset.data.sub.shape[0], 2)), dtype=torch.int64)
    for i, (u, v) in enumerate(zip(dataset.data.sub, dataset.data.obj)):
        edge_index[i][0] = u
        edge_index[i][1] = v
    dataset.data.edge_index = edge_index.t()
    return dataset

def transform_pt(train_data, test_data):
    train = {'edge': []}
    test = {'edge': [], 'edge_neg': []}
    valid = {'edge': [], 'edge_neg': []}

    print("processing train data")
    # train data
    for u, v in zip(train_data.edge_index[0], train_data.edge_index[1]):
        train['edge'].append([u.item(), v.item()])
        
    print("processing test data")
    # test data
    test['edge'] = test_data[0]
    test['edge_neg'] = test_data[1]
    
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
    
    logging.info(f"""Train Edges: {train['edge'].shape[0]}, 
                 Test Pos Edges: {test['edge'].shape[0]}, Test Neg Edges: {test['edge_neg'].shape[0]}""")
    
    return train, test, valid

def split_dataset(name, split_ratio):
    if name == 'ogbl_ppa':
        dataset = OGBLPpaDataset("./datasets")
        split_subdataset_path = osp.join("./datasets", name, 'split', 'throughput')
    elif name == 'ogbl_collab':
        dataset = OGBLCollabDataset("./datasets")
        split_subdataset_path = osp.join("./datasets", name, 'split', 'time')
    elif name == 'Cora':
        dataset = CoraDataset("./datasets")
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'CiteSeer':
        dataset = CiteSeerDataset("./datasets")
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'PubMed':
        dataset = PubMedDataset("./datasets")
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'icews18':
        dataset = load_icews18(root = "./datasets/icews18")
        split_subdataset_path = osp.join("./datasets", name, 'split')
    
    logging.info(f"Dataset: {name}, Split Subdataset Path: {split_subdataset_path}")
    datawrapper = EmbeddingLinkPredictionDataWrapper(dataset, negative_ratio=5, icews18=(name == 'icews18'))
    logging.info(f"Splitting ratio: {split_ratio}")

    # >>> Get the train/test split >>> #
    datawrapper.pre_transform(split_ratio)
    train_data = datawrapper.train_wrapper()
    test_data = datawrapper.test_wrapper()
    
    # >>> Transform data to PyTorch tensors >>> #
    train, test, valid = transform_pt(train_data, test_data)
    
    # >>> Save the split data >>> #
    if not osp.exists(split_subdataset_path):
        os.makedirs(split_subdataset_path)
    
    torch.save({'train': train, 'test': test, 'valid': valid}, 
               osp.join(split_subdataset_path, f'split_dict_{split_ratio}.pt'))
    
    logging.info(f"""Train Edges: {train['edge'].shape[0]}, 
                 Test Pos Edges: {test['edge'].shape[0]}, Test Neg Edges: {test['edge_neg'].shape[0]}""")
    logging.info(f"""Train/Test Pos Edges: {train['edge'].shape[0] / test['edge'].shape[0]}""")
    logging.info(f"""Train Edges/Nodes: {len(train['edge'])} / {len(torch.unique(train['edge']))} 
                 = {len(train['edge']) / len(torch.unique(torch.unique(train['edge'])))}\n""")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Dataset')
    parser.add_argument('--split_ratio', type=float, default=0.02)
    parser.add_argument('--dataset', type=str, required=True)
    
    args = parser.parse_args()
    split_ratio = args.split_ratio
    dataset = args.dataset
    
    logging.basicConfig(filename=f'./datasets/logs/split_{dataset}_ratio_{split_ratio}.log',
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    split_dataset(dataset, split_ratio)
    
    
    
