import random
import networkx as nx
import numpy as np
import torch

from .. import DataWrapper
from cogdl.data import Graph


class EmbeddingLinkPredictionDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--negative-ratio", type=int, default=5)
        # fmt: on

    def __init__(self, dataset, negative_ratio, icews18=False):
        super(EmbeddingLinkPredictionDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.negative_ratio = negative_ratio
        self.train_data, self.test_data = None, None
        if icews18:
            self.split_ratio2time = {
                0.1: 31,
                0.2: 59,
                0.3: 86,
                0.4: 113,
                0.5: 143,
                0.6: 174,
                0.7: 205,
                0.8: 241,
                0.9: 272
            }

    def train_wrapper(self):
        return self.train_data

    def test_wrapper(self):
        return self.test_data

    def pre_transform(self, split_ratio):
        print("start to pre_transform")
        if hasattr(self.dataset.data, "edge_index"):
            row, col = self.dataset.data.edge_index
        else:
            row, col = self.dataset.data
        
        
        edge_list = list(zip(row.numpy(), col.numpy()))
        edge_set = set()
        for edge in edge_list:
            if (edge[0], edge[1]) not in edge_set and (edge[1], edge[0]) not in edge_set:
                edge_set.add(edge)
        edge_list = list(edge_set) # 此时肯定没有重边
        if hasattr(self, "split_ratio2time"):
            train_edges = []
            test_edges = []
            thr_t = self.split_ratio2time[split_ratio]
            for u, v, t in zip(row, col, self.dataset.data.t):
                t = t.item()
                if t < thr_t:
                    train_edges.append((u.item(), v.item()))
                else:
                    test_edges.append((u.item(), v.item()))
        else:
            train_edges, test_edges = divide_data(edge_list, [split_ratio, 1 - split_ratio]) # 没有重边
        
        test_true_edges, test_false_edges = gen_node_pairs(train_edges, test_edges, self.negative_ratio)
        train_edges = np.array(train_edges).transpose()
        train_edges = torch.from_numpy(train_edges)
        
        self.train_data = Graph(edge_index=train_edges)
        self.test_data = (test_true_edges, test_false_edges)
        
        print("end to pre_transform")

def divide_data(input_list, division_rate):
    print("start to divide data")
    local_division = len(input_list) * np.cumsum(np.array(division_rate))
    random.shuffle(input_list)
    print("end to divide data")
    return [
        input_list[int(round(local_division[i - 1])) if i > 0 else 0 : int(round(local_division[i]))]
        for i in range(len(local_division))
    ]
    


def randomly_choose_false_edges(nodes, true_edges, num):
    print("start to randomly_choose_false_edges")
    true_edges_set = set(true_edges)
    added_edges_set = set()
    tmp_list = list()
    all_flag = False
    for _ in range(num):
        if _ % (num // 100) == 0:
            print(f"Progress: {_}/{num}")
        trial = 0
        while True:
            x = nodes[random.randint(0, len(nodes) - 1)]
            y = nodes[random.randint(0, len(nodes) - 1)]
            trial += 1
            if trial >= 1000:
                all_flag = True
                break
            if x != y and (x, y) not in true_edges_set and (y, x) not in true_edges_set \
                      and (x, y) not in added_edges_set and (y, x) not in added_edges_set:
                tmp_list.append((x, y))
                added_edges_set.add((x, y))
                break
        if all_flag:
            break
    print("end to randomly_choose_false_edges")
    return tmp_list


def gen_node_pairs(train_data, test_data, negative_ratio=5):
    print("start to gen_node_pairs")
    G = nx.Graph()
    G.add_edges_from(train_data)

    training_nodes = set(list(G.nodes()))
    test_true_data = []
    for u, v in test_data:
        if u in training_nodes and v in training_nodes:
            test_true_data.append((u, v))
    test_false_data = randomly_choose_false_edges(list(training_nodes), 
                                                  train_data + test_data, 
                                                  len(test_true_data) * negative_ratio)
    print("end to gen_node_pairs")
    return (test_true_data, test_false_data)


def get_score(embs, node1, node2):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

'''

class EmbeddingLinkPredictionDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--negative-ratio", type=int, default=5)
        # fmt: on

    def __init__(self, dataset, negative_ratio):
        super(EmbeddingLinkPredictionDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        self.negative_ratio = negative_ratio
        self.train_data, self.test_data = None, None

    def train_wrapper(self):
        return self.train_data

    def test_wrapper(self):
        return self.test_data

    def pre_transform(self, split_ratio):
        row, col = self.dataset.data.edge_index
        edge_list = list(zip(row.numpy(), col.numpy()))
        edge_set = set()
        for edge in edge_list:
            if (edge[0], edge[1]) not in edge_set and (edge[1], edge[0]) not in edge_set:
                edge_set.add(edge)
        edge_list = list(edge_set)
        train_edges, test_edges = divide_data(edge_list, [split_ratio, 1 - split_ratio])
        self.test_data = gen_node_pairs(train_edges, test_edges, self.negative_ratio)
        train_edges = np.array(train_edges).transpose()
        train_edges = torch.from_numpy(train_edges)
        self.train_data = Graph(edge_index=train_edges)


def divide_data(input_list, division_rate):
    local_division = len(input_list) * np.cumsum(np.array(division_rate))
    random.shuffle(input_list)
    return [
        input_list[int(round(local_division[i - 1])) if i > 0 else 0 : int(round(local_division[i]))]
        for i in range(len(local_division))
    ]


def randomly_choose_false_edges(nodes, true_edges, num):
    true_edges_set = set(true_edges)
    tmp_list = list()
    all_flag = False
    for _ in range(num):
        trial = 0
        while True:
            x = nodes[random.randint(0, len(nodes) - 1)]
            y = nodes[random.randint(0, len(nodes) - 1)]
            trial += 1
            if trial >= 1000:
                all_flag = True
                break
            if x != y and (x, y) not in true_edges_set and (y, x) not in true_edges_set:
                tmp_list.append((x, y))
                break
        if all_flag:
            break
    return tmp_list


def gen_node_pairs(train_data, test_data, negative_ratio=5):
    G = nx.Graph()
    G.add_edges_from(train_data)

    training_nodes = set(list(G.nodes()))
    test_true_data = []
    for u, v in test_data:
        if u in training_nodes and v in training_nodes:
            test_true_data.append((u, v))
    test_false_data = randomly_choose_false_edges(list(training_nodes), 
                                                  train_data + test_data, 
                                                  len(test_true_data) * negative_ratio)
    return (test_true_data, test_false_data)


def get_score(embs, node1, node2):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
'''