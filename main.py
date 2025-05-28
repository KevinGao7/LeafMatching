import numpy as np
import pickle
import os
import torch
import warnings
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import *
import pandas as pd
import networkx as nx
import random
warnings.filterwarnings('ignore', 'divide by zero encountered in log2')
import torch
import argparse
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# >>> Dataset and Model <<< #
class MyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sta_ind = self.indices[idx]
        return sta_ind.clone().detach().to(dtype=torch.int64).cuda() 
class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, neg, gamma, alpha, epoch, batch_size, pos_ratio, 
                 chunks, convergence, eval_step):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.k = int(c*h)
        self.eps = eps
        self.neg = neg
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = epoch  
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.chunks = chunks
        self.convergence = convergence
        self.eval_step = eval_step
    # def __call__(self, graph):
    #     return self.forward(graph)
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=device)
        return torch.where(xor_results == 0, 
                           torch.tensor(0, dtype=torch.int64, device=device), 
                           torch.floor(torch.log2(xor_results.float())) + 1).int()
    def distance(self, coord1, coord2):
        xor_result = torch.bitwise_xor(coord1, coord2)
        return self.distance_lookup_table[xor_result]
    def generate_random_masks(self, sz):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=device)
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3)
    def connection(self, ori_int):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks
        random_masks = self.generate_random_masks(flipped_ints.size(0))
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        return torch.cat((result, ori_int.unsqueeze(1)), dim=1) 
    def p(self, dis, ig1, ig2):
        deg1, deg2 = self.degree[ig1], self.degree[ig2]        
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)[:,:,None,None]
        return 1/(1+aa**self.gamma/self.alpha)
    def p_test(self, dis, ig1, ig2):
        deg1, deg2 = self.degree[ig1], self.degree[ig2]        
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)
        return 1/(1+aa**self.gamma/self.alpha)
    def loom(self, epoch, sta_ind):
        pos_ind, lg, mask = self.neighbor_batch(sta_ind, epoch)
        sta_loc = self.locations[sta_ind]
        with torch.no_grad():
            pos_loc = torch.full((pos_ind.size(0), pos_ind.size(1), self.tp), -1, dtype=torch.int64, device=device)
            pos_loc[mask] = self.locations[pos_ind[mask]]
            cnc_loc = self.connection(self.locations[sta_ind])
            neg_ind = torch.full(pos_ind.size(), -1, dtype=torch.int64, device=device)
            neg_ind[mask] = torch.randint(0, self.num_nodes, (pos_ind.size(0), pos_ind.size(1)), dtype=torch.int64, device=device)[mask]
            neg_loc = torch.full((pos_ind.size(0), pos_ind.size(1), self.tp), -1, dtype=torch.int64, device=device)
            neg_loc[mask] = self.locations[neg_ind[mask]]
            indices = torch.randperm(cnc_loc.size(1))
            cnc_loc = cnc_loc[:, indices, :]
            mask1 = mask.unsqueeze(2).repeat(1, 1, self.tp)
            mask2 = mask1.unsqueeze(2).repeat(1, 1, cnc_loc.size(1),1)
            dis_sta_pos = torch.where(mask1, self.distance(sta_loc[:,None,:], pos_loc), -1)
            dis_sta_posum = torch.sum(dis_sta_pos, dim=-1)
            dis_sta_neg = torch.where(mask1, self.distance(sta_loc[:,None,:], neg_loc), -1)
            dis_sta_negum = torch.sum(dis_sta_neg, dim=-1)
            dis_pos_cnc = torch.where(mask2, self.distance(cnc_loc[:,None,:,:], pos_loc[:,:,None,:]), -1)
            dis_neg_cnc = torch.where(mask2, self.distance(cnc_loc[:,None,:,:], neg_loc[:,:,None,:]), -1)
            dis_new_pos = torch.where(mask2, (dis_pos_cnc-dis_sta_pos[:,:,None,:]+dis_sta_posum[:,:,None,None])/self.tp, 0)
            pos_loss = -torch.sum(torch.log(self.p(dis_new_pos, sta_ind[:,None], pos_ind)), dim=1) / lg[:,None,None]
            dis_new_neg = torch.where(mask2, (dis_neg_cnc-dis_sta_neg[:,:,None,:]+dis_sta_negum[:,:,None,None])/self.tp, 1000)
            neg_loss = -torch.sum(torch.log(1+self.eps-self.p(dis_new_neg, sta_ind[:,None], neg_ind)), dim=1) / lg[:,None,None]
            total_loss = self.pos_ratio * pos_loss + neg_loss
            index = torch.argmin(total_loss, dim=1)
            i_indices, j_indices = torch.meshgrid(torch.arange(sta_ind.size(0)), torch.arange(self.tp), indexing='ij')
            self.locations[sta_ind[i_indices], j_indices] = cnc_loc[i_indices, index[i_indices, j_indices], j_indices]
            tl = torch.mean(total_loss[i_indices, index[i_indices, j_indices], j_indices])
            pl = torch.mean(pos_loss[i_indices, index[i_indices, j_indices], j_indices])
            nl = torch.mean(neg_loss[i_indices, index[i_indices, j_indices], j_indices])
        return tl, pl, nl

    def get_neighbor(self):
        neighbor = [torch.tensor(list(self.G.neighbors(ii)), dtype=torch.int64, device=device) for ii in range(self.num_nodes)]
        neighbor_dict = {ii: neighbor[ii] for ii in range(self.num_nodes)}
        neighbor_tensor = torch.full((self.num_nodes, self.max_degree), -1, dtype=torch.int64, device=device)
        for n, nbs in neighbor_dict.items():
            neighbor_tensor[n, :len(nbs)] = nbs
        
        return neighbor_dict, neighbor_tensor
    def neighbor_batch(self, sta_ind, epoch):
        bs = sta_ind.size(0)
        batch_degree = self.degree[sta_ind] # (bs)
        batch_max_degree = batch_degree.max().item()
        batch_neighbor = self.neighbor_tensor[sta_ind, :batch_max_degree] # (bs, batch_max_degree), -1 for padding
        
        if epoch > convergence * self.epoch:
            batch_lengths = batch_degree
            batch_mask = batch_neighbor != -1
            
        else:
            random_probs = torch.rand(bs, device=device) # (bs, )
            choosing_mask = random_probs > 0.2 # (bs, ), 选全邻居还是随机 1 个邻居
            batch_lengths = torch.where(choosing_mask, batch_degree, 1)
            
            one_random_neighbor = (torch.rand(bs, device=device) * batch_degree).floor().long()
            random_neighbor_mask = torch.zeros((bs, batch_max_degree), dtype=torch.bool, device=device) # (bs, max_degree)
            random_neighbor_mask[torch.arange(bs), one_random_neighbor] = True # (bs, max_degree), 随机选一个邻居
            one_random_neighbor = torch.full((bs, batch_max_degree), -1, dtype=torch.int64, device=device)
            one_random_neighbor[:, 0] = batch_neighbor[random_neighbor_mask] # (bs, max_degree)            
            
            batch_neighbor = torch.where(choosing_mask.unsqueeze(1), 
                                         batch_neighbor,
                                         one_random_neighbor) # (bs, max_degree), mask 为 True 时选 full, mask为 False 时选 random
            batch_neighbor = batch_neighbor[:, :batch_lengths.max()]
            batch_mask = batch_neighbor != -1
        return batch_neighbor, batch_lengths, batch_mask
    
    
    def forward(self, graph, test_pos, test_neg):
        current_time = datetime.now()
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("start to load data")
        self.Go = nx.Graph()
        # print(graph.device)
        for edge in graph:
            self.Go.add_edge(edge[0].item(), edge[1].item())
        
        old_nodes = list(self.Go.nodes())
        self.mapping = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}
        num_old_nodes = max(old_nodes) + 10  # Ensure we accommodate all possible indices  
        self.bucket_tensor = torch.full((num_old_nodes,), -1, dtype=torch.long)  
        for new_node, old_node in enumerate(old_nodes):  
            self.bucket_tensor[old_node] = new_node 
        self.bucket_tensor = self.bucket_tensor.cuda()
        self.G = nx.relabel_nodes(self.Go, self.mapping)
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        print('num_nodes', self.num_nodes)
        print('num_edges', self.num_edges)
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.degree = torch.IntTensor([self.G.degree(n) for n in self.G.nodes()]).cuda().to(torch.int64)
        self.max_degree = self.degree.max()
        self.neighbor, self.neighbor_tensor = self.get_neighbor()
        self.locations = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
        self.li = torch.arange(self.num_nodes, dtype=torch.int64, device=device)[self.degree>0]
        dataset = MyDataset(self.li)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.get_neighbor()
        
        for epoch in range(self.epoch):    
            current_time = datetime.now()
            print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "pos+neg, pos, neg")
            sys.stdout.flush()
            total, positive, negative, ite = 0, 0, 0, 0
            for batch in dataloader:
                tot, pos, neg = self.loom(epoch, batch)
                total += tot
                positive += pos
                negative += neg
                ite += 1
            total /= ite
            positive /= ite
            negative /= ite
            print(epoch, total.device, total.item(), positive.item(), negative.item())
            if epoch % self.eval_step == 0 or epoch == self.epoch - 1:
                print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("start to eval")
                sys.stdout.flush()
                self.eval(test_pos, test_neg, epoch)
    
    def get_score(self, data):
        dt = data.t()
        dt1 = self.bucket_tensor[dt[0]]
        dt2 = self.bucket_tensor[dt[1]]
        N = dt1.shape[0]
        chunk_size = (N + self.chunks - 1) // self.chunks  # ceil division for 16 chunks
        pp_chunks = []
        for i in range(self.chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, N)
            if start >= end:
                break
            chunk_dt1 = dt1[start:end]
            chunk_dt2 = dt2[start:end]
            chunk_vector1 = self.locations[chunk_dt1]
            chunk_vector2 = self.locations[chunk_dt2]
            dis = self.distance(chunk_vector1, chunk_vector2).float()
            dis_mean = torch.mean(dis, dim=-1)
            pp_chunk = self.p_test(dis_mean, chunk_dt1, chunk_dt2)
            pp_chunks.append(pp_chunk)
        pp = torch.cat(pp_chunks, dim=0)
        return pp

    def eval(self, test_pos, test_neg, epoch):
        pos = self.get_score(test_pos)
        neg = self.get_score(test_neg)
        sorted_pos = torch.sort(pos)[0]
        sorted_neg = torch.sort(neg)[0]
        
        hit20 = eval_hits(sorted_pos, sorted_neg, 20)
        hit50 = eval_hits(sorted_pos, sorted_neg, 50)
        hit100 = eval_hits(sorted_pos, sorted_neg, 100)
        self.hits50_max = [hit50, epoch]
        self.hits100_max = [hit100, epoch]
        mrr1, mrr2 = eval_mrr(sorted_pos, sorted_neg)
        mrr = (mrr1 + mrr2) / 2
        roc_auc, pr_auc, f1 = eval_auc(sorted_pos, sorted_neg)
        
        print('hit20', hit20)
        print('hit50', hit50)
        print('hit100', hit100)
        print('roc_auc, pr_auc, f1, mrr', roc_auc, pr_auc, f1, mrr)
        

# >>> Data Loading Functions <<<
def get_train_pt(data_path, split_ratio):
    split_edges = torch.load(os.path.join(data_path, f'split_dict_{split_ratio}.pt'))
    return split_edges['train']['edge'].to(device).to(torch.int64)
def get_test_pt(data_path, split_ratio):
    split_edges = torch.load(os.path.join(data_path, f'split_dict_{split_ratio}.pt'))
    return split_edges['test']['edge'].to(device).to(torch.int64), split_edges['test']['edge_neg'].to(device).to(torch.int64)


# >>> Random Seed Setting <<<
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # 1. Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    
    # 2. Add arguments to the parser
    parser.add_argument("--dataset", type=str, default='ogbl_ppa')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--h", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--neg", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pos_ratio", type=float, default=1.0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--convergence", type=float, default=0.8)
    parser.add_argument("--eval_step", type=int, default=1)
    
    # 3. Parse the command-line arguments
    args = parser.parse_args()
    
    # 4. Access the parsed arguments
    dataset_name = args.dataset
    seed = args.seed
    epoch = args.epoch
    split_ratio = args.split_ratio
    alpha = args.alpha
    h = args.h
    gamma = args.gamma
    tp = args.tp
    c = args.c
    neg = args.neg
    batch_size = args.batch_size
    pos_ratio = args.pos_ratio
    chunks = args.chunks
    convergence = args.convergence
    eval_step = args.eval_step
    set_random_seed(seed)
    print(f"dataset_name={dataset_name}, seed={seed}, epoch={epoch}, split_ratio={split_ratio}, alpha={alpha}, h={h}, gamma={gamma}, tp={tp}, c={c}, neg={neg}, batch_size={batch_size}, pos_ratio={pos_ratio}, chunks={chunks}, convergence={convergence}, eval_step={eval_step}")
    
    # 5. Load the dataset
    if dataset_name.startswith("ogbl_"):
        if dataset_name == "ogbl_collab":
            data_path = f'./datasets/{dataset_name}/split/time'
        elif dataset_name == "ogbl_ppa":
            data_path = f'./datasets/{dataset_name}/split/throughput'       
    else:
        data_path = f'./datasets/{dataset_name}/split/'
        
    print("start to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train_data = get_train_pt(data_path, split_ratio)
    test_data = get_test_pt(data_path, split_ratio)
    print("end to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 6. Train and Test the model
    model = CritiGraph(h=h, tp=tp, c=c, eps=1e-5, neg=neg, gamma=gamma, 
                        alpha=alpha, epoch=epoch, batch_size=batch_size, pos_ratio=pos_ratio,
                        chunks=chunks, convergence=convergence, eval_step=eval_step)
    model(train_data, test_data[0], test_data[1])
    

    