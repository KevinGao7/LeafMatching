import torch
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np


def eval_hits(prob_pos: torch.Tensor, prob_neg: torch.Tensor, K=100):
    if len(prob_neg) < K:
        return 1.0
    kth_score_in_negative_edges = prob_neg[-K]
    hitsK_right = torch.searchsorted(prob_pos, kth_score_in_negative_edges, right=True)
    histK_left = torch.searchsorted(prob_pos, kth_score_in_negative_edges, right=False)
    hitsK = (hitsK_right + histK_left)/2
    hitsK_score = float(1 - hitsK/len(prob_pos))
    return hitsK_score

def eval_mrr(prob_pos: torch.Tensor, prob_neg: torch.Tensor):
    inds_opt = torch.searchsorted(prob_neg, prob_pos, right=True)
    mr_opt = 1/(len(prob_neg) - inds_opt + 1)
    mrr_opt = torch.mean(mr_opt)
    inds_pess = torch.searchsorted(prob_neg, prob_pos, right=False)
    mr_pess = 1/(len(prob_neg) - inds_pess + 1)
    mrr_pess = torch.mean(mr_pess)

    return float(mrr_opt), float(mrr_pess)

def eval_auc(prob_pos: torch.Tensor, prob_neg: torch.Tensor):
    prob_pos_numpy = prob_pos.detach().cpu().numpy()
    prob_neg_numpy = prob_neg.detach().cpu().numpy()

    prob_all = np.concatenate([prob_pos_numpy, prob_neg_numpy])
    true_all = np.concatenate([np.ones(len(prob_pos_numpy)), np.zeros(len(prob_neg_numpy))]).astype(np.int32)

    rocauc = roc_auc_score(true_all, prob_all)
    ap = average_precision_score(true_all, prob_all) # also PRAUC

    sorted_prob = deepcopy(prob_all)
    sorted_prob.sort()
    threshold = sorted_prob[-len(prob_pos_numpy)]
    pred_all = (prob_all >= threshold).astype(np.int32)

    f1 = f1_score(true_all, pred_all)

    return rocauc, ap, f1

    