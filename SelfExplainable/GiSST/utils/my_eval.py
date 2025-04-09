
import datetime
import os
import shutil
from typing import Dict
from typing import Union
import random
from collections import defaultdict
from scipy.stats import pearsonr

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max
from munch import Munch

# from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, sort_edge_index, shuffle_node
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as sk_roc_auc, f1_score, accuracy_score


@torch.no_grad()
def compute_metric_ratio(
    split: str,
    metric: str,
    edge_scores,
    graphs,
    graphs_nx,
    labels,
    avg_graph_size,
    causal_subgraphs_r, 
    spu_subgraphs_r,
    expl_accs_r,
    causal_masks_r,
    intervention_bank,
    intervention_distrib:str = "model_dependent",
    debug=False,
):
    assert metric in ["suff", "fidm", "nec", "nec++", "fidp", "suff++", "suff_simple"] # nec e suf++ 
    assert intervention_distrib == "model_dependent"
    #################### Anon!!!!
    # causal_subgraphs_r  ==> ratios = [.3, .6, .9, 1.] torch tensor edge index
    # causal_masks_r ==> mask binaria sul edg ind del grafo orig 1 if in causal_subgraph 0 otehrwise

    if "CIGA" in self.config.model.model_name:
        is_ratio = True
        weights = [self.model.att_net.ratio]
    else:
        is_ratio = True
        if "sst2" in self.config.dataset.dataset_name.lower() and split in ("id_val", "train"):
            weights = [0.6, 0.9, 1.0]
        else:
            weights = [0.3, 0.6, 0.9, 1.0]

    print(f"\n\n")
    print("-"*50)
    print(f"\n\n#D#Computing {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")
    reset_random_seed(self.config)
    self.model.eval()  

    scores, results, acc_ints = defaultdict(list), {}, []
    for ratio in weights:
        reset_random_seed(self.config)
        print(f"\n\nratio={ratio}\n\n")            

        eval_samples, belonging, reference = [], [], []
        preds_ori, labels_ori, expl_acc_ori = [], [], []
        effective_ratio = [causal_subgraphs_r[ratio][i].shape[1] / (causal_subgraphs_r[ratio][i].shape[1] + spu_subgraphs_r[ratio][i].shape[1] + 1e-5) for i in range(len(spu_subgraphs_r[ratio]))]
        empty_idx = set()

        pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
        for i in pbar:
            if graphs[i].edge_index.shape[1] <= 6:
                continue                
            if metric in ("suff", "suff++") and intervention_distrib == "model_dependent":
                G = graphs_nx[i].copy()
                G_filt = xai_utils.remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[ratio][i])
                num_elem = xai_utils.mark_frontier(G, G_filt)
                if len(G_filt) == 0 or num_elem == 0:
                    continue
                # G = G_filt # P(Y|G) vs P(Y|R)

            eval_samples.append(graphs[i])
            reference.append(len(eval_samples) - 1)
            belonging.append(-1)
            labels_ori.append(labels[i])
            expl_acc_ori.append(expl_accs_r[ratio][i])

            if metric in ("fidm", "fidp", "nec", "nec++") or len(empty_idx) == len(graphs) or intervention_distrib in ("fixed", "bank"):
                if metric in ("suff", "suff++", "suff_simple") and intervention_distrib in ("fixed", "bank") and i == 0:
                    print(f"Using {intervention_distrib} interventional distribution")
                elif metric in ("suff", "suff++", "suff_simple") and intervention_distrib == "model_dependent":
                    # print("Empty graphs for SUFF. Rolling-back to FIDM")
                    pass

                for m in range(self.config.expval_budget): 
                    G_c = xai_utils.sample_edges_tensorized(
                        graphs[i],
                        nec_number_samples=self.config.nec_number_samples,
                        nec_alpha_1=self.config.nec_alpha_1,
                        avg_graph_size=avg_graph_size,
                        edge_index_to_remove=causal_masks_r[ratio][i],
                        sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                    )
                    belonging.append(i)
                    eval_samples.append(G_c)
            elif metric == "suff" or metric == "suff++" or metric == "suff_simple":
                if ratio == 1.0:
                    eval_samples.extend([graphs[i]]*self.config.expval_budget)
                    belonging.extend([i]*self.config.expval_budget)
                else:
                    z, c = -1, 0
                    idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                    budget = self.config.expval_budget
                    if metric == "suff++":
                        budget = budget // 2
                    if metric == "suff_simple":
                        budget = 0 # skip interventions and just pick subsamples
                    while c < budget:
                        if z == len(idxs) - 1:
                            break
                        z += 1
                        j = idxs[z]
                        if j in empty_idx:
                            continue

                        G_union = self.get_intervened_graph(
                            metric,
                            intervention_distrib,
                            graphs_nx[j],
                            empty_idx,
                            causal_subgraphs_r[ratio][j],
                            spu_subgraphs_r[ratio][j],
                            G_filt,
                            debug,
                            (i, j, c),
                            feature_intervention=False,
                            feature_bank=None
                        )
                        if G_union is None:
                            continue
                        eval_samples.append(G_union)
                        belonging.append(i)
                        c += 1
                    for k in range(c, self.config.expval_budget): # if not enough interventions, pad with sub-sampling
                        # G_c = xai_utils.sample_edges(G, "spu", self.config.fidelity_alpha_2)
                        G_c = xai_utils.sample_edges_tensorized(
                            graphs[i],
                            nec_number_samples=self.config.nec_number_samples,
                            nec_alpha_1=self.config.nec_alpha_1,
                            avg_graph_size=avg_graph_size,
                            edge_index_to_remove=~causal_masks_r[ratio][i],
                            sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                        )
                        belonging.append(i)
                        eval_samples.append(G_c)

        if len(eval_samples) == 0:
            print(f"\nZero intervened samples, skipping weight={ratio}")
            for c in labels_ori_ori.unique():
                scores[c.item()].append(1.0)
            scores["all_KL"].append(1.0)
            scores["all_L1"].append(1.0)
            continue

        # # Inspect edge_scores of intervened edges
        # self.debug_edge_scores(int_dataset, reference, ratio)            
        # Compute new prediction and evaluate KL
        int_dataset = CustomDataset("", eval_samples, belonging)

        loader = DataLoader(int_dataset, batch_size=256, shuffle=False)
        if self.config.mask:
            print("Computing with masking")
            preds_eval, belonging = self.evaluate_graphs(loader, log=False if "fid" in metric else True, weight=ratio, is_ratio=is_ratio, eval_kl=True)
        else:
            assert False
            preds_eval, belonging = self.evaluate_graphs(loader, log=False if "fid" in metric else True, eval_kl=True)
        preds_ori = preds_eval[reference]

        mask = torch.ones(preds_eval.shape[0], dtype=bool)
        mask[reference] = False
        preds_eval = preds_eval[mask]
        belonging = belonging[mask]            
        assert torch.all(belonging >= 0), f"{torch.all(belonging >= 0)}"

        labels_ori_ori = torch.tensor(labels_ori)
        preds_ori_ori = preds_ori
        preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
        labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

        aggr, aggr_std = self.get_aggregated_metric(metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging, results, ratio)

        for m in aggr.keys():
            assert aggr[m].shape[0] == labels_ori_ori.shape[0]
            for c in labels_ori_ori.unique():
                idx_class = np.arange(labels_ori_ori.shape[0])[labels_ori_ori == c]
                scores[c.item()].append(round(aggr[m][idx_class].mean().item(), 3))
            scores[f"all_{m}"].append(round(aggr[m].mean().item(), 3))

        assert preds_ori_ori.shape[1] > 1, preds_ori_ori.shape
        dataset_metric = self.loader["id_val"].dataset.metric
        if dataset_metric == "ROC-AUC":
            if not "fid" in metric:
                preds_ori_ori = preds_ori_ori.exp() # undo the log
                preds_eval = preds_eval.exp()
            acc = sk_roc_auc(labels_ori_ori.long(), preds_ori_ori[:, 1], multi_class='ovo')
            acc_int = sk_roc_auc(labels_ori.long(), preds_eval[:, 1], multi_class='ovo')
        elif dataset_metric == "F1":
            acc = f1_score(labels_ori_ori.long(), preds_ori_ori.argmax(-1), average="binary", pos_label=dataset.minority_class)
            acc_int = f1_score(labels_ori.long(), preds_eval.argmax(-1), average="binary", pos_label=dataset.minority_class)
        else:
            if preds_ori_ori.shape[1] == 1:
                assert False
                if not "fid" in metric:
                    preds_ori_ori = preds_ori_ori.exp() # undo the log
                    preds_eval = preds_eval.exp()
                preds_ori_ori = preds_ori_ori.round().reshape(-1)
                preds_eval = preds_eval.round().reshape(-1)
            acc = (labels_ori_ori == preds_ori_ori.argmax(-1)).sum() / (preds_ori_ori.shape[0])
            acc_int = (labels_ori == preds_eval.argmax(-1)).sum() / preds_eval.shape[0]

        acc_ints.append(acc_int)
        print(f"\nModel {dataset_metric} of binarized graphs for r={ratio} = ", round(acc.item(), 3))
        print(f"Model XAI F1 of binarized graphs for r={ratio} = ", np.mean([e[1] for e in expl_accs_r[ratio]]))
        print(f"Model XAI WIoU of binarized graphs for r={ratio} = ", np.mean([e[0] for e in expl_accs_r[ratio]]))
        print(f"len(reference) = {len(reference)}")
        print(f"Effective ratio: {np.mean(effective_ratio):.3f} +- {np.std(effective_ratio):.3f}")
        if preds_eval.shape[0] > 0:
            print(f"Model {dataset_metric} over intervened graphs for r={ratio} = ", round(acc_int.item(), 3))
            for c in labels_ori_ori.unique().numpy().tolist():
                print(f"{metric.upper()} for r={ratio} class {c} = {scores[c][-1]} +- {aggr['KL'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")
                del scores[c]
            for m in aggr.keys():
                print(f"{metric.upper()} for r={ratio} all {m} = {scores[f'all_{m}'][-1]} +- {aggr[f'{m}'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")
    return scores, acc_ints, results
