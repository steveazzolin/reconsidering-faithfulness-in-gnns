r"""Training pipeline: training/evaluation structure, batch training.
"""
import datetime
import os
import shutil
from typing import Dict
from typing import Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std
from munch import Munch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as sk_roc_auc
from imblearn.under_sampling import RandomUnderSampler

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.logger import pbar_setting
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.initial import reset_random_seed
import GOOD.kernel.pipelines.xai_metric_utils as xai_utils
from GOOD.utils.splitting import split_graph

pbar_setting["disable"] = True

class CustomDataset(InMemoryDataset):
    def __init__(self, root, samples, belonging, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.edge_types = {
            "inv": 0,
            "spu": 1,
            "added": 2,
            "BA": 3
        }
        
        data_list = []
        for i , G in enumerate(samples):
            if type(G) is nx.classes.digraph.DiGraph:
                data = from_networkx(G)
            else:
                if G.edge_index.shape[1] == 0:
                    raise ValueError("Empty intervened graph")
                data = Data(ori_x=G.ori_x.clone(), edge_index=G.edge_index.clone()) #G.clone()

            if not hasattr(data, "ori_x"):
                print(i, data, type(data))
                print(G.nodes(data=True))
            if len(data.ori_x.shape) == 1:
                data.ori_x = data.ori_x.unsqueeze(1)
            data.x = data.ori_x
            data.belonging = belonging[i]
            
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)


@register.pipeline_register
class Pipeline:
    r"""
    Kernel pipeline.

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]],
                 ood_algorithm: BaseOODAlg,
                 config: Union[CommonArgs, Munch]):
        super(Pipeline, self).__init__()
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.ood_algorithm: BaseOODAlg = ood_algorithm
        self.config: Union[CommonArgs, Munch] = config

    def train_batch(self, data: Batch, pbar) -> dict:
        r"""
        Train a batch. (Project use only)

        Args:
            data (Batch): Current batch of data.

        Returns:
            Calculated loss.
        """
        data = data.to(self.config.device)

        self.ood_algorithm.optimizer.zero_grad()

        mask, targets = nan2zero_get_mask(data, 'train', self.config)
        node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
        data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                             self.model.training,
                                                                             self.config)
        edge_weight = data.get('edge_norm') if self.config.model.model_level == 'node' else None

        model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)
        raw_pred = self.ood_algorithm.output_postprocess(model_output)

        loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config)
        loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config)

        self.ood_algorithm.backward(loss)

        return {'loss': loss.detach()}

    def train(self):
        r"""
        Training pipeline. (Project use only)
        """
        # config model
        print('Config model')
        self.config_model('train')

        # Load training utils
        print('Load training utils')
        self.ood_algorithm.set_up(self.model, self.config)

        # train the model
        for epoch in range(self.config.train.ctn_epoch, self.config.train.max_epoch):
            self.config.train.epoch = epoch
            print(f'Epoch {epoch}:')

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **pbar_setting)
            for index, data in pbar:
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue

                # train a batch
                train_stat = self.train_batch(data, pbar)
                mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        desc = f'ML: {mean_loss:.4f}|'
                        for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                            if not isinstance(spec_loss, dict):
                                spec_loss = dict()
                            if loss_name not in spec_loss.keys():
                                spec_loss[loss_name] = 0
                            spec_loss[loss_name] = (spec_loss[loss_name] * index + loss_value) / (index + 1)
                            desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                        pbar.set_description(desc[:-1])
                    else:
                        spec_loss = (spec_loss * index + self.ood_algorithm.spec_loss) / (index + 1)
                        pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
                else:
                    pbar.set_description(f'Loss: {mean_loss:.4f}')

            # Eval training score

            # Epoch val
            print('Evaluating...')
            if self.ood_algorithm.spec_loss is not None:
                if isinstance(self.ood_algorithm.spec_loss, dict):
                    desc = f'ML: {mean_loss:.4f}|'
                    for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                        desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                    print(f'Approximated ' + desc[:-1])
                else:
                    print(f'Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
            else:
                print(f'Approximated average training loss {mean_loss.cpu().item():.4f}')

            epoch_train_stat = self.evaluate('eval_train')
            id_val_stat = self.evaluate('id_val')
            id_test_stat = self.evaluate('id_test')
            val_stat = self.evaluate('val')
            test_stat = self.evaluate('test')

            # checkpoints save
            self.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, self.config)

            # --- scheduler step ---
            self.ood_algorithm.scheduler.step()

        print('\nTraining end.\n')
    

    @torch.no_grad()
    def get_subragphs_ratio(self, graphs, ratio, edge_scores):
        spu_subgraphs, causal_subgraphs, expl_accs, causal_masks = [], [], [], []
        if "CIGA" in self.config.model.model_name:
            norm_edge_scores = [e.sigmoid() for e in edge_scores]
        else:
            norm_edge_scores = edge_scores

        # Select relevant subgraph (bs = all)
        big_data = Batch().from_data_list(graphs)        
        (causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
            (spu_edge_index, spu_edge_attr, spu_edge_weight), mask = split_graph(
                big_data,
                torch.cat(edge_scores, dim=0),
                ratio,
                return_batch=True
            )
        
        cumnum = torch.tensor([g.num_nodes for g in graphs]).cumsum(0)
        cumnum[-1] = 0
        for j in range(causal_batch.max() + 1):
            causal_subgraphs.append(causal_edge_index[:, big_data.batch[causal_edge_index[0]] == j] - cumnum[j-1])
            spu_subgraphs.append(spu_edge_index[:, big_data.batch[spu_edge_index[0]] == j] - cumnum[j-1])
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[j], norm_edge_scores[j]) if hasattr(graphs[j], "edge_gt") else (np.nan,np.nan))
            causal_masks.append(mask[big_data.batch[big_data.edge_index[0]] == j])        
        return causal_subgraphs, spu_subgraphs, expl_accs, causal_masks

    @torch.no_grad()
    def evaluate_graphs(self, loader, log=False, **kwargs):
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval, belonging = [], []
        for data in pbar:
            data: Batch = data.to(self.config.device)            
            if log:
                output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, **kwargs)
            else:
                output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, **kwargs)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())
        preds_eval = torch.tensor(preds_eval)
        belonging = torch.tensor(belonging, dtype=int)
        return preds_eval, belonging

    def get_intervened_graph(self, graph, empty_idx=None, causal=None, source=None, idx=None):
        _, j, _ = idx
        
        G_t = graph.copy()
        G_t_filt = xai_utils.remove_from_graph(G_t, edge_index_to_remove=causal)
        xai_utils.mark_frontier(G_t, G_t_filt)

        if len(G_t_filt) == 0:
            empty_idx.add(j)
            return None

        G_union = xai_utils.random_attach_no_target_frontier(source, G_t_filt)
        return G_union

    def get_indices_dataset(self, dataset, extract_all=False):
        if self.config.numsamples_budget == "all" or self.config.numsamples_budget >= len(dataset) or extract_all:
            idx = np.arange(len(dataset))        
        elif self.config.numsamples_budget < len(dataset):        
            idx, _ = train_test_split(
                np.arange(len(dataset)),
                train_size=min(self.config.numsamples_budget, len(dataset)), # / len(dataset)
                random_state=42,
                shuffle=True,
                stratify=dataset.y if torch_geometric.__version__ == "2.4.0" else dataset.data.y
            )
        return idx
    
    @torch.no_grad()
    def compute_scores_and_graphs(self, ratios, splits, convert_to_nx=True, log=True, extract_all=False):
        reset_random_seed(self.config)
        self.model.eval()

        edge_scores, graphs, labels = {"train": [], "id_val": [], "id_test": [], "test": [], "val": []}, {"train": [], "id_val": [], "id_test": [], "test": [], "val": []}, {"train": [], "id_val": [], "id_test": [], "test": [], "val": []}
        causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
        graphs_nx, avg_graph_size = dict(), dict()
        for SPLIT in splits:
            dataset = self.get_local_dataset(SPLIT, log=log)
            
            idx = self.get_indices_dataset(dataset, extract_all=extract_all)
            loader = DataLoader(dataset[idx], batch_size=256, shuffle=False, num_workers=2)
            for data in loader:
                data = data.to(self.config.device)
                edge_score = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False,
                    return_attn=False,
                    ratio=None
                )
                    
                for j, g in enumerate(data.to_data_list()):
                    g.ori_x = data.ori_x[data.batch == j]
                    edge_scores[SPLIT].append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu())
                    graphs[SPLIT].append(g.detach().cpu())

                labels[SPLIT].extend(data.y.detach().cpu().numpy().tolist())
            labels[SPLIT] = torch.tensor(labels[SPLIT])
            avg_graph_size[SPLIT] = np.mean([g.edge_index.shape[1] for g in graphs[SPLIT]])
            
            if convert_to_nx:
                graphs_nx[SPLIT] = [to_networkx(g, node_attrs=["ori_x"], edge_attrs=["edge_attr"] if not g.edge_attr is None else None) for g in graphs[SPLIT]]
            else:
                graphs_nx[SPLIT] = list()
            
            for ratio in ratios:
                reset_random_seed(self.config)
                causal_subgraphs_r[SPLIT][ratio], spu_subgraphs_r[SPLIT][ratio], expl_accs_r[SPLIT][ratio], causal_masks[SPLIT][ratio] = self.get_subragphs_ratio(graphs[SPLIT], ratio, edge_scores[SPLIT])
                if log:
                    print(f"F1 for r={ratio} = {np.mean([e[1] for e in expl_accs_r[SPLIT][ratio]]):.3f}")
                    print(f"WIoU for r={ratio} = {np.mean([e[0] for e in expl_accs_r[SPLIT][ratio]]):.3f}")
        return (edge_scores, graphs, graphs_nx, labels, \
                avg_graph_size, causal_subgraphs_r, spu_subgraphs_r,  expl_accs_r, causal_masks)


    @torch.no_grad()
    def compute_metric_ratio(
        self,
        ratios,
        split,
        metric,
        edge_scores,
        graphs,
        graphs_nx,
        labels,
        avg_graph_size,
        causal_subgraphs_r,
        spu_subgraphs_r,
        expl_accs_r,
        causal_masks_r,
        intervention_distrib,
    ):
        assert metric in ["fidm", "nec", "fidp", "suff++"]
        assert intervention_distrib == "model_dependent"

        weights = ratios         
        if "sst2" in self.config.dataset.dataset_name.lower() and split in ("id_test", "id_val", "train") and not ("CIGA" in self.config.model.model_name):
            weights = [0.6, 0.9, 1.0]                

        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Computing {metric.upper()} over {split} across ratios")
        
        self.model.eval()   

        scores = defaultdict(list)
        for ratio in weights:
            reset_random_seed(self.config)
            print(f"\n\nratio={ratio}\n\n")            

            eval_samples, belonging, reference = [], [], []
            preds_ori, labels_ori, expl_acc_ori = [], [], []
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

                if metric == "nec":
                    for m in range(self.config.expval_budget): 
                        G_c = xai_utils.sample_edges_tensorized(
                            graphs[i],
                            nec_number_samples=self.config.nec_number_samples,
                            nec_alpha_1=self.config.nec_alpha_1,
                            avg_graph_size=avg_graph_size,
                            edge_index_to_remove=causal_masks_r[ratio][i],
                            sampling_type=self.config.samplingtype
                        )
                        belonging.append(i)
                        eval_samples.append(G_c)
                elif metric == "suff++":
                    if ratio == 1.0:
                        eval_samples.extend([graphs[i]]*self.config.expval_budget)
                        belonging.extend([i]*self.config.expval_budget)
                    else:
                        z, c = -1, 0
                        idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                        budget = self.config.expval_budget // 2
                        
                        while c < budget:
                            if z == len(idxs) - 1:
                                break
                            z += 1
                            j = idxs[z]
                            if j in empty_idx:
                                continue

                            G_union = self.get_intervened_graph(
                                graphs_nx[j],
                                empty_idx,
                                causal_subgraphs_r[ratio][j],
                                G_filt,
                                (i, j, c)
                            )

                            if G_union is None:
                                continue

                            eval_samples.append(G_union)
                            belonging.append(i)
                            c += 1
                        
                        # if not enough interventions, complement with sub-sampling
                        for _ in range(c, self.config.expval_budget):
                            G_c = xai_utils.sample_edges_tensorized(
                                graphs[i],
                                nec_number_samples=self.config.nec_number_samples,
                                nec_alpha_1=self.config.nec_alpha_1,
                                avg_graph_size=avg_graph_size,
                                edge_index_to_remove=~causal_masks_r[ratio][i],
                                sampling_type=self.config.samplingtype
                            )
                            belonging.append(i)
                            eval_samples.append(G_c)

            if len(eval_samples) == 0:
                print(f"\nZero intervened samples, skipping weight={ratio}")
                scores["all_KL"].append(1.0)
                scores["all_L1"].append(1.0)
                continue            
            
            if self.config.mask:
                print("Computing new predictions with masking")

                # Compute predictions on perturbed graphs and compute metric
                int_dataset = CustomDataset("", eval_samples, belonging)
                loader = DataLoader(int_dataset, batch_size=256, shuffle=False)
                preds_eval, belonging = self.evaluate_graphs(loader, log=False if "fid" in metric else True, weight=ratio, is_ratio=True, eval_kl=True)
            else:
                assert False, "Enable config.mask"
                
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

            aggr, aggr_std = self.get_aggregated_metric(metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging)
            
            for m in aggr.keys():
                assert aggr[m].shape[0] == labels_ori_ori.shape[0]
                scores[f"all_{m}"].append(round(aggr[m].mean().item(), 3))
                print(f"{metric.upper()} for r={ratio} all {m} = {scores[f'all_{m}'][-1]} +- {aggr[f'{m}'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")                                
        return scores


    def normalize_belonging(self, belonging):
        ret = []
        i = -1
        for j , elem in enumerate(belonging):
            if len(ret) > 0 and elem == belonging[j-1]:
                ret.append(i)
            else:
                i += 1
                ret.append(i)
        return ret


    def get_aggregated_metric(self, metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging):
        ret = {"KL": None, "L1": None}
        belonging = torch.tensor(self.normalize_belonging(belonging))

        if metric in ("suff++", "nec", ) and preds_eval.shape[0] > 0:
            div_kl = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
            div_l1 = torch.abs(preds_ori.exp() - preds_eval.exp()).sum(-1)

            if metric in ("suff++", ):
                ret["KL"] = torch.exp(-scatter_mean(div_kl, belonging, dim=0))
                ret["L1"] = torch.exp(-scatter_mean(div_l1, belonging, dim=0))
            elif metric in ("nec", ):
                ret["KL"] = 1 - torch.exp(-scatter_mean(div_kl, belonging, dim=0))
                ret["L1"] = 1 - torch.exp(-scatter_mean(div_l1, belonging, dim=0))
            aggr_std = scatter_std(div_l1, belonging, dim=0)
        elif "fid" in metric and preds_eval.shape[0] > 0:
            if preds_ori_ori.shape[1] == 1:
                l1 = torch.abs(preds_eval.reshape(-1) - preds_ori.reshape(-1))
            else:
                l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
            ret["L1"] = scatter_mean(l1, belonging, dim=0)
            aggr_std = scatter_std(l1, belonging, dim=0)                    
        else:
            raise ValueError(metric)
        return ret, aggr_std


    def get_local_dataset(self, split, log=True):
        if torch_geometric.__version__ == "2.4.0" and log:
            print(self.loader[split].dataset)
            print(f"Data example from {split}: {self.loader[split].dataset.get(0)}")
            print(f"Label distribution from {split}: {self.loader[split].dataset.y.unique(return_counts=True)}")        

        dataset = self.loader[split].dataset
        if abs(dataset.y.unique(return_counts=True)[1].min() - dataset.y.unique(return_counts=True)[1].max()) > 1000:
            print(f"#D#Unbalanced warning for {self.config.dataset.dataset_name} ({split})")
        if "hiv" in self.config.dataset.dataset_name.lower() and str(self.config.numsamples_budget) != "all":
            balanced_idx, _ = RandomUnderSampler(random_state=42).fit_resample(np.arange(len(dataset)).reshape(-1,1), dataset.y)

            dataset = dataset[balanced_idx.reshape(-1)]
            print(f"Creating balanced dataset: {dataset.y.unique(return_counts=True)}")
        return dataset


    @torch.no_grad()
    def evaluate(self, split: str, compute_suff=False):
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Args:
            split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
                'val', and 'test'.

        Returns:
            A score and a loss.

        """
        stat = {'score': None, 'loss': None}
        if self.loader.get(split) is None:
            return stat
        
        was_training = self.model.training
        self.model.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        likelihoods_all = []
        pbar = tqdm(self.loader[split], desc=f'Eval {split.capitalize()}', total=len(self.loader[split]),
                    **pbar_setting)
        for data in pbar:
            data: Batch = data.to(self.config.device)

            mask, targets = nan2zero_get_mask(data, split, self.config)
            if mask is None:
                return stat
            node_norm = torch.ones((data.num_nodes,),
                                   device=self.config.device) if self.config.model.model_level == 'node' else None
            data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                                 self.model.training,
                                                                                 self.config)
            model_output = self.model(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            raw_preds = self.ood_algorithm.output_postprocess(model_output)

            # --------------- Loss collection ------------------
            loss: torch.tensor = self.config.metric.loss_func(raw_preds, targets, reduction='none') * mask
            mask_all.append(mask)
            loss_all.append(loss)

            # ------------- Likelihood data collection ------------------
            if raw_preds.shape[-1] > 1:
                probs = raw_preds.softmax(dim=1)
                likelihoods_all.append(probs.gather(1, targets.unsqueeze(1)))
            else:
                probs = raw_preds.sigmoid()
                likelihoods_all.append(torch.full_like(probs, fill_value=-1))            

            # ------------- Score data collection ------------------
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            pred_all.append(pred)
            target_all.append(target)

        # ------- Loss calculate -------
        loss_all = torch.cat(loss_all)
        mask_all = torch.cat(mask_all)
        likelihoods_all = torch.cat(likelihoods_all)
        stat['loss'] = loss_all.sum() / mask_all.sum()
        stat['likelihood_avg'] = likelihoods_all.mean()
        stat['likelihood_prod'] = torch.prod(likelihoods_all)
        stat['likelihood_logprod'] = torch.sum(likelihoods_all.log())

        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat['score'] = eval_score(pred_all, target_all, self.config, self.loader[split].dataset.minority_class)

        print(f'{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}'
              f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

        if was_training:
            self.model.train()

        return {
            'score': stat['score'],
            'loss': stat['loss'],
            'likelihood_avg': stat['likelihood_avg'],
            'likelihood_prod': stat['likelihood_prod'],
            'likelihood_logprod': stat['likelihood_logprod'],
        }

    def load_task(self, load_param=False, load_split="ood"):
        r"""
        Launch a training or a test.
        """
        if self.task == 'train':
            self.train()
            return None, None
        elif self.task == 'test':
            # config model
            print('#D#Config model and output the best checkpoint info...')
            test_score, test_loss = self.config_model('test', load_param=load_param, load_split=load_split)
            return test_score, test_loss

    def config_model(self, mode: str, load_param=False, load_split="ood"):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.config.device)
        self.model.train()

        # load checkpoint
        if mode == 'train' and self.config.train.tr_ctn:
            ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'last.ckpt'))
            self.model.load_state_dict(ckpt['state_dict'])
            best_ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'best.ckpt'))
            self.config.metric.best_stat['score'] = best_ckpt['val_score']
            self.config.metric.best_stat['loss'] = best_ckpt['val_loss']
            self.config.train.ctn_epoch = ckpt['epoch'] + 1
            print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

        if mode == 'test':
            try:
                ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
            except FileNotFoundError:
                print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
                exit(1)
            if os.path.exists(self.config.id_test_ckpt):
                id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
                # model.load_state_dict(id_ckpt['state_dict'])
                print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {id_ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {id_ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {id_ckpt["test_loss"].item():.4f}\n')
                print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
                      f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["id_val_score"]:.4f} {ckpt["val_score"]:.4f}', end='')

            else:
                print(f'#IN#No In-Domain checkpoint.')
                # model.load_state_dict(ckpt['state_dict'])
                print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(
                    f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
            if load_param:
                if self.config.ood.ood_alg != 'EERM':
                    if load_split == "ood":
                        self.model.load_state_dict(ckpt['state_dict'])
                    elif load_split == "id":
                        self.model.load_state_dict(id_ckpt['state_dict'])
                    else:
                        raise ValueError(f"{load_split} not supported")
                else:
                    self.model.gnn.load_state_dict(ckpt['state_dict'])
            return ckpt["test_score"], ckpt["test_loss"]

    def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
                   test_stat: dir, config: Union[CommonArgs, Munch]):
        r"""
        Training util for checkpoint saving.

        Args:
            epoch (int): epoch number
            train_stat (dir): train statistics
            id_val_stat (dir): in-domain validation statistics
            id_test_stat (dir): in-domain test statistics
            val_stat (dir): ood validation statistics
            test_stat (dir): ood test statistics
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

        Returns:
            None

        """
        state_dict = self.model.state_dict() if config.ood.ood_alg != 'EERM' else self.model.gnn.state_dict()
        ckpt = {
            'state_dict': state_dict,
            'train_score': train_stat['score'],
            'train_loss': train_stat['loss'],
            'id_val_score': id_val_stat['score'],
            'id_val_loss': id_val_stat['loss'],
            'id_test_score': id_test_stat['score'],
            'id_test_loss': id_test_stat['loss'],
            'val_score': val_stat['score'],
            'val_loss': val_stat['loss'],
            'test_score': test_stat['score'],
            'test_loss': test_stat['loss'],
            'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
            'model': {
                'model name': f'{config.model.model_name} {config.model.model_level} layers',
                'dim_hidden': config.model.dim_hidden,
                'dim_ffn': config.model.dim_ffn,
                'global pooling': config.model.global_pool
            },
            'dataset': config.dataset.dataset_name,
            'train': {
                'weight_decay': config.train.weight_decay,
                'learning_rate': config.train.lr,
                'mile stone': config.train.mile_stones,
                'shift_type': config.dataset.shift_type,
                'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
            },
            'OOD': {
                'OOD alg': config.ood.ood_alg,
                'OOD param': config.ood.ood_param,
                'number of environments': config.dataset.num_envs
            },
            'log file': config.log_path,
            'epoch': epoch,
            'max epoch': config.train.max_epoch
        }
        if epoch < config.train.pre_train:
            return

        if not (config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better *
                config.metric.best_stat['score']
                or (id_val_stat.get('score') and (
                        config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
                    'score'] < config.metric.lower_better * config.metric.id_best_stat['score']))
                or epoch % config.train.save_gap == 0):
            return

        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
            print(f'#W#Directory does not exists. Have built it automatically.\n'
                  f'{os.path.abspath(config.ckpt_dir)}')
        saved_file = os.path.join(config.ckpt_dir, f'{epoch}.ckpt')
        torch.save(ckpt, saved_file)
        shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'last.ckpt'))

        # --- In-Domain checkpoint ---
        if id_val_stat.get('score') and (
                config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
            'score'] < config.metric.lower_better * config.metric.id_best_stat['score']):
            config.metric.id_best_stat['score'] = id_val_stat['score']
            config.metric.id_best_stat['loss'] = id_val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
            print('#IM#Saved a new best In-Domain checkpoint.')

        # --- Out-Of-Domain checkpoint ---
        # if id_val_stat.get('score'):
        #     if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
        #         return
        if config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better * \
                config.metric.best_stat['score']:
            config.metric.best_stat['score'] = val_stat['score']
            config.metric.best_stat['loss'] = val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
            print('#IM#Saved a new best checkpoint.')
        if config.clean_save:
            os.unlink(saved_file)


    def generate_plot_sampling_type(self, splits, ratios, sampling_alphas, graphs, graphs_nx, causal_subgraphs_r, causal_masks, avg_graph_size):
        def nec_kl(ori_pred, pred, belonging, labels):
            return 1 - torch.exp(-scatter_mean(torch.nn.KLDivLoss(reduction="none", log_target=True)(ori_pred, pred).sum(-1), belonging, dim=0))
        def nec_l1(ori_pred, pred, belonging, labels):
            return 1 - torch.exp(-scatter_mean(torch.abs(ori_pred.exp() - pred.exp()).sum(-1), belonging, dim=0))
        def fid_l1_div(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp() - pred.exp()).sum(-1).mean()
        def fid_model(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp().gather(1, ori_pred.argmax(-1).unsqueeze(1)) - pred.exp().gather(1, ori_pred.argmax(-1).unsqueeze(1))).mean()
        def fid_phenom(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp().gather(1, labels.long().unsqueeze(1)) - pred.exp().gather(1, labels.long().unsqueeze(1))).mean()
        def ratio_pred_change(ori_pred, pred, belonging, labels):
            return sum(ori_pred.argmax(-1) != pred.argmax(-1)) / pred.shape[0]
        def predict_sample(graphs, ratio):
            eval_set = CustomDataset("", graphs, torch.arange(len(graphs)))
            loader = DataLoader(eval_set, batch_size=256, shuffle=False, num_workers=0)
            preds, belonging = self.evaluate_graphs(loader, log=True, weight=ratio, is_ratio=True, eval_kl=True)
            return preds
    
        self.model.eval()

        metrics, accs = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        for SPLIT in splits:
            for ratio in ratios:
                ori_graphs = []
                G_sampled_rfid, G_sampled_deconf, G_sampled_fixed, G_sampled_deconf_R = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                belonging, labels = [], torch.tensor([])
                labels_norep = torch.tensor([])
                for idx in range(len(graphs[SPLIT])):
                    labels_norep = torch.cat((labels_norep, graphs[SPLIT][idx].y), dim=0)
                    for j in range(self.config.expval_budget):
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples=None, nec_alpha_1=alpha, avg_graph_size=None, sampling_type="bernoulli", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_rfid[alpha].append(sample)
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="prop_G_dataset", nec_alpha_1=alpha, avg_graph_size=avg_graph_size[SPLIT], sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_deconf[alpha].append(sample)
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="prop_R", nec_alpha_1=alpha, avg_graph_size=avg_graph_size[SPLIT], sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_deconf_R[alpha].append(sample)
                        
                        ori_graphs.append(graphs[SPLIT][idx])
                        belonging.append(idx)
                        labels = torch.cat((labels, graphs[SPLIT][idx].y), dim=0)

                labels = labels.view(-1)                
                belonging = torch.tensor(self.normalize_belonging(belonging), dtype=torch.long, device=labels.device)
                ori_pred = predict_sample(ori_graphs, ratio)
                divergences = {}                

                for k, alpha in enumerate(sampling_alphas):
                    k = k + 1 #for G_sampled_fixed
                    divergences[f"RFID_{alpha}"] = {}
                    divergences[f"DECONF_{alpha}"] = {}
                    divergences[f"DECONF_R_{alpha}"] = {}
                    preds = predict_sample(G_sampled_rfid[alpha] + G_sampled_deconf[alpha] + G_sampled_deconf_R[alpha], ratio)
                    pred_rfid   = preds[:len(G_sampled_rfid[alpha])]
                    pred_deconf = preds[len(G_sampled_rfid[alpha]): len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha])]
                    pred_deconf_R = preds[len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha]):]
                    
                    for metric_name, div_f in zip(["NEC L1"], [nec_kl, nec_l1]): #"FID L1 div", fid_l1_div
                        divergences[f"RFID_{alpha}"][metric_name]   = div_f(ori_pred, pred_rfid, belonging, labels).mean().item()
                        divergences[f"DECONF_{alpha}"][metric_name] = div_f(ori_pred, pred_deconf, belonging, labels).mean().item()
                        divergences[f"DECONF_R_{alpha}"][metric_name] = div_f(ori_pred, pred_deconf_R, belonging, labels).mean().item()
                        
                metrics[SPLIT][ratio] = divergences
                accs[SPLIT][ratio] = sum(labels == ori_pred.argmax(-1)) / ori_pred.shape[0]
        return metrics, accs

               
                