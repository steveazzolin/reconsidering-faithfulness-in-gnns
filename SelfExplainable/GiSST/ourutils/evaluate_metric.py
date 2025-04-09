
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import ourutils.xai_metric_utils as xai_utils
import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max
from datetime import datetime

def evaluate_metric(args):
    #load_splits = ["id"]
    splits = ["val", "test"] # val in realta è un id_val
    ratios = [.3, .6, .9, 1.]
    startTime = datetime.now()

    metrics_score = {}

    print("\n\n" + "-"*50)

    metrics_score["id"] = {s: defaultdict(list) for s in splits + ["test", "test_R"]}
    for i, seed in enumerate(args.seeds.split("/")): # tutti i seed
        seed = int(seed)        
        args.random_seed = seed
        args.exp_round = seed
        


        config = config_summoner(args) # load eval.json     
        model, loader = initialize_model_dataset(config) # carica modello e dataset


        if not (len(args.metrics.split("/")) == 1 and args.metrics.split("/")[0] == "acc"):
            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = compute_scores_and_graphs( #verifica
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
            )
            intervention_bank = None

        for metric in args.metrics.split("/"):
            print(f"\n\nEvaluating {metric.upper()} for seed {seed}\n")

            for split in splits:
                score, acc_int, results = compute_metric_ratio(
                    split,
                    metric=metric,
                    intervention_distrib=config.intervention_distrib,
                    intervention_bank=intervention_bank,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    graphs_nx=graphs_nx[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                    causal_subgraphs_r=causal_subgraphs_r[split],
                    spu_subgraphs_r=spu_subgraphs_r[split],
                    expl_accs_r=expl_accs_r[split],
                    causal_masks_r=causal_masks_r[split]
                )
                metrics_score["id"][split][metric].append(score)
                metrics_score["id"][split][metric + "_acc_int"].append(acc_int)
    

    if not os.path.exists(f"storage/metric_results/aggregated_id_results_{config.log_id}.json"):
        with open(f"storage/metric_results/aggregated_id_results_{config.log_id}.json", 'w') as file:
            file.write("{}")
    with open(f"storage/metric_results/aggregated_id_results_{config.log_id}.json", "r") as jsonFile: #legge il file
        results_aggregated = json.load(jsonFile)

    for load_split in ["id"]:
        print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                if "acc" == metric:
                    continue
                for div in ["L1", "KL"]:
                    s = [
                        metrics_score[load_split][split][metric][i][f"all_{div}"] for i in range(len(metrics_score[load_split][split][metric]))
                    ]
                    print_metric(metric + f" class all_{div}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, metric+f"_{div}"])
                print_metric(metric + "_acc_int", metrics_score[load_split][split][metric + "_acc_int"], results_aggregated, key=[config.dataset.dataset_name+" "+config.dataset.domain, config.model.model_name, split, metric+"_acc_int"])
                
        if "acc" in args.metrics.split("/"):
            for split in splits + ["test", "test_R"]:
                print(f"\nEval split {split}")
                print_metric("acc", metrics_score[load_split][split]["acc"])
                for a in ["plaus", "wiou"]:
                    for c in metrics_score[load_split][split][a][0].keys():
                        s = [
                            metrics_score[load_split][split][a][i][c] for i in range(len(metrics_score[load_split][split][a]))
                        ]
                        print_metric(a + f" class {c}", s)

        print("\n\n", "-"*50, "\nComputing faithfulness")
        for split in splits:
            print(f"\nEval split {split}")            
            for div in ["L1", "KL"]:
                if "suff" in args.metrics.split("/") and "nec" in args.metrics.split("/"):                
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_gmean_{div}"])

                if "suff++" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff++"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_gmean_{div}"])
                
                if "suff_simple" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff_simple"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_gmean_{div}"])

                if "suff" in args.metrics.split("/") and "nec++" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
                    necpp = get_tensorized_metric(metrics_score[load_split][split]["nec++"], f"all_{div}")[:, :suff.shape[1]]
                    faith_aritm = aritm(suff, necpp)
                    faith_armonic = armonic(suff, necpp)
                    faith_gmean = gmean(suff, necpp)
                    print_metric(f"Faith.++ Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith++_aritm_{div}"])
                    print_metric(f"Faith.++ Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith++_gmean_{div}"])
                    print_metric(f"Faith.++ GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith++_gmean_{div}"])

        print(f"Computed for split load_split = {load_split}\n\n\n")
        
@torch.no_grad()
def compute_metric_ratio(
    model,
    config,
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

    
    is_ratio = True
    weights = [0.3, 0.6, 0.9, 1.0]

    print(f"\n\n")
    print("-"*50)
    print(f"\n\n#D#Computing {metric.upper()} over {split} across ratios (random_expl={config.random_expl})")
    #TODO
    #reset_random_seed(self.config)
    model.eval()   
    

    scores, results, acc_ints = defaultdict(list), {}, []
    for ratio in weights:
        #TODO
        #reset_random_seed(self.config)
        print(f"\n\nratio={ratio}\n\n")            

        eval_samples, belonging, reference = [], [], []
        preds_ori, labels_ori, expl_acc_ori = [], [], []
        effective_ratio = [causal_subgraphs_r[ratio][i].shape[1] / (causal_subgraphs_r[ratio][i].shape[1] + spu_subgraphs_r[ratio][i].shape[1] + 1e-5) for i in range(len(spu_subgraphs_r[ratio]))]
        empty_idx = set()

        pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores))
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

                for m in range(config.expval_budget): 
                    G_c = xai_utils.sample_edges_tensorized(
                        graphs[i],
                        nec_number_samples=config.nec_number_samples,
                        nec_alpha_1=config.nec_alpha_1,
                        avg_graph_size=avg_graph_size,
                        edge_index_to_remove=causal_masks_r[ratio][i],
                        sampling_type="bernoulli" if metric in ("fidm", "fidp") else config.samplingtype
                    )
                    belonging.append(i)
                    eval_samples.append(G_c)
            elif metric == "suff" or metric == "suff++" or metric == "suff_simple":
                if ratio == 1.0:
                    eval_samples.extend([graphs[i]]*config.expval_budget)
                    belonging.extend([i]*config.expval_budget)
                else:
                    z, c = -1, 0
                    idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                    budget = config.expval_budget
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
                        G_union = get_intervened_graph(
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
                    for k in range(c, config.expval_budget): # if not enough interventions, pad with sub-sampling
                        G_c = xai_utils.sample_edges_tensorized(
                            graphs[i],
                            nec_number_samples=config.nec_number_samples,
                            nec_alpha_1=config.nec_alpha_1,
                            avg_graph_size=avg_graph_size,
                            edge_index_to_remove=~causal_masks_r[ratio][i],
                            sampling_type="bernoulli" if metric in ("fidm", "fidp") else config.samplingtype
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
            
        int_dataset = CustomDataset("", eval_samples, belonging)

        loader = DataLoader(int_dataset, batch_size=256, shuffle=False)
        if config.mask: #?
            print("Computing with masking")
            preds_eval, belonging = evaluate_graphs(model,loader, log=False if "fid" in metric else True, weight=ratio, is_ratio=is_ratio, eval_kl=True, config=config)
#         else:
#             assert False
#             preds_eval, belonging = evaluate_graphs(model,loader, log=False if "fid" in metric else True, eval_kl=True)
        
        preds_ori = preds_eval[reference]


        mask = torch.ones(preds_eval.shape[0], dtype=bool)
        mask[reference] = False
        preds_eval = preds_eval[mask]
        belonging = belonging[mask]            
        assert torch.all(belonging >= 0), f"{torch.all(belonging >= 0)}"

        labels_ori_ori = torch.tensor(labels_ori)
        preds_ori_ori = preds_ori
        preds_ori = preds_ori_ori.repeat_interleave(config.expval_budget, dim=0)
        labels_ori = labels_ori_ori.repeat_interleave(config.expval_budget, dim=0)

        aggr, aggr_std = get_aggregated_metric(metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging, results, ratio)

        for m in aggr.keys():
            assert aggr[m].shape[0] == labels_ori_ori.shape[0]
            for c in labels_ori_ori.unique():
                idx_class = np.arange(labels_ori_ori.shape[0])[labels_ori_ori == c]
                scores[c.item()].append(round(aggr[m][idx_class].mean().item(), 3))
            scores[f"all_{m}"].append(round(aggr[m].mean().item(), 3))

        assert preds_ori_ori.shape[1] > 1, preds_ori_ori.shape
        # note!!!!!!!!
        #dataset_metric = self.loader["id_val"].dataset.metric
        dataset_metric = config.dataset_metric
        
        if dataset_metric == "ROC-AUC":
            if not "fid" in metric:
                preds_ori_ori = preds_ori_ori.exp() # undo the log
                preds_eval = preds_eval.exp()
            acc = sk_roc_auc(labels_ori_ori.long(), preds_ori_ori[:, 1], multi_class='ovo')
            acc_int = sk_roc_auc(labels_ori.long(), preds_eval[:, 1], multi_class='ovo')
        elif dataset_metric == "F1":
            acc = f1_score(labels_ori_ori.long(), preds_ori_ori.argmax(-1), average="binary", pos_label=minority_class)
            acc_int = f1_score(labels_ori.long(), preds_eval.argmax(-1), average="binary", pos_label=minority_class)
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


        
def get_intervened_graph( metric, intervention_distrib, graph, empty_idx=None, causal=None, spu=None, source=None, debug=None, idx=None, bank=None, feature_intervention=False, feature_bank=None):
    i, j, c = idx
    if metric == "fidm" or (metric == "suff" and intervention_distrib == "model_dependent" and causal is None):
        return xai_utils.sample_edges(graph, "spu", config.fidelity_alpha_2, spu)
    elif metric in ("nec", "nec++", "fidp"):
        if metric == "nec++":
            alpha = max(config.nec_alpha_1 - 0.1 * (j // 3), 0.1)
        else:
            alpha = config.nec_alpha_1
        return xai_utils.sample_edges(graph, alpha, deconfounded=True, edge_index_to_remove=causal)
        # return xai_utils.sample_edges_tensorized(graph, k=1, edge_index_to_remove=causal, sampling_type="deconfounded")

    elif metric == "suff" and intervention_distrib == "bank":
        assert False
        G = graph.copy()
        I = bank[j].copy()
        ret = nx.union(G, I, rename=("", "T"))
        for n in range(random.randint(3, max(10, int(len(I) / 2)))):
            s_idx = random.randint(0, len(G) - 1)
            t_idx = random.randint(0, len(I) - 1)
            u = str(list(G.nodes())[s_idx])
            v = "T" + str(list(I.nodes())[t_idx])
            ret.add_edge(u, v, origin="added")
            ret.add_edge(v, u, origin="added")
        return ret
    elif metric == "suff" and intervention_distrib == "fixed":
        # random attach fixed graph to the explanation
        G = graph.copy()

        I = nx.DiGraph(nx.barabasi_albert_graph(random.randint(5, max(len(G), 8)), random.randint(1, 3)), seed=42) #BA1 -> nx.barabasi_albert_graph(randint(5, max(len(G), 8)), randint(1, 3))
        nx.set_edge_attributes(I, name="origin", values="spu")
        nx.set_node_attributes(I, name="x", values=[1.0])
        print("remebder to check values here for non-motif datasets")
        # nx.set_node_attributes(I, name="frontier", values=False)

        ret = nx.union(G, I, rename=("", "T"))
        for n in range(random.randint(3, max(10, int(len(G) / 2)))):
            s_idx = random.randint(0, len(G) - 1)
            t_idx = random.randint(0, len(I) - 1)
            u = str(list(G.nodes())[s_idx])
            v = "T" + str(list(I.nodes())[t_idx])
            ret.add_edge(u, v, origin="added")
            ret.add_edge(v, u, origin="added")
        return ret
    else:
        G_t = graph.copy()
        G_t_filt = xai_utils.remove_from_graph(G_t, edge_index_to_remove=causal)
        num_elem = xai_utils.mark_frontier(G_t, G_t_filt)

        if len(G_t_filt) == 0:
            empty_idx.add(j)
            return None

        if feature_intervention:
            if i == 0 and j == 0:
                print(f"Applying feature interventions with alpha = {config.feat_int_alpha}")
            G_t_filt = xai_utils.feature_intervention(G_t_filt, feature_bank, config.feat_int_alpha)

        G_union = xai_utils.random_attach_no_target_frontier(source, G_t_filt)
    return G_union

@torch.no_grad()
def evaluate_graphs(model,loader,config, log=False, **kwargs):
    #pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader))
    preds_eval, belonging = [], []
    for data in loader:
        data: Batch = data.to(config.device)            
        if log:
            #output = model.log_probs(data=data, edge_weight=None, ood_algorithm=ood_algorithm, **kwargs)
            #ood_algorithm not done           
            output = model.log_probs(x=data.x, edge_index=data.edge_index, batch=data.batch, **kwargs) #add edge_weight Anon
        else:
            #output = model.probs(data=data, edge_weight=None, ood_algorithm=ood_algorithm, **kwargs)
            output = model.probs(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_weight=None, **kwargs)
            #ood_algorithm not done
            
        preds_eval.extend(output.detach().cpu().numpy().tolist())
        belonging.extend(data.belonging.detach().cpu().numpy().tolist())
    preds_eval = torch.tensor(preds_eval)
    belonging = torch.tensor(belonging, dtype=int)
    return preds_eval, belonging

def get_aggregated_metric(metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging, results, ratio):
    ret = {"KL": None, "L1": None}
    belonging = torch.tensor(normalize_belonging(belonging))

    if metric in ("suff", "suff++", "nec", "nec++", "suff_simple") and preds_eval.shape[0] > 0:
        div_kl = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
        div_l1 = torch.abs(preds_ori.exp() - preds_eval.exp()).sum(-1)

        # results[ratio] = div_l1.numpy().tolist()
        if metric in ("suff", "suff++", "suff_simple"):
            ret["KL"] = torch.exp(-scatter_mean(div_kl, belonging, dim=0)) # on paper
            ret["L1"] = torch.exp(-scatter_mean(div_l1, belonging, dim=0))
        elif metric in ("nec", "nec++"):
            ret["KL"] = 1 - torch.exp(-scatter_mean(div_kl, belonging, dim=0)) # on paper
            ret["L1"] = 1 - torch.exp(-scatter_mean(div_l1, belonging, dim=0))
        aggr_std = scatter_std(div_l1, belonging, dim=0)
    elif "fid" in metric and preds_eval.shape[0] > 0:
        if preds_ori_ori.shape[1] == 1:
            l1 = torch.abs(preds_eval.reshape(-1) - preds_ori.reshape(-1))
        else:
            l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
        # results[ratio] = l1.numpy().tolist()
        ret["L1"] = scatter_mean(l1, belonging, dim=0)
        aggr_std = scatter_std(l1, belonging, dim=0)                    
    else:
        raise ValueError(metric)
    return ret, aggr_std

def normalize_belonging(belonging):
    #TODO: make more efficient
    ret = []
    i = -1
    for j , elem in enumerate(belonging):
        if len(ret) > 0 and elem == belonging[j-1]:
            ret.append(i)
        else:
            i += 1
            ret.append(i)
    return ret



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
