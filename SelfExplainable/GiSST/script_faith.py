from models.gisst_gc import Gisst, GisstConfig
from dataset_loader_gc import load_dataset, add_random_split_to_dataset
import json
import random
import numpy as np
import torch

from ourutils.splitting import get_subragphs_ratio
from torch_geometric.utils import to_networkx

from ourutils.evaluate_metric import compute_metric_ratio
from collections import defaultdict
from ourutils.good_motif2 import GOODMotif2
import os
from sklearn.model_selection import train_test_split
import torch_geometric
from torch_geometric.loader import DataLoader

class DotAccessibleDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
            
def reset_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Default state is a training state
    torch.enable_grad()


def load_dataset_model(dataset,seed,TRAIN,debias,mitigation=None):

    folder = "res_and_models/"+dataset+"/"
    seed = seed * 97 + 13
    model_path = folder+"seed_"+str(seed)

    if dataset == "GOODMotif2" or dataset == "GOODMotif_size":
        f = open(folder + 'config_train.json')
        config_train = DotAccessibleDict(json.load(f))
        model_path += "_"+config_train.architecture
        print(model_path)

        
        dict_datasets ,_ = GOODMotif2.load("./datasets/",domain="basis",debias=debias)

        config_Class = GisstConfig(config_dict=config_train)
        
        gisst = Gisst(dict_datasets["val"], dataset, config = config_Class,dict_dataset=dict_datasets)

        if TRAIN:
            assert False
        else:
            if config_Class.mitigation == "p2":
                gisst.my_load(model_path+"_p2")
                print("loaded model with mitigation! P2")
            elif config_Class.mitigation == "HM":
                gisst.my_load(model_path+"_HM")
                print("loaded model with mitigation! HM")
            elif config_Class.mitigation == "p2HM":
                gisst.my_load(model_path+"_p2HM")
            else:
                gisst.my_load(model_path)
        
    else:
        f = open(folder + 'config_train.json')
        config_train = DotAccessibleDict(json.load(f))    
        model_path += "_"+config_train.architecture
        print(model_path)

        _dataset = load_dataset(dataset,debias=debias)
        reset_random_seed(seed)
        add_random_split_to_dataset(_dataset)
        config_Class = GisstConfig(config_dict=config_train)
        gisst = Gisst(_dataset, dataset, config = config_Class)    

        if TRAIN:
            gisst.train()
            if config_Class.mitigation == "p2":
                gisst.my_save(model_path+"_p2")
            elif config_Class.mitigation == "HM":
                gisst.my_save(model_path+"_HM")
            else:
                gisst.my_save(model_path)
        else:
            if config_Class.mitigation == "p2":
                gisst.my_load(model_path+"_p2")
                print("loaded model with mitigation! P2")
            elif config_Class.mitigation == "HM":
                gisst.my_load(model_path+"_HM")
                print("loaded model with mitigation! HM")
            else:
                gisst.my_load(model_path)
         
    return gisst

def get_tensorized_metric(scores, c):
    return torch.tensor([
        scores[i][c] for i in range(len(scores))
    ])
             
def gmean(a,b):
    return (a*b).sqrt()
def aritm(a,b):
    return (a+b) / 2
def armonic(a,b):
    return 2 * (a*b) / (a+b)
def gstd(a):
    return a.log().std().exp()

def print_metric(name, data, results_aggregated=None, key=None):
    avg = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    print(name, " = ", ", ".join([f"{avg[i]:.3f} +- {std[i]:.3f}" for i in range(len(avg))]))
    if not results_aggregated is None:
        assign_dict(
            results_aggregated,
            key,
            avg.tolist()
        )
        key[-1] += "_std" # add _std to the metric name
        assign_dict(
            results_aggregated,
            key,
            std.tolist()
        )
def assign_dict(data, keys, to_add):
    if len(keys) == 1:
        data[keys[0]] = to_add
        return
    elif keys[0] not in data.keys():
        data[keys[0]] = dict()
    assign_dict(data[keys[0]], keys[1:], to_add)

def get_indices_dataset(dataset,numsamples_budget, extract_all=False):
    if numsamples_budget == "all" or numsamples_budget >= len(dataset) or extract_all:
        idx = np.arange(len(dataset))        
    elif numsamples_budget < len(dataset):        
        idx, _ = train_test_split(
            np.arange(len(dataset)),
            train_size=min(numsamples_budget, len(dataset)), # / len(dataset)
            random_state=42,
            shuffle=True,
            stratify=dataset.y if torch_geometric.__version__ >= "2.4.0" else dataset.data.y
        )
    return idx



def main(dataset,mitigation="None"):
    model = "GINmod"
    TRAIN = False

    config = {"random_expl": False,
            "expval_budget": 8,
            "nec_number_samples": "prop_G_dataset",
            "nec_alpha_1" : 0.05,
            "samplingtype" : "deconfounded",
            "mask" : True,
            "dataset_metric" : "ACCURACY", 
            "fidelity_alpha_2" : 0.9,
            "feat_int_alpha" : 0.2,
            "device" : "cuda",
            "numsamples_budget": 800,
            "debias": True,
            "log_id": "",
            "save_metrics": False,
            "dataset_name": dataset,
            "dataset_domain":"no_domain",
            "model_name":model,
            "save_metrics": True,
            "debias": True
            }

    config = DotAccessibleDict(config)
    splits = ["test"]
    metrics = [ "suff++","nec"]
    ratios = [.3, .6, .9, 1.]

    metrics_score = {}
    metrics_score["id"] = {s: defaultdict(list) for s in splits + ["test", "test_R"]}


    for seed in [1,2,3,4,5]:

        gisst = load_dataset_model(dataset,seed,TRAIN,config.debias,mitigation=mitigation)

        for split in splits:
            for metric in metrics:
                minority_class = -2
                if split == "val":
                    _dataset = gisst.kwargs_val["loader"]
                elif split == "test":
                    _dataset = gisst.kwargs_test["loader"]
                else:
                    print("Wrong split")
                    assert False

                # setta il budget
                if len(_dataset.dataset) >= config.numsamples_budget:
                    idx = get_indices_dataset(_dataset.dataset, config.numsamples_budget, extract_all=False)
                    _dataset = DataLoader(_dataset.dataset[idx], batch_size=32, shuffle=False, num_workers=2)


                model = gisst.model
                avg_graph_size = np.mean([d.x.shape[0] for d in _dataset.dataset])
                gisst.model.eval()
                model.eval()

                edge_scores = []
                for i in _dataset.dataset:
                    feat_score, edge_score  = gisst.get_explanation(i) 
                    edge_scores.append(edge_score)
                    
                graphs = [_dataset.dataset[i] for i in range(len(_dataset.dataset))]
                for g in graphs:
                    g.ori_x = g.x.clone()

                graphs_nx = [to_networkx(graphs[i], node_attrs=["ori_x"]) for i in range(len(_dataset.dataset))]
                labels = _dataset.dataset[:len(_dataset.dataset)].y


                causal_subgraphs_r = dict()
                spu_subgraphs_r = dict()
                expl_accs_r = dict()
                causal_masks_r = dict()

                for ratio in ratios:    
                    causal_subgraphs, spu_subgraphs, expl_accs, causal_masks = get_subragphs_ratio(graphs,ratio,edge_scores)
                    causal_subgraphs_r[ratio] = causal_subgraphs
                    spu_subgraphs_r[ratio] = spu_subgraphs
                    expl_accs_r[ratio] = expl_accs
                    causal_masks_r[ratio] = causal_masks

                score, acc_int, results = compute_metric_ratio(
                    model,
                    config,
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
                    intervention_bank = None,
                    intervention_distrib = "model_dependent",
                    debug=False,
                )
                metrics_score["id"][split][metric].append(score)
                metrics_score["id"][split][metric + "_acc_int"].append(acc_int)
                

        if not os.path.exists(f"results_file/res"+mitigation+".json"):
            with open(f"results_file/res"+mitigation+".json", 'w') as file:
                file.write("{}")
        with open(f"results_file/res"+mitigation+".json", "r") as jsonFile: #legge il file
            results_aggregated = json.load(jsonFile)

        for load_split in ["id"]:
            print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
            for split in splits:
                print(f"\nEval split {split}")
                for metric in metrics:
                    if "acc" == metric:
                        continue
                    for div in ["L1", "KL"]:
                        s = [
                            metrics_score[load_split][split][metric][i][f"all_{div}"] for i in range(len(metrics_score[load_split][split][metric]))
                        ]
                        print_metric(metric + f" class all_{div}", s, results_aggregated, key=[config.dataset_name + " " + config.dataset_domain, config.model_name, split, metric+f"_{div}"])
                    print_metric(metric + "_acc_int", metrics_score[load_split][split][metric + "_acc_int"], results_aggregated, key=[config.dataset_name+" "+config.dataset_domain, config.model_name, split, metric+"_acc_int"])
                    
            if "acc" in metrics:
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
                    if "suff++" in metrics and "nec" in metrics:
                        suff = get_tensorized_metric(metrics_score[load_split][split]["suff++"], f"all_{div}")
                        nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                        faith_aritm = aritm(suff, nec)
                        faith_armonic = armonic(suff, nec)
                        faith_gmean = gmean(suff, nec)
                        print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset_name + " " + config.dataset_domain, config.model_name, split, f"faith_aritm_{div}"])
                        print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset_name + " " + config.dataset_domain, config.model_name, split, f"faith_armon_{div}"])
                        print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset_name + " " + config.dataset_domain, config.model_name, split, f"faith_gmean_{div}"])
                    
            print(f"Computed for split load_split = {load_split}\n\n\n")

        if config.save_metrics:
            with open(f"results_file/res"+mitigation+".json", "w") as f:
                json.dump(results_aggregated, f)    

        results_aggregated    
        idx = np.argmax(results_aggregated[dataset+" no_domain"][config.model_name]["test"]["faith_armon_L1"])
        print(dataset, "\n\t %0.3f \t %0.3f" % (results_aggregated[dataset+" no_domain"][config.model_name]["test"]["faith_armon_L1"][idx], results_aggregated[dataset+" no_domain"][config.model_name]["test"]["faith_armon_L1_std"][idx]))


if __name__ == '__main__':
    mitigation = "p2HM"
    main("bbbp",mitigation)
    #main("mutag")
    #main("GOODMotif2")
    #main("GOODMotif_size")
    #main("bamultishapes",mitigation)