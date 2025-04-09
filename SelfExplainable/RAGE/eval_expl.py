import os
import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import random
import data_utils
from rage import RAGE,train,eval


from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch_geometric
from torch_geometric.utils import to_networkx
from ourutils.evaluate_metric import compute_metric_ratio,CustomDataset,evaluate_graphs
from ourutils.splitting import get_subragphs_ratio
from ourutils.xai_metric_utils import remove_from_graph
import json

@torch.no_grad()
def compute_accuracy_binarizing_Anon(
        givenR,
        model,
        metric_collector=None,
        dataset = None,
        graphs_nx = None,
        spu_subgraphs_r = None,
        expl_accs_r = None,
        labels = None,
        config = None,
    ):

        is_ratio = True
        weights = [0.3, 0.6, 0.9, 1.0]
        model.eval()

        if givenR:
            print("Accuracy computed as P(Y|R)")
        else:
            print("Accuracy computed as P(Y|G)")

        acc_scores, plaus_scores, wiou_scores = [], defaultdict(list), defaultdict(list)
        for weight in weights:
            eval_samples, labels_ori = [], []
            empty_graphs = 0

            #pbar = tqdm(range(len(dataset)), desc=f'Int. distrib', total=len(dataset))
            for i in range(len(dataset)):                
                G = graphs_nx[i].copy()
                G_filt = G
                if len(G.edges()) == 0:
                    empty_graphs += 1
                    continue
                if givenR: # for P(Y|R)
                    G_filt = remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[weight][i])                    
                    if len(G_filt) == 0:
                        empty_graphs += 1
                        continue

                eval_samples.append(G_filt)
                labels_ori.append(labels[i])

            # Compute accuracy
            labels_ori = torch.tensor(labels_ori)
            if len(eval_samples) == 0:
                acc = 0.
            else:
                eval_set = CustomDataset("", eval_samples, torch.arange(len(eval_samples)))
                loader = DataLoader(eval_set, batch_size=256, shuffle=False, num_workers=2)
                if config.mask and weight <= 1.:
                    preds, _ = evaluate_graphs(model=model,loader=loader,config=config, log=False, weight=None if givenR else weight, is_ratio=is_ratio)
                else:                    
                    preds, _ = evaluate_graphs(model=model,loader=loader,config=config, log=False)

                if preds.shape[1] == 1:
                    preds = preds.round().reshape(-1)
                else:
                    preds = preds.argmax(-1)     
                acc = (labels_ori == preds).sum() / (preds.shape[0] + empty_graphs)
            acc_scores.append(acc.item())   
            #print(f"Model Acc of binarized graphs for weight={weight} = {acc:.3f}")

            for c in labels_ori.unique():
                idx_class = np.arange(labels_ori.shape[0])[labels_ori == c]
                for q, (d, s) in enumerate(zip([wiou_scores, plaus_scores], ["WIoU", "F1"])):
                    d[c.item()].append(np.mean([e[q] for e in expl_accs_r[weight]]))
            for q, (d, s) in enumerate(zip([wiou_scores, plaus_scores], ["WIoU", "F1"])):
                d["all"].append(np.mean([e[q] for e in expl_accs_r[weight]]))
        print(acc_scores)
        metric_collector["acc"].append(acc_scores)
        #metric_collector["plaus"].append(plaus_scores)
        #metric_collector["wiou"].append(wiou_scores)
        return None

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Proteins')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate. Default is 0.0. ')
    parser.add_argument('--batch-size', type=int, default=256) # era 128
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GCN layers. Default is 3.')
    parser.add_argument('--dim', type=int, default=20, help='Number of GCN dimensions. Default is 20. ')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. Default is 100. ')
    parser.add_argument('--inner_epoch', type=int, default=10) # era 20
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default is 0.001. ')
    parser.add_argument('--device', type=int, default=0, help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--method', type=str, default='classification')
    parser.add_argument('--explainer_layer', type=str, default='gin') # it was gcn
    parser.add_argument('--gnn_layer', type=str, default='gin') # it was gcn
    parser.add_argument('--gnn_pool', type=str, default='mean') # it was max
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--start_run', type=int, default=1)
    parser.add_argument('--mitigation', type=str, default=None)


    return parser.parse_args()

class DotAccessibleDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
            
def get_config(dataset,device,model="RAGE"):

    config = {"random_expl": False,
            "expval_budget": 8,
            "nec_number_samples": "prop_G_dataset",
            "nec_alpha_1" : 0.05,
            "samplingtype" : "deconfounded",
            "mask" : True,
            "dataset_metric" : "ACCURACY", 
            "fidelity_alpha_2" : 0.9,
            "feat_int_alpha" : 0.2,
            "device" : device,
            "numsamples_budget": 800,
            "debias": True,
            "log_id": "",
            "save_metrics": False,
            "dataset_name": dataset,
            "dataset_domain":"no_domain",
            "model_name":model,
            "save_metrics": True,
            "debias": True,
            "acc_givenR": False
            }

    config = DotAccessibleDict(config)
    return config

def get_indices_dataset(dataset,y,numsamples_budget, extract_all=False):

    if numsamples_budget == "all" or numsamples_budget >= len(dataset) or extract_all:
        idx = np.arange(len(dataset))        
    elif numsamples_budget < len(dataset):
        idx, _ = train_test_split(
            np.arange(len(dataset)),
            train_size=min(numsamples_budget, len(dataset)), # / len(dataset)
            random_state=42,
            shuffle=True,
            stratify=y
        )
    return idx

@torch.no_grad()
def get_explanation(model, graph, device):
    model.eval()
    explanation = model(graph.to(device))[0]
    return explanation
    
def main():

    splits = ["test"] #["val","test"]
    metrics = [ "suff++","nec"]
    ratios = [.3, .6, .9, 1.]

    metrics_score = {}
    metrics_score["id"] = {s: defaultdict(list) for s in splits + ["test", "test_R"]}

    for seed in [1,2,3,4,5]:#[1,2,3,4,5]:
        args = parse_args()
        args.random_seed = seed * 97 + 13
        
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        # Load and split the dataset.
        if args.dataset == "GOODMotif2" or args.dataset == "GOODMotif_size":
            dataset_dict = data_utils.load_dataset(args.dataset)
            train_set = dataset_dict["train"]
            valid_set = dataset_dict["val"]
            test_set = dataset_dict["test"]
            dataset = train_set
            valid_y = valid_set.y
            test_y = test_set.y

        else:
            dataset = data_utils.load_dataset(args.dataset)
            #split mio
            idx_train, idx_test = train_test_split(np.arange(len(dataset)),test_size= 0.2,random_state=42)
            idx_test, idx_val = train_test_split(idx_test, test_size= 0.5,random_state=42)    
            train_set = dataset[idx_train]
            valid_set = dataset[idx_val]
            test_set = dataset[idx_test]
            train_y = dataset.data.y[idx_train]
            valid_y = dataset.data.y[idx_val]
            test_y = dataset.data.y[idx_test]
        # split originale
        # splits, indices = data_utils.split_data(dataset)
        # train_set, valid_set, test_set = splits
        args.num_features = dataset.num_features
        args.num_classes = train_set.num_classes

        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        device = "cpu"
        args.device = device


        train_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
        valid_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
        test_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
        
        run = args.random_seed

        args.run = run
        torch.backends.cudnn.deterministic = True
        random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        np.random.seed(run)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Initialize the model.
        result_folder = f'data/{args.dataset}/rage/'
        model = RAGE(args).to(device)
        if args.gnn_pool == "m2":
            print("sto caricando il mitigation")
            model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}_m2.pt', map_location=device))
        elif args.mitigation == "m2HM":
            model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}_m2HM.pt', map_location=device))
            print(result_folder + f'best_model_run_{run}_m2HM.pt')
        elif args.mitigation == "HM":
            model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}_HM.pt', map_location=device))
            print(result_folder + f'best_model_run_{run}_HM.pt')
        else:
            model.load_state_dict(torch.load(result_folder + f'best_model_run_{run}.pt', map_location=device))

        train_loss, train_auc, train_ap, train_acc, train_preds, train_grounds = eval(model, train_loader, device, method=args.method, args=args)
        valid_loss, valid_auc, valid_ap, valid_acc, valid_preds, valid_grounds = eval(model, valid_loader, device, method=args.method, args=args)
        test_loss, test_auc, test_ap, test_acc, test_preds, test_grounds = eval(model, test_loader, device, method=args.method, args=args)
        print("train_acc: %.2f val_acc: %.2f test_acc: %.2f " %(train_acc,valid_acc,test_acc))
        # fine caricamento modello



        config = get_config(args.dataset,args.device)

        for metric in metrics:
            for split in splits:
                if split == "val":
                    _dataset = DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=0)
                    curr_data = valid_set
                    curr_y = valid_y
                elif split == "test": 
                    _dataset = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
                    curr_data = test_set
                    curr_y = test_y

                else:
                    print("Wrong split")
                    assert False

                # setta il budget
                if len(_dataset.dataset) >= config.numsamples_budget:
                    idx = get_indices_dataset(dataset_dict[split],dataset_dict[split].y, config.numsamples_budget, extract_all=False)
                    _dataset = DataLoader(dataset_dict[split][idx], batch_size=1, shuffle=False, num_workers=2)


                model = model
                avg_graph_size = np.mean([d.x.shape[0] for d in _dataset.dataset])
                model.eval()
                print(avg_graph_size)


                edge_scores = []
                for i in _dataset:
                    edge_score  = model.get_explanation(i.to(device))
                    edge_scores.append(edge_score.to(device))
                        
                graphs = [_dataset.dataset[i] for i in range(len(_dataset.dataset))]
                for g in graphs:
                    g.ori_x = g.x.clone()


                graphs_nx = [to_networkx(graphs[i].to(device), node_attrs=["ori_x"]) for i in range(len(_dataset.dataset))]
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
            
    if args.gnn_pool == "m2":
        mit = "m2"
    elif args.mitigation == "HM":
        mit = "HM"
    else:
        mit = ""

    if not os.path.exists(f"results/res"+mit+".json"):
        with open(f"results/res"+mit+".json", 'w') as file:
            file.write("{}")
    with open(f"results/res"+mit+".json", "r") as jsonFile: #legge il file
        results_aggregated = json.load(jsonFile)
    
    print(results_aggregated)

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
        with open(f"results_Anon/res"+mit+".json", "w") as f:
            json.dump(results_aggregated, f,indent=4)    

    idx = np.argmax(results_aggregated[args.dataset+" no_domain"]["RAGE"]["test"]["faith_armon_L1"])
    print(args.dataset, "\n\t %0.3f \t %0.3f" % (results_aggregated[args.dataset+" no_domain"]["RAGE"]["test"]["faith_armon_L1"][idx], results_aggregated[args.dataset+" no_domain"]["RAGE"]["test"]["faith_armon_L1_std"][idx]))



if __name__ == '__main__':
    main()
