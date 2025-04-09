r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import sys
import os
import time
from typing import Tuple, Union
import json
from collections import defaultdict
from datetime import datetime

import torch.nn
from torch.utils.data import DataLoader

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.logger import load_logger
from GOOD.utils.metric import assign_dict
from GOOD.definitions import OOM_CODE

import numpy as np
import matplotlib.pyplot as plt

torch.set_num_threads(6)

def initialize_model_dataset(config: Union[CommonArgs, Munch]) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])
    print(dataset["id_val"].get(0))

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader


def generate_plot_sampling(args):
    load_splits = ["id"]
    splits = ["test"]
    seeds = args.seeds.split("/")
    ratios = [0.3, 0.6, 0.8, 0.9, 1.0]
    sampling_alphas = [0.03, 0.05]
    all_metrics, all_accs = {}, {}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(seeds):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)

            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx="suff" in args.metrics
            )
            
            metrics, accs = pipeline.generate_plot_sampling_type(splits, ratios, sampling_alphas, graphs, graphs_nx, causal_subgraphs_r, causal_masks_r, avg_graph_size)
            all_metrics[str(seed)] = metrics
            all_accs[str(seed)] = accs
        
        for SPLIT in splits:
            num_cols = len(sampling_alphas)
            fig, axs = plt.subplots(1, num_cols, figsize=(2.9*num_cols, 3.9), sharey=True)
            colors = {
                "NEC KL": "blue", "NEC L1":"lightblue", "FID L1 div": "green", "Model FID": "orange", "Phen. FID": "red", "Change pred": "violet"
            }
            sampling_name = {"RFID_": "RFID+ ($)", "FIXED_": "Fixed Deconfounded ($)", "DECONF_": "NEC ($)", "DECONF_R_": "NEC ($)"}
            for j, sampling_type_ori in enumerate(["RFID_", "DECONF_", "DECONF_R_"]): #"FIXED_", 
                for alpha_i, alpha in enumerate(sampling_alphas):
                    param = str(alpha_i+1 if sampling_type_ori == "FIXED_" else alpha)
                    sampling_type = sampling_type_ori + param
                    anneal, anneal_std = [], []
                    for r in ratios:
                        for i, metric_name in enumerate(["NEC L1"]):
                            anneal.append(np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]))
                            anneal_std.append(np.std([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]))
                    
                    if "RFID" in sampling_type:
                        l = f"$\kappa=${param}"
                    elif "DECONF_R_" in sampling_type:
                        l = f"$b=${param}||R||"
                    elif "DECONF_" in sampling_type:
                        l = f"$b=${param}" + "$\\bar{m}$"

                    axs[alpha_i%num_cols].errorbar(
                        ratios,
                        anneal,
                        yerr=anneal_std,
                        fmt='-o',
                        capsize=5,
                        label=sampling_name[sampling_type_ori].replace('$', l))
                    axs[alpha_i%num_cols].grid(visible=True, alpha=0.5)
                    axs[alpha_i%num_cols].set_ylim((0., 1.1))
                    axs[alpha_i%num_cols].legend(loc='best', fontsize=11)
            fig.supxlabel('size ratio', fontsize=13)
            fig.supylabel('value', fontsize=13)
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./GOOD/kernel/pipelines/plots/metrics/R_dev_nec_sampling_{config.ood.ood_alg}_{config.dataset.dataset_name}_{config.dataset.domain}_({SPLIT}).png")
            plt.show()


def evaluate_metric(args):
    load_splits = ["id"]

    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_test", "test"]
    print("Using splits = ", splits)
        
    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    print("Using ratios = ", ratios)
    startTime = datetime.now()

    metrics_score = {}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")

        metrics_score[load_split] = {s: defaultdict(list) for s in splits + ["test", "test_R"]}
        for i, seed in enumerate(args.seeds.split("/")):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)
            
            if "CIGA" in config.model.model_name:
                ratios = [pipeline.model.att_net.ratio]

            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics)
            )

            for metric in args.metrics.split("/"):
                print(f"\n\nEvaluating {metric.upper()} for seed {seed} with load_split {load_split}\n")

                if metric == "plaus":
                    for split in splits:
                        metrics_score[load_split][split]["wiou"].append([np.mean([e[0] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["wiou_std"].append([np.std([e[0] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["F1"].append([np.mean([e[1] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["F1_std"].append(np.std([e[1] for e in expl_accs_r[split][1.0]]))
                    continue

                for split in splits:
                    score = pipeline.compute_metric_ratio(
                        ratios,
                        split,
                        metric=metric,
                        intervention_distrib=config.intervention_distrib,
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
                    metrics_score[load_split][split][metric].append(score)

    if config.save_metrics:
        if not os.path.exists(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json"):
            with open(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json", 'w') as file:
                file.write("{}")
        with open(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json", "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    else:
        results_aggregated = None

    for load_split in load_splits:
        print("\n\n", "-"*50, f"\nPrinting evaluation results for load_split {load_split}\n\n")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                print(f"{metric} = {metrics_score[load_split][split][metric]}")

        if "plaus" in args.metrics:
            print("\n\n", "-"*50, "\nComputing Plausibility")
            for split in splits:
                print(f"\nEval split {split}")
                for div in ["wiou", "F1"]:
                    s = metrics_score[load_split][split][div]
                    print_metric(div, s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, div])
            continue

        print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                if "acc" == metric:
                    continue
                for div in ["L1"]:
                    s = [
                        metrics_score[load_split][split][metric][i][f"all_{div}"] for i in range(len(metrics_score[load_split][split][metric]))
                    ]
                    print_metric(metric + f" class all_{div}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, metric+f"_{div}"])

        print("\n\n", "-"*50, "\nComputing faithfulness")
        for split in splits:
            print(f"\nEval split {split}")            
            for div in ["L1"]:
                if "suff++" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff++"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_gmean_{div}"])
        print(f"Computed for split load_split = {load_split}\n\n\n")
    
    if config.save_metrics:
        with open(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json", "w") as f:
            json.dump(results_aggregated, f)     
    
    print("Completed in ", datetime.now() - startTime, f" for {config.model.model_name} {config.dataset.dataset_name}/{config.dataset.domain}")
    print("\n\n")
    sys.stdout.flush()
                    
def gmean(a,b):
    return (a*b).sqrt()

def aritm(a,b):
    return (a+b) / 2

def armonic(a,b):
    return 2 * (a*b) / (a+b)

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

def get_tensorized_metric(scores, c):
    return torch.tensor([
        scores[i][c] for i in range(len(scores))
    ])

def main():
    args = args_parser()

    assert not args.seeds is None, args.seeds

    if args.task == 'eval_metric':
        evaluate_metric(args)
        exit(0)
    if args.task == 'plot_sampling':
        generate_plot_sampling(args)
        exit(0)        

    test_scores, test_losses = defaultdict(list), defaultdict(list)
    test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, seed in enumerate(args.seeds.split("/")):
        seed = int(seed)
        print(f"\n\n#D#Running with seed = {seed}")
        
        args.random_seed = seed
        args.exp_round = seed
        
        config = config_summoner(args)
        config["mitigation_sampling"] = args.mitigation_sampling
        print(config.random_seed, config.exp_round)
        print(args)
        if i == 0:
            load_logger(config)
        
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)

        if config.task == 'train':
            pipeline.load_task() # train model
            pipeline.task = 'test'
            test_score, test_loss = pipeline.load_task()
            test_scores["trained"].append(test_score)
        elif config.task == 'test':
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")
            for s in ["train", "id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(s, compute_suff=False)
                test_scores[s].append(sa['score'])
                test_losses[s].append(sa['loss'].item())
                test_likelihoods_avg[s].append(sa['likelihood_avg'].item())
                test_likelihoods_prod[s].append(sa['likelihood_prod'].item())
                test_likelihoods_logprod[s].append(sa['likelihood_logprod'].item())
    
    if config.save_metrics:
        with open(f"storage/metric_results/acc_plaus.json", "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    
    print("\n\nFinal accuracies: ")
    for s in test_scores.keys():
        print(f"{s.upper():<10} = {np.mean(test_scores[s]):.3f} +- {np.std(test_scores[s]):.3f}")
    
    print("\nFinal losses: ")
    for s in test_losses.keys():
        print(f"{s.upper():<10} = {np.mean(test_losses[s]):.4f} +- {np.std(test_losses[s]):.4f}")
            
    for s in [""]:
        print(f"Diff id_val-test {s} = {abs(np.mean(test_losses[s + 'id_val']) - np.mean(test_losses[s + 'test'])):.4f} ")

    if config.save_metrics:
        print("Saving metrics to json...")
        for s in test_losses.keys():
            for name, d in zip(
                ["loss_entiresplit", "likelihood_avg_entiresplit", "likelihood_prod_entiresplit", "likelihood_logprod_entiresplit"], 
                [test_losses, test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod]
            ):
                key = [config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, s, name]
                if s in results_aggregated[key[0]][key[1]].keys():            
                    assign_dict(
                        results_aggregated,
                        key,
                        np.mean(d[s])
                    )
                    key[-1] += "_std"
                    assign_dict(
                        results_aggregated,
                        key,
                        np.std(d[s])
                    )
        with open(f"storage/metric_results/acc_plaus.json", "w") as f:
            json.dump(results_aggregated, f)

def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e

if __name__ == '__main__':
    main()
