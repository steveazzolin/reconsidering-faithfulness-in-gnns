import json
import numpy as np
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import pearsonr

def armonic(a,b):
    return 2 * (a*b) / (a+b)

def pick_best_faith(data, where, faith_type):
    if len(data[where][faith_type]) == 1:
        return 0
    else:
        return np.argmax(data[where][faith_type][:-1])

plaus_type = "wiou"
file_name = "suff++_old"

with open(f"storage/metric_results/aggregated_id_results_{file_name}.json", "r") as jsonFile:
    data = json.load(jsonFile)

with open(f"storage/metric_results/acc_plaus.json", "r") as jsonFile:
    acc_plaus = json.load(jsonFile)

markers = {
    "GOODMotif basis": "o",
    "GOODMotif2 basis": "*",
    "GOODMotif size": "^",
    "GOODSST2 length": "v",
    "GOODTwitter length": "s",
    "GOODHIV scaffold": "D",
    "LBAPcore assay": "d",
    "GOODCMNIST color": "p"
}
colors = {
    "LECIGIN": "blue",
    "CIGAGIN": "orange", 
    "GSATGIN": "green",
    "LECIvGIN": "blue",
    "CIGAvGIN": "orange",
    "GSATvGIN": "green"
}

print(data.keys())

def lower_bound_plaus():
    splits = [("id_val", "test")]
    reference_metric = "likelihood" # "acc", "likelihood"
    num_cols = len(splits)
    num_rows = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5))

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for j, faith_type in enumerate(["faith_armon_L1"]):
        for i, (split_metric_id, split_metric_ood) in enumerate(splits):
            acc_coll, combined_coll = [], []
            for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size"]:
                for model in ["LECIGIN", "CIGAGIN", "GSATGIN", "LECIvGIN", "CIGAvGIN", "GSATvGIN"]:
                    if not model in data[dataset].keys():
                        continue
                    if not faith_type in data[dataset][model][split_metric_id].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], split_metric_id, faith_type)
                    faith_id   = np.array(data[dataset][model][split_metric_id][faith_type])[best_r]

                    best_r = pick_best_faith(data[dataset][model], split_metric_ood, faith_type)
                    faith_ood  = np.array(data[dataset][model][split_metric_ood][faith_type])[best_r]
                    combined = faith_id + faith_ood
                    
                    plaus_id       = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_id]["wiou"]))[-1]
                    plaus_ood      = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_ood]["wiou"]))[-1]
                    combined = combined + plaus_id + plaus_ood

                    if reference_metric == "acc":
                        acc_id    = acc_plaus[dataset][model][split_metric_id]["acc"][-1]
                        acc_ood   = acc_plaus[dataset][model][split_metric_ood]["acc"][-1]
                    elif reference_metric == "likelihood":
                        acc_id    = acc_plaus[dataset][model][split_metric_id]["likelihood_avg_entiresplit"]
                        acc_ood   = acc_plaus[dataset][model][split_metric_ood]["likelihood_avg_entiresplit"]
                    else:
                        raise ValueError(reference_metric)
                    
                    
                    acc = abs(acc_id - acc_ood)
                    if acc > 20:
                        continue
                    if isinstance(acc, float):
                        acc_coll.append(acc)
                    else:
                        acc_coll.extend(acc)
                    if isinstance(combined, float):
                        combined_coll.append(combined)
                    else:
                        combined_coll.extend(combined)
                    
                    axs[i%num_cols].scatter(combined, acc, marker=markers[dataset], label=model, c=colors[model], s=100)
                    axs[i%num_cols].grid(visible=True, alpha=0.5)
                    axs[i%num_cols].set_xlim(0.0, 4.)
                    axs[i%num_cols].set_ylim(-0.2, 1.)
                    axs[i%num_cols].set_ylabel(f"{reference_metric} difference", fontsize=12)
                    axs[i%num_cols].set_xlabel(f"faithfulness + domain invariance", fontsize=12)
                    axs[i%num_cols].set_title(f"")
            if len(acc_coll) > 0 and len(combined_coll) > 0:
                combined_coll, acc_coll = np.array(combined_coll), np.array(acc_coll)
                pcc = pearsonr(combined_coll, acc_coll)
                # axs[i%num_cols].annotate(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})", (0.8, 0.2), fontsize=7)
                print(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})")
                m, b = np.polyfit(combined_coll, acc_coll, 1)
                x = combined_coll.tolist() + [0, 4]
                axs[i%num_cols].plot(x, np.poly1d((m, b))(x), "r", alpha=0.5)

    legend_elements = []
    for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size"]:
        legend_elements.append(
            Line2D([0], [0], marker=markers[dataset], color='w', label=dataset, markerfacecolor='grey', markersize=15)
        )
    for model in ["LECIGIN", "CIGAGIN", "GSATGIN"]:
        legend_elements.append(
            Patch(facecolor=colors[model], label=model.replace("GIN", ""))
        )
    axs[-1].legend(handles=legend_elements, loc='upper right', fontsize=12) #, loc='center'

    plt.tight_layout()
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/paper_lower_bound.png")
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/pdfs/paper_lower_bound.pdf")
    plt.close()


def ablation_numsamples_budget_faith():
    num_cols = 3
    num_rows = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 5))
    budgets = [100, 500, 800, 1500, 2500] #100, 500, 800, 1500, 2500
    datasets = ["GOODCMNIST color"] #"GOODMotif2 basis", "GOODMotif size", "GOODCMNIST color"
    faith_type = "faith_armon_L1"

    faiths, faiths_std = defaultdict(list), defaultdict(list)
    for j, budget in enumerate(budgets):
        with open(f"storage/metric_results/aggregated_id_results_{file_name}_ablation_numsamples_budget_{budget}.json", "r") as jsonFile:
            data = json.load(jsonFile)        
        
        for i, split_metric in enumerate(["id_val", "val", "test"]):
            for dataset in datasets:
                for model in ["LECIGIN", "LECIvGIN"]:
                    if not model in data[dataset].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], split_metric, faith_type)
                    faith     = np.array(data[dataset][model][split_metric][faith_type])[best_r]
                    faith_std = np.array(data[dataset][model][split_metric][faith_type + "_std"])[best_r]
                    faiths[split_metric].append(faith)
                    faiths_std[split_metric].append(faith_std)
                    
    for i, split_metric in enumerate(["id_val", "val", "test"]):
        # axs[i%num_cols].plot(budgets, faiths[split_metric])
        axs[i%num_cols].errorbar(budgets, faiths[split_metric], yerr=faiths_std[split_metric], fmt='-o', capsize=5, label='Error')
        axs[i%num_cols].grid(visible=True, alpha=0.5)
        axs[i%num_cols].set_xticks(budgets)
        axs[i%num_cols].set_ylim(0.2, 0.8)
        axs[i%num_cols].set_title(f"{split_metric}")
        axs[i%num_cols].set_ylabel(f"faithfulness")
        axs[i%num_cols].set_xlabel(f"budget")

    plt.suptitle(f"Ablation num samples budget for {datasets[0]}")
    plt.tight_layout()
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/ablation_numsamples_budget_faith.png")
    plt.savefig(f"GOOD/kernel/pipelines/plots/illustrations/automatic/pdfs/ablation_numsamples_budget_faith_{datasets[0]}.pdf")
    plt.close()


def ablation_expval_budget_faith():
    num_cols = 3
    num_rows = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 5))
    budgets = [4, 8, 12, 16, 20]
    datasets = ["GOODMotif2 basis"] #"GOODMotif2 basis", "GOODMotif size", "GOODCMNIST color"
    faith_type = "faith_armon_L1"

    faiths, faiths_std = defaultdict(list), defaultdict(list)
    for j, budget in enumerate(budgets):
        with open(f"storage/metric_results/aggregated_id_results_{file_name}_ablation_expval_budget_{budget}.json", "r") as jsonFile:
            data = json.load(jsonFile)        
        
        for i, split_metric in enumerate(["id_val", "val", "test"]):
            for dataset in datasets:
                for model in ["LECIGIN", "LECIvGIN"]:
                    if not model in data[dataset].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], split_metric, faith_type)
                    faith     = np.array(data[dataset][model][split_metric][faith_type])[best_r]
                    faith_std = np.array(data[dataset][model][split_metric][faith_type + "_std"])[best_r]
                    faiths[split_metric].append(faith)
                    faiths_std[split_metric].append(faith_std)
                    
    for i, split_metric in enumerate(["id_val", "val", "test"]):
        # axs[i%num_cols].plot(budgets, faiths[split_metric])
        axs[i%num_cols].errorbar(budgets, faiths[split_metric], yerr=faiths_std[split_metric], fmt='-o', capsize=5, label='Error')
        axs[i%num_cols].grid(visible=True, alpha=0.5)
        axs[i%num_cols].set_xticks(budgets)
        axs[i%num_cols].set_ylim(0.2, 0.8)
        axs[i%num_cols].set_title(f"{split_metric}")
        axs[i%num_cols].set_ylabel(f"faithfulness")
        axs[i%num_cols].set_xlabel(f"budget")

    plt.suptitle(f"Ablation num samples budget for {datasets[0]}")
    plt.tight_layout()
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/ablation_expval_budget_faith.png")
    plt.savefig(f"GOOD/kernel/pipelines/plots/illustrations/automatic/pdfs/ablation_expval_budget_faith_{datasets[0]}.pdf")
    plt.close()


if __name__ == "__main__":
    lower_bound_plaus()    
    # ablation_numsamples_budget_faith()
    # ablation_expval_budget_faith()
