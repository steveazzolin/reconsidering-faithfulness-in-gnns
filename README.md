# Reconsidering Faithfulness in Regular, Self-Explainable and Domain Invariant GNNs

This is the official code for the ICLR25 paper [**Reconsidering Faithfulness in Regular, Self-Explainable and Domain Invariant GNNs**](https://openreview.net/pdf?id=kiOxNsrpQy).

## Installation

Using python=3.8.17, run the following commands:

```shell
# create venv environment
python -m venv myvenv/reconsidering

source myvenv/reconsidering/bin/activate

# install dependencies (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install torch_geometric==2.4.0

# install present library
pip install -e .

# additional required packages
pip install imbalanced-learn
```

For further details, refer to the original codebase of [GOOD](https://github.com/divelab/GOOD?tab=readme-ov-file) and [LECI](https://github.com/divelab/LECI/tree/LECI-1.0.0).

## Details

To enable one of the four mitigations, please append the following arguments to the command line argument

```shell
--mitigation_expl_scores hard # for HS
--mitigation_readout weighted # for ER
--model_name ${MODEL}GIN      # for LA (no virtual nodes)
--mitigation_sampling raw     # for CF (CIGA only)
```

### Train a model

For more details on training a model, refer to the original codebase of [GOOD](https://github.com/divelab/GOOD?tab=readme-ov-file) and [LECI](https://github.com/divelab/LECI/tree/LECI-1.0.0).

```shell
bash train.sh
```

### Evaluate a model

```shell
bash eval.sh
```

### Compute faithfulness

```shell
bash faithfulness.sh
```

By default, `suff++` computes a number `expval_budget/2` of perturbations by replacing the complement with that of another samples, and the other `expval_budget/2` by random erasures of the complement.

Then, to compute `NEC_b` as described in the paper, use `--samplingtype deconfounded` and `--nec_number_samples prop_G_dataset`. 
Other choices for `nec_number_samples` can be `prop_R` and `always_K`, which repscetively choose the number of modification proportionally to the size of the subgraph, or a predetermined value. Each percentual budget is set by the parameter `nec_alpha_1`.

To resort to a bernoulli-like subsampling (as done in RFID), use `--samplingtype bernoulli`, and the *alpha* parameter is set as `nec_alpha_1`.

### Compute plausibility (degree of invariance of the explanation)

```shell
bash plausibility.sh
```

This is only for datasets annotated with ground truth, i.e., with `data.edge_gt`.




For the code regarding the experiments with SEGNNs, we provide in the *SelfExplainable* folder the related codebase with further details.


### Reference to paper

```bibtex
@inproceedings{azzolin2025reconsidering,
    title={Reconsidering Faithfulness in Regular, Self-Explainable and Domain Invariant {GNN}s},
    author={Steve Azzolin and Antonio Longa and Stefano Teso and Andrea Passerini},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=kiOxNsrpQy}
}
```
