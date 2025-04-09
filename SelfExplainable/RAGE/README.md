
# Reconsidering Faithfulness in Regular, Self-Explainable and Domain Invariant GNNs

In this folder you can finde the source code to run RAGE. The code is adapted from the official code of the paper: "Robust Ante-hoc Graph Explainer using Bilevel OptimizationRobust Ante-hoc Graph Explainer using Bilevel Optimization".


## Training

To train the original model
```python
python train.py --dataset [dataset name]
```
To train the model with mitigation strategy Explanation Readout (ER)
```python
python train.py --dataset [dataset name] --gnn_pool m2
```

To train the model with mitigation strategy Hard Masking (HM)
```python
python train.py --dataset [dataset name] --mitigation HM
```

To train the model with both mitigation strategies (ER + HM)
```python
python train.py --dataset [dataset name] --mitigation m2HM
```

## Faithfulness evaluation


To evaluate the faithfulness of the model
```python
python eval_expl.py --dataset [dataset name] --mitigation [mitigation] 
```

