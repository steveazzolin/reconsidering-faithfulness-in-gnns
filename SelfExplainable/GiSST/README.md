
# Reconsidering Faithfulness in Regular, Self-Explainable and Domain Invariant GNNs

In this folder you can find the source code to run GISST. The code is adapted from the official code of the paper: How Faithful are Self-Explainable GNNs?".


## Training


To train the model 
```python
python train.py --dataset [dataset name] --mode [train/test] --seed [1/2/3/4/4] --dataset [dataset_name]
```

If you want to train the model with a mitigation, change the config file located in
```python
res_and_models/dataset_name/config_train.json
```
and modify the mitigation strategy:

* "mitigation":"None" for the original model
* "mitigation":"p2" for the Explanation readout
* "mitigation":"HM" for the Hard Masking
* "mitigation":"p2HM" for  both of them

## Faithfulness evaluation


To evaluate the faithfulness of the model
```python
python script_faith.py
```
in the main of the file specify the mitigation and the dataset.