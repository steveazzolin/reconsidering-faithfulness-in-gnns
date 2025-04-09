from models.gisst_gc import Gisst, GisstConfig
from dataset_loader_gc import load_dataset

from global_config import GlobalConfig as gl
import random
import numpy as np
import torch
import argparse

import json

from ourutils.good_motif2 import GOODMotif2
from ourutils.good_motif import GOODMotif

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



def main(seed):
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some parameters.')

    # Add arguments
    parser.add_argument('--mode', type=str, required=True, help='Train flag: True or False')
    parser.add_argument('--seed', type=int, required=True, help='Seed value for random number generation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    dataset = args.dataset


    folder = "res_and_models/"+dataset+"/"

    seed = seed * 97 + 13
    model_path = folder+"seed_"+str(seed)
    if args.mode == "train":
        TRAIN = True
    else:
        TRAIN = False
        
    if dataset == "GOODMotif2":
        f = open(folder + 'config_train.json')
        config_train = DotAccessibleDict(json.load(f))
        model_path += "_"+config_train.architecture
        print(model_path)

        dict_datasets ,meta_info = GOODMotif2.load("./datasets/",domain="basis",generate=True)
        print(meta_info)
        config_Class = GisstConfig(config_dict=config_train)

        print("dict_dataset",dict_datasets)
        gisst = Gisst(dict_datasets["train"], dataset, config = config_Class,dict_dataset=dict_datasets)
        print(config_Class.mitigation)
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
            elif config_Class.mitigation == "HM":
                gisst.my_load(model_path+"_HM")
            else:
                gisst.my_load(model_path)
            res = gisst.test()
            print(res)
            return res
            
    elif dataset == "GOODMotif_size":
        f = open(folder + 'config_train.json')
        config_train = DotAccessibleDict(json.load(f))
        model_path += "_"+config_train.architecture
        print(model_path)

        dict_datasets ,meta_info = GOODMotif.load("./datasets/",domain="size",generate=True,debias=False)
        print(meta_info)
        print("dict_dataset",dict_datasets)
        config_Class = GisstConfig(config_dict=config_train)        
        _dataset = dict_datasets["train"]


        # used for debug
        #dict_datasets["train"] = dict_datasets["train"][0:200].copy()
        #dict_datasets["val"] = dict_datasets["val"][0:100].copy()
        #dict_datasets["test"] = dict_datasets["test"][0:100].copy()
        
        gisst = Gisst(dict_datasets["train"], dataset, config = config_Class,dict_dataset=dict_datasets)

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
            elif config_Class.mitigation == "HM":
                gisst.my_load(model_path+"_HM")
            else:
                gisst.my_load(model_path)
            res = gisst.test()
            print(res)
            return res
    
    else:
        f = open(folder + 'config_train.json')
        config_train = DotAccessibleDict(json.load(f))
        model_path += "_"+config_train.architecture
        print(model_path)


        _dataset = load_dataset(dataset,debias=False)
        reset_random_seed(seed)
        config_Class = GisstConfig(config_dict=config_train)
        gisst = Gisst(_dataset, dataset, config = config_Class)    


        if TRAIN:
            gisst.train()
            if config_Class.mitigation == "p2":
                gisst.my_save(model_path+"_p2")
            elif config_Class.mitigation == "HM":
                gisst.my_save(model_path+"_HM")
            elif config_Class.mitigation == "p2HM":
                gisst.my_save(model_path+"_p2HM")
            else:
                gisst.my_save(model_path)
        else:
            if config_Class.mitigation == "p2":
                gisst.my_load(model_path+"_p2")
            elif config_Class.mitigation == "HM":
                gisst.my_load(model_path+"_HM")
            elif config_Class.mitigation == "p2HM":
                gisst.my_load(model_path+"_p2HM")
            else:
                gisst.my_load(model_path)
            res = gisst.test()
            print(res)
            return res
if __name__ == '__main__':

    a = []
    for i in [1,2,3,4,5]:
        a.append(main(i))
    print(a)
    print(np.mean(a),np.std(a))
    