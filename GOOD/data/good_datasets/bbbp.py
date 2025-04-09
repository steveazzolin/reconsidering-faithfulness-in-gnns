"""
The GOOD-Motif dataset motivated by `Spurious-Motif
<https://arxiv.org/abs/2201.12872>`_.
"""
import math
import os
import os.path as osp
import random

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.datasets import BAMultiShapesDataset, MoleculeNet
from torch_geometric.utils import from_networkx, shuffle_node
from torch_geometric.data.separate import separate

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *
from GOOD.utils.synthetic_data import synthetic_structsim


@register.dataset_register
class BBBP(InMemoryDataset):
    r"""
    The BBBP split of MoleculeNet dataset from PyG
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, debias=False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.minority_class = None
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = ''

        assert False

    @property
    def raw_dir(self):
        return osp.join(self.root)

    @staticmethod
    def fixMoleculeNet(dataset, name, dataset_root):
        r"""
        From `<https://arxiv.org/pdf/2308.15096.pdf>`_.
        """
        #Fix float labels and make them int...
        #Remove empty graphs?
        datalist = []
        for e in dataset:
            ndata = e.clone()
            ndata.x = ndata.x #.float()
            ndata.y = ndata.y[0,0].long()
            if ndata.x.size()[0] > 0:
                datalist.append(ndata)             
        newDataset = MoleculeFix(dataset_root, datalist, name)
        return newDataset

    @staticmethod
    def load(dataset_root: str, domain: str, shift: str = 'no_shift', generate: bool = False, debias: bool =False):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        assert domain == "basis" and shift == "no_shift", f"{domain} - {shift} not supported"
        meta_info = Munch()
        meta_info.dataset_type = 'mol'
        meta_info.model_level = 'graph'

        dataset = MoleculeNet(dataset_root, name="bbbp")
        dataset = BBBP.fixMoleculeNet(dataset, "bbbp", dataset_root)
        # dataset._data.edge_attr = None # remove edge attributes for fair comparison with other baselines

        

        index_train, index_val_test = train_test_split(
            torch.arange(len(dataset)), 
            train_size=0.8,
            stratify=dataset.y
        )
        index_val, index_test = train_test_split(
            torch.arange(len(dataset[index_val_test])), 
            train_size=0.5,
            stratify=dataset[index_val_test].y
        )

        train_dataset = dataset[index_train]
        id_val_dataset = dataset[index_val]
        id_test_dataset = dataset[index_test]

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features

        meta_info.edge_feat_dims = dataset._data.edge_attr.max(0).values - dataset._data.edge_attr.min(0).values + 2

        meta_info.num_envs = 1
        meta_info.num_classes = 2

        train_dataset.minority_class = None
        id_val_dataset.minority_class = None
        id_test_dataset.minority_class = None
        train_dataset.metric = 'Accuracy'
        id_val_dataset.metric = 'Accuracy'
        id_test_dataset.metric = 'Accuracy'

        # --- clear buffer dataset._data_list ---        
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'metric': 'Accuracy', 'task': 'Multi-label classification',
                'val': id_val_dataset, 'test': id_test_dataset}, meta_info
                


class MoleculeFix(InMemoryDataset):
    r"""
    From `<https://arxiv.org/pdf/2308.15096.pdf>`_.
    """
    def __init__(self, root, data_list, name, transform=None):
        self.name = name
        self.data_list = data_list
        super().__init__(root, transform)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return self.name + "Fixed"

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])
    