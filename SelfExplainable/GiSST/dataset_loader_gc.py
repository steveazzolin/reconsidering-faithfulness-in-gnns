from torch_geometric.datasets import MoleculeNet
import global_config as gl
import torch_geometric.transforms as T
import torch
import os.path
import networkx as nx
import pickle
from torch_geometric.seed import seed_everything
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx, to_networkx
import numpy as np

from torch_geometric.datasets import ExplainerDataset, TUDataset, BAMultiShapesDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif
from utils.BAMultiShapesGeneration import generate_data_list
from torch_geometric.utils import from_networkx, shuffle_node


def load_dataset(dataset_name,debias=False):
    """_summary_
    !!!This function seeds with a global seed, use this function before seeding and seed afterwards!!!
    Solution differs from NC, this is beacuse all datasets needed a random data split.
    The python frameworks do not easily allow for not seeding globally so this solution was used
    If different seeds are required they should be set after this function has been used
    
    Datasets to load include:
        bbbp
        mutag
        ba2motif
        bamultishapes
    
    Other datasets are also included but were not used in the final thesis and might not work correctly:
        reddit-binary
        enzymes
        bamultishapesorg    
    Args:
        dataset_name (_type_): The name of the dataset to load

    Returns:
        dataset (pytorch_geometric.data.dataset): The dataset that was loaded
    """
    seed_everything(121231541)
    
    if dataset_name.lower() in ['bbbp']:
        dataset = MoleculeNet(root = gl.GlobalConfig.dataset_dir, name = dataset_name.lower())
        ndataset = fixMoleculeNet(dataset, dataset_name.lower())
        add_random_split_to_dataset(ndataset)
        return ndataset
    
    if dataset_name.lower() in ['mutag', "reddit-binary", "enzymes"]:
        dataset = TUDataset(root = gl.GlobalConfig.dataset_dir, name=dataset_name.upper())
        add_random_split_to_dataset(dataset)
        return dataset    
        
    if dataset_name.lower() in ['ba2motif']:
        dataset = gen_BA2()
        print("in")
        add_random_split_to_dataset(dataset)
        return dataset
    
    if dataset_name.lower() in ['bamultishapes']:
        data_list = generate_data_list(1000)
        dataset = SynExplain(gl.GlobalConfig.dataset_dir, data_list, name = "bamultishapes.pt",debias=debias)
        add_random_split_to_dataset(dataset)
        return dataset
    
    if dataset_name.lower() in ['bamultishapesorg']:
        data_list = generate_data_list(1000)
        dataset = BAMultiShapesDataset(gl.GlobalConfig.dataset_dir)
        add_random_split_to_dataset(dataset)
        return dataset


def add_random_split_to_dataset(dataset):
    index = [i for i in range(len(dataset.y))]
    
    index_train, index_val, index_test, \
        y_train, y_val, y_test = split_index_train_val_test(index, dataset.y)
        
    dataset.index_train = torch.LongTensor(index_train)
    dataset.index_val = torch.LongTensor(index_val)
    dataset.index_test = torch.LongTensor(index_test)
    dataset.y_train = torch.LongTensor(y_train)
    dataset.y_val = torch.LongTensor(y_val)
    dataset.y_test = torch.LongTensor(y_test)

def split_index_train_val_test(
    index,
    y,
    train_p=0.8, 
    seed=None
):
    """
    Split a list of indices into training, validation, and test set.
    Test and validation set will have the same size

    Args:
        index (list): Indices to split.
        y (list): Corresponding labels for stratification.
        train_p (float): Proportion of training data.
        seed (int): random seed for splitting the data.

    Return:
        index_train, index_val, index_test (list): Indices in each set.
        y_train, y_val, y_test (list): Labels in each set.
    """
    index_train, index_val_test, y_train, y_val_test = train_test_split(
        index, 
        y, 
        train_size=train_p, 
        random_state=seed, 
        stratify=y
    )
    index_val, index_test, y_val, y_test = train_test_split(
        index_val_test, 
        y_val_test, 
        test_size=0.5, 
        random_state=seed, 
        stratify=y_val_test
    )
    return index_train, index_val, index_test, \
        y_train, y_val, y_test
        
def gen_BA2():
    dataset1 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=25, num_edges=1),
    motif_generator=HouseMotif(),
    num_motifs=1,
    num_graphs=500,)

    dataset2 = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=25, num_edges=1),
    motif_generator=CycleMotif(5),
    num_motifs=1,
    num_graphs=500,)
    
    trans = T.Compose([T.Constant(0.1)]*10)
    
    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    
    data = []
    for e in dataset:
        ndata = trans(e)
        G = to_networkx(dataset[0])
        G = G.to_undirected()

        for i in G.nodes(): #Add features that count node degree and number of triangles a node participates in
            G.nodes[i]['x']=np.asarray([nx.degree(G,i),nx.triangles(G,i)],dtype=np.float32)

        ndata.x = from_networkx(G).x
             
        data.append(ndata)
    
    for i in range(0, 500):
        data[i].y = torch.zeros(1, dtype=int)
    for i in range(500, 1000):
        data[i].y = torch.ones(1, dtype=int)
    

    return SynExplain(gl.GlobalConfig.dataset_dir, data, name = "ExplainBA2Motif.pt")


def fixMoleculeNet(dataset, name):
    #Fix float labels and make them int...
    datalist = []
    for e in dataset:
        ndata = e.clone()
        ndata.x = ndata.x.float()
        ndata.y = ndata.y[0,0].long() #Default has a wierd list and is float TODO Check if this is also the case for any other moleculenet dataset
        if ndata.x.size()[0] > 0:
            datalist.append(ndata) 
        
    newDataset = MoleculeFix(gl.GlobalConfig.dataset_dir, datalist, name)
    return newDataset

class SynExplain(InMemoryDataset):
    def __init__(self, root, data_list, name, transform=None,debias=False):
        self.data_list = data_list
        self.name = name
        super().__init__(root, transform)
        
        #Somewhat hacky way of doing processing. Normally it would be done in the parent constructor
        #Issue is you can't force it to update the dataset if changes are made to the dataset and a cached version is on disk
        
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

        if debias:
            print(f"#D#Permuting node indices to remove explanation bias for")
            sa = []
            for i in range(self.len()):
                data = self.get(i)
                data.x, perm = shuffle_node(data.x, data.batch)
                dict_perm = {p.item(): j for j, p in enumerate(perm)}
                data.ori_edge_index = data.edge_index.clone()
                data.edge_index = torch.tensor([ [dict_perm[x.item()], dict_perm[y.item()]] for x,y in data.edge_index.T ]).T
                data.node_perm = perm
                sa.append(data)

            self.data, self.slices = self.collate(sa)

    @property
    def processed_file_names(self):
        return self.name 

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])

class MoleculeFix(InMemoryDataset):
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
    
            
    