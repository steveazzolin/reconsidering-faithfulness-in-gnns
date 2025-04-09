from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
from tqdm import tqdm
import numpy as np

from scipy.sparse import coo_matrix
from torch_geometric.datasets import TUDataset,BAMultiShapesDataset
from torch_geometric.utils import degree

from torch.utils.data import random_split
import torch
import torch.nn.functional as F

from torch_geometric.utils import dense_to_sparse
from torch_geometric.datasets import MoleculeNet
from ourutils.good_motif2 import GOODMotif2
from ourutils.good_motif import GOODMotif

class CMU_FACE_M(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.name = 'CMU-FACE_M'
        self.cleaned = False
        self.graph_file = 'cmu_face_m.pt'
        self.graph_count = 624
        super(CMU_FACE_M, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return [self.graph_file]

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return 2

    def download(self):
        pass

    def process(self):
        dataset = torch.load(os.path.join(self.raw_dir, self.graph_file))

        for i, d in enumerate(dataset):
            d = Data(edge_index=d.edge_index.clone(),
                     x=d.x.float().clone(),
                     y=d.y.clone())
            torch.save(d, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


class Planted_Clique(InMemoryDataset):
    def __init__(self, root, data_size=100, node_size=100, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.name = 'Planted-Clique'
        self.cleaned = False

        self.data_size = data_size
        self.node_size = node_size

        self.graphs_path = f"graphs_{data_size}_{node_size}.pt"
        self.features_path = f"features_{data_size}_{node_size}.pt"
        self.labels_path = f"labels_{data_size}_{node_size}.pt"

        super(Planted_Clique, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return [f"graphs_{self.data_size}_{self.node_size}.pt", f"features_{self.data_size}_{self.node_size}.pt", f"labels_{self.data_size}_{self.node_size}.pt"]

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{self.data_size}_{self.node_size}_{i}.pt' for i in range(self.data_size)]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    def download(self):
        pass

    @property
    def num_classes(self) -> int:
        return 2

    def process(self):
        graphs = torch.load(os.path.join(self.raw_dir, self.graphs_path))
        features = torch.load(os.path.join(self.raw_dir, self.features_path))
        labels = torch.load(os.path.join(self.raw_dir, self.labels_path))

        for i in tqdm(range(len(graphs))):
            edge_index, _ = dense_to_sparse(graphs[i])
            data = Data(edge_index=edge_index.clone(),
                        x=features[i].float().clone(),
                        y=torch.LongTensor([labels[i]]).clone()
                        )

            torch.save(data, os.path.join(self.processed_dir, f'data_{self.data_size}_{self.node_size}_{i}.pt'))

    def len(self):
        return self.data_size

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{self.data_size}_{self.node_size}_{idx}.pt'))
        return data


class Tree_of_Life(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.name = 'Tree-of-Life'
        self.cleaned = False
        self.graph_count = 1245  # with self.max_graph_size = float('inf')
        self.max_graph_size = float('inf')

        # self.graph_count = 1234  # with self.max_graph_size = 5000
        # self.max_graph_size = 5000

        # self.graph_count = 1225  # with self.max_graph_size = 2190
        # self.max_graph_size = 2190

        super(Tree_of_Life, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped"""
        return [f'data_{i}.pt' for i in range(self.graph_count)]

    def download(self):
        pass

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return os.path.join(self.root, self.name, name)

    @property
    def num_classes(self) -> int:
        return 1

    def process(self):
        good_species_ids = self.load_good_species_ids()
        evolution_values_dict = np.load(os.path.join(self.raw_dir, 'evolution_values.npy'), allow_pickle=True).item()
        graph_path = os.path.join(self.raw_dir, 'graphs')
        feature_path = os.path.join(self.raw_dir, 'features')

        count = 0
        for species_id in tqdm(good_species_ids):
            y_i = evolution_values_dict.get(species_id)
            if y_i is not None:  # label is available
                # Graph
                graph = np.load(os.path.join(graph_path, f'{species_id}.npy'), allow_pickle=True)[1]
                if graph.shape[0] <= self.max_graph_size:
                    coo = coo_matrix(graph)
                    i = torch.LongTensor(np.vstack((coo.row, coo.col)))

                    # Feature
                    feature = torch.FloatTensor(np.load(os.path.join(feature_path, f'{species_id}.npy')))

                    data = Data(edge_index=torch.LongTensor(i).clone(),
                                x=feature.clone(),
                                y=torch.FloatTensor([y_i])
                                )

                    # GCNNorm()(data)  # sym gcn normalization
                    # NormalizeFeatures()(data)
                    torch.save(data, os.path.join(self.processed_dir, f'data_{count}.pt'))
                    count += 1
        # print(count)

    def load_good_species_ids(self):
        good_species_ids = np.load(os.path.join(self.raw_dir, 'good_species_ids_25.npy'))
        good_species_ids = map(lambda x: int(x.split('.')[0]), good_species_ids)
        return good_species_ids

    def len(self):
        return self.graph_count

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    gen = torch.Generator().manual_seed(0)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    test_size = len(data) - train_size - val_size
    splits = random_split(data, lengths=[train_size, val_size, test_size], generator=gen)
    return splits, [split.indices for split in splits]


def fixMoleculeNet(dataset, name):
    #Fix float labels and make them int...
    datalist = []
    for e in dataset:
        ndata = e.clone()
        ndata.x = ndata.x.float()
        ndata.y = ndata.y[0,0].long() #Default has a wierd list and is float TODO Check if this is also the case for any other moleculenet dataset
        if ndata.x.size()[0] > 0:
            datalist.append(ndata) 
        
    newDataset = MoleculeFix("data/"+name, datalist, name)
    return newDataset
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



def load_dataset(dataset_name):
    class IMDBPreTransform(object):
        def __call__(self, data):
            data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            data.x = F.one_hot(data.x, num_classes=136).to(torch.float)  # 136 in k-gnn?
            return data


    if dataset_name == 'Mutagenicity':
        data = TUDataset(root='data/', name='Mutagenicity', use_node_attr=True)
    elif dataset_name == "GOODMotif2":
        data ,meta_info = GOODMotif2.load("data/",domain="basis",generate=False,shift="covariate")    

    elif dataset_name == "GOODMotif_size":
        data ,meta_info = GOODMotif.load("data/",domain="size",generate=False,shift="covariate")


    elif dataset_name == "bbbp":
        data = MoleculeNet(root='data/',name="bbbp")
        data = fixMoleculeNet(data,dataset_name)
    elif dataset_name == 'MUTAG':
        data = TUDataset(root='data/', name='MUTAG', use_node_attr=True)
    elif dataset_name == "bamultishapes":
        data = BAMultiShapesDataset(root="data/")
    elif dataset_name == 'Proteins':
        data = TUDataset(root='data/', name='PROTEINS_full', use_node_attr=True)
    elif dataset_name == 'IMDB-B':
        data = TUDataset(root='data/', name='IMDB-BINARY', pre_transform=IMDBPreTransform())
    elif dataset_name == 'CMU-FACE_M':
        data = CMU_FACE_M(root='data/')
    elif dataset_name == 'Planted-Clique':
        data = Planted_Clique(root='data/')
    elif dataset_name == 'Tree-of-Life':
        data = Tree_of_Life(root='data/')
    else:
        raise NotImplementedError(f'Dataset: {dataset_name} is not implemented!')

    return data


if __name__ == '__main__':
    dataset = load_dataset('IMDB-B')
    splits, indices = split_data(dataset)
    train_set, valid_set, test_set = splits
    print(f'Size: {len(dataset)}')
    print(f'Labels sum: {sum([graph.y for graph in dataset]) / len(dataset)}')
