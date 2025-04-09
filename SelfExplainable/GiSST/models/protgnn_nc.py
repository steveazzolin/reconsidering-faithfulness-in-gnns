import torch
import torch.nn as nn
import os
from typing import List
from models.ProtGNN.models.train_gnns import train_NC, evaluate_NC, edge_mask
from models.ProtGNN.load_dataset import get_dataset
from models.ProtGNN.models.GCN import GCNNet, GCNNet_NC
from models.ProtGNN.models.GAT import GATNet, GATNet_NC
from models.ProtGNN.models.GIN import GINNet, GINNet_NC
import global_config as gl
import torch_geometric
from torch_geometric.data import Data

class protgnn_nc():
    def __init__(self, dataset, datasetName, deviceToUse=None, config=None):
        if deviceToUse == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"
            
            
        if config == None:
            self.config = ProtConfig(datasetName)
        else:
            self.config = config
            
        
        
        self.datasetName = datasetName
        self.dataset = dataset
        self.data = self.dataset[0].to(self.device)
        
        
        self.input_dim = self.dataset[0].num_node_features
        self.output_dim = int(torch.max(self.dataset[0].y+1))
        
        self.model = GnnNets_NC(self.input_dim, self.output_dim, self.config.model_args)
        self.model.to_device()
        
        #self.model.load_state_dict()
        #self.model.state_dict()
        
    
    def get_explanation(self, node_index, manual_hops = 2):
        self.model.eval()
        with torch.no_grad():
            data = self.data
            
            subgraph = torch_geometric.utils.k_hop_subgraph(node_index, 2, data.edge_index, relabel_nodes = True)
                
            #create new dataset graph out of subgraph
            sub_nodes = subgraph[0].tolist()
            sub_edge_index = subgraph[1]
            sub_hard_edge_mask = subgraph[3].cpu()
            
            for i in range(len(sub_nodes)):
                if sub_nodes[i] == node_index:
                    sub_node_index = i
            
            trueRow = torch.ones(len(data.x[0]))
            
            maskY = torch.zeros(len(data.y), dtype=torch.bool)
            maskX = torch.zeros(len(data.x), len(data.x[0]), dtype=torch.bool)
            for n in sub_nodes:
                maskY[n] = True
                maskX[n] = trueRow
            
            maskY = maskY.to(self.device)
            maskX = maskX.to(self.device)
            
            subDataX = torch.masked_select(data.x, maskX).reshape((len(sub_nodes), len(maskX[0])))
            subDataY = torch.masked_select(data.y, maskY)
            subData = Data(x=subDataX, edge_index=sub_edge_index, y=subDataY)
        

            logits, probs, embs, min_distances = self.model(subData)
        
            prot_index = torch.argmax(min_distances[sub_node_index])
            
            #print(self.model.model.prototype_vectors)
            prot = self.model.model.prototype_vectors[prot_index]
            #embed = embs[node_index]
            
            input = (self.data, embs, subData.edge_index, prot, 1)
            
            sub_edge_prob = edge_mask(input).cpu()
            
            edge_prob = torch.zeros(len(data.edge_index[0])).cpu()
            edge_prob.masked_scatter_(sub_hard_edge_mask, sub_edge_prob)
            
            return None, edge_prob
        
        
    def get_prediction_vector(self, nodes, data=None):
        #logits
        self.model.eval()
        with torch.no_grad():
            if data==None:
                data = self.data
            logits, probs, embs, min_distances = self.model(data)
            
            return logits[nodes]
        
    def train(self):
        print("Trying to train")
        train_NC(self.model, self.dataset, self.config, self.input_dim, self.output_dim)
        
    def test(self):
        data = self.dataset[0]
        criterion = nn.CrossEntropyLoss()
        eval_info = evaluate_NC(data, self.model, criterion)
        print(f'Test Loss: {eval_info["test_loss"]:.4f}, Test Accuracy: {eval_info["test_acc"]:.3f}')
        
        #print(eval_info['{}_embs'.format('test')])
        #print(eval_info['{}_min-dists'.format('test')])
        #print(eval_info['{}_embs'.format('test')].size())
        #print(eval_info['{}_min-dists'.format('test')].size())
        
        return eval_info["test_acc"]
        
    def save(self):
        #TODO make checkpoints dict use some central config value
        torch.save(self.model.state_dict(),"checkpoints/{}/{}".format("protGNN_nc", self.config.model_args.model_name, self.datasetName))
        
    def load(self):
        checkpoint = torch.load("checkpoints/{}/{}".format("protGNN_nc", self.config.model_args.model_name, self.datasetName))
        #gnnNets_NC.update_state_dict(checkpoint['net'])
        self.model.update_state_dict(checkpoint)
                    
#Models
def get_model(input_dim, output_dim, model_args):
    if model_args.model_name.lower() == 'gcn':
        return GCNNet(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gat':
        return GATNet(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gin':
        return GINNet(input_dim, output_dim, model_args)
    else:
        raise NotImplementedError


def get_model_NC(input_dim, output_dim, model_args):
    if model_args.model_name.lower() == 'gcn':
        return GCNNet_NC(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gat':
        return GATNet_NC(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gin':
        return GINNet_NC(input_dim, output_dim, model_args)
    else:
        raise NotImplementedError


class GnnBase(nn.Module):
    def __init__(self):
        super(GnnBase, self).__init__()

    def forward(self, data):
        data = data.to(self.device)
        logits, prob, emb1, emb2, min_distances = self.model(data)
        return logits, prob, emb1, emb2, min_distances

    def update_state_dict(self, state_dict):
        original_state_dict = self.state_dict()
        loaded_state_dict = dict()
        for k, v in state_dict.items():
            if k in original_state_dict.keys():
                loaded_state_dict[k] = v
        self.load_state_dict(loaded_state_dict)

    def to_device(self):
        self.to(self.device)

    def save_state_dict(self):
        pass


class GnnNets(GnnBase):
    def __init__(self, input_dim, output_dim, model_args):
        super(GnnNets, self).__init__()
        self.model = get_model(input_dim, output_dim, model_args)
        self.device = model_args.device

    def forward(self, data, protgnn_plus=False, similarity=None):
        data = data.to(self.device)
        logits, prob, emb1, emb2, min_distances = self.model(data, protgnn_plus, similarity)
        return logits, prob, emb1, emb2, min_distances


class GnnNets_NC(GnnBase):
    def __init__(self, input_dim, output_dim, model_args):
        super(GnnNets_NC, self).__init__()
        self.model = get_model_NC(input_dim, output_dim, model_args)
        self.device = model_args.device

    def forward(self, data):
        data = data.to(self.device)
        logits, prob, emb, min_distances = self.model(data)
        return logits, prob, emb, min_distances


#Configuration
class ProtConfig():
    def __init__(self, datasetName, sep=0.0, clst=0.0):
        self.data_args = DataParser(datasetName)
        self.model_args = ModelParser()
        self.train_args = TrainParser()
        self.sep = sep
        self.clst = clst


class DataParser():
    def __init__(self, datasetName):
        super().__init__()
        self.dataset_name = datasetName
        self.dataset_dir = gl.GlobalConfig.dataset_dir
        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
        self.seed = 1


class GATParser():# hyper-parameter for gat model
    def __init__(self):
        super().__init__()
        self.gat_dropout = 0.6    # dropout in gat layer
        self.gat_heads = 10         # multi-head
        self.gat_hidden = 10        # the hidden units for each head
        self.gat_concate = True    # the concatenation of the multi-head feature
        self.num_gat_layer = 3

class ModelParser():
    def __init__(self):
        super().__init__()
        self.device = 0
        self.model_name: str = 'gcn'
        self.checkpoint: str = gl.GlobalConfig.checkpoint_dir
        self.concate: bool = False                     # whether to concate the gnn features before mlp
        self.latent_dim: List[int] = [128, 128, 128]   # the hidden units for each gnn layer
        self.readout: 'str' = 'max'                    # the graph pooling method
        self.mlp_hidden: List[int] = []                # the hidden units for mlp classifier
        self.gnn_dropout: float = 0.0                  # the dropout after gnn layers
        self.dropout: float = 0.5                      # the dropout after mlp layers
        self.adj_normlize: bool = True                 # the edge_weight normalization for gcn conv
        self.emb_normlize: bool = False                # the l2 normalization after gnn layer
        self.enable_prot = True                        # whether to enable prototype training
        self.num_prototypes_per_class = 5              # the num_prototypes_per_class
        self.gat_dropout = 0.6  # dropout in gat layer
        self.gat_heads = 10  # multi-head
        self.gat_hidden = 10  # the hidden units for each head
        self.gat_concate = True  # the concatenation of the multi-head feature
        self.num_gat_layer = 3
    def process_args(self) -> None:
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass


class TrainParser():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.005
        self.batch_size = 24
        self.weight_decay = 0.0
        self.max_epochs = 800
        self.save_epoch = 10
        self.early_stopping = 80
        self.last_layer_optimizer_lr = 1e-4            # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}      # the learning rates of the joint training optimizer
        self.warm_epochs = 10                          # the number of warm epochs
        self.proj_epochs = 100                         # the epoch to start mcts
        self.sampling_epochs = 100                     # the epoch to start sampling edges
        self.nearest_graphs = 10                       # number of graphs in projection