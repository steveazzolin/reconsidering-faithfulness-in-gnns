import torch
import torch.nn as nn
import os
from typing import List
from models.ProtGNN.models.train_gnns import train_GC, test_GC
from models.ProtGNN.load_dataset import get_dataset, get_dataloader
from models.ProtGNN.models.GCN import GCNNet, GCNNet_NC
from models.ProtGNN.models.GAT import GATNet, GATNet_NC
from models.ProtGNN.models.GIN import GINNet, GINNet_NC
import global_config as gl
from torch_geometric.data import Batch
from torch_geometric.utils import index_to_mask
from models.ProtGNN.my_mcts import mcts

class protgnn_gc():
    def __init__(self, dataset, datasetName, deviceToUse=None, config=None):
        if deviceToUse == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = deviceToUse
            
            
        if config == None:
            self.config = ProtConfig(datasetName)
        else:
            self.config = config
        
        self.datasetName = datasetName
        self.dataset = dataset
        input_dim = self.dataset.num_node_features
        output_dim = int(self.dataset.num_classes)
        
        self.dataloader = get_dataloader(self.dataset, self.config.train_args.batch_size, data_split_ratio=self.config.data_args.data_split_ratio)
        
        self.model = GnnNets(input_dim, output_dim, self.config.model_args)
        self.model.to_device()
        
        #self.model.load_state_dict()
        #self.model.state_dict()
        
        
    def train(self):
        print("Trying to train")
        
        train_GC(self.model, self.dataset, self.dataloader, self.config)
        
    def test(self, testSet=None):
        if testSet == None:
            testSet = self.dataloader['test']
        criterion = nn.CrossEntropyLoss()
        test_state, _, _ = test_GC(testSet, self.model, criterion)
        print(f"Test: | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
        #append_record("loss: {:.3f}, acc: {:.3f}".format(test_state['loss'], test_state['acc']))
        return test_state['acc']
        
    def save(self):
        #TODO make checkpoints dict use some central config value
        torch.save(self.model.state_dict(),"checkpoints/{}/{}".format("protGNN_gc", self.config.model_args.model_name + "_" + self.datasetName))
        
    def load(self):
        checkpoint = torch.load("checkpoints/{}/{}".format("protGNN_gc", self.config.model_args.model_name + "_" + self.datasetName))
        self.model.update_state_dict(checkpoint)
        
    def get_explanation(self, graph_index):
        data = self.dataset[graph_index]
        databatch = Batch.from_data_list([data])
        
        with torch.no_grad():
            _, _, _, _, min_distances =self.model(databatch)
    
        #print(min_distances)
        prot_index = torch.argmin(min_distances)
        
        
        #print(prot_index)
        
        coalition, _, _ = mcts(data, self.model, self.model.model.prototype_vectors[prot_index])
        
        node_mask = index_to_mask(torch.tensor(coalition), data.x.size()[0])

        
        edge_prob = torch.zeros(len(data.edge_index[0]))
        
        for e in range(len(data.edge_index[0])):
            edge_prob[e] = (int(node_mask[int(data.edge_index[0][e])]) + int(node_mask[int(data.edge_index[1][e])]))/2
        
        
        node_mask = torch.reshape(node_mask, (len(node_mask), 1))
        
        return node_mask.cpu(), edge_prob.cpu()
    
        #Likely a better solution but it is slower by a factor of the number of prototypes. usually 5
        # data = self.dataset[graph_index]
        # databatch = Batch.from_data_list([data])
        
        # with torch.no_grad():
        #     logits, _, _, _, min_distances =self.model(databatch)
            
        #     y_hat=logits.argmax(-1).item()
        #     #print(y_hat)
        #     #print(min_distances)
            
        # node_mask = torch.zeros(data.x.size()[0]) #init node mask we can average across later
        
        # numer_of_prototypes = self.config.model_args.num_prototypes_per_class
    
        # for i in range(numer_of_prototypes*y_hat,numer_of_prototypes*(y_hat+1)):
        #     prot_index = torch.argmin(min_distances)
        #     coalition, _, _ = mcts(data, self.model, self.model.model.prototype_vectors[prot_index])
        #     node_mask = node_mask + index_to_mask(torch.tensor(coalition), data.x.size()[0]).float()
            
        # node_mask = node_mask/numer_of_prototypes
        
        # edge_prob = torch.zeros(len(data.edge_index[0]))
        
        # for e in range(len(data.edge_index[0])):
        #     edge_prob[e] = (int(node_mask[int(data.edge_index[0][e])]) + int(node_mask[int(data.edge_index[1][e])]))/2
        
        # node_mask = torch.reshape(node_mask, (len(node_mask), 1))
        
        # return node_mask.cpu(), edge_prob.cpu()
    
    def get_prediction_vector(self, data):
        databatch = Batch.from_data_list([data])
        
        with torch.no_grad():
            logits, probs, _, _, _ =self.model(databatch)
        
        return logits
                    
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

def create_config(datasetName, base_config, data_args, model_args, train_args):
    config = ProtConfig(datasetName)
    config.sep = base_config['sep']
    config.clst = base_config['clst']
    config.div = base_config['div']
    
    config.data_args.dataset_name = datasetName
    config.data_args.task = data_args['task']
    config.data_args.random_split: bool = data_args['random_split']
    config.data_args.data_split_ratio: List = data_args['data_split_ratio']   # the ratio of training, validation and testing set for random split
    config.data_args.seed = data_args['seed']
    
    config.model_args.model_name: str = model_args['model_name']
    config.model_args.checkpoint: str = model_args['checkpoint']
    config.model_args.concate: bool =   model_args['concate']                  # whether to concate the gnn features before mlp
    config.model_args.latent_dim: List[int] = model_args['latent_dim']    # the hidden units for each gnn layer
    config.model_args.readout: 'str' =  model_args['readout']                   # the graph pooling method
    config.model_args.mlp_hidden: List[int] = model_args['mlp_hidden']               # the hidden units for mlp classifier
    config.model_args.gnn_dropout: float = model_args['gnn_dropout']                  # the dropout after gnn layers
    config.model_args.dropout: float =  model_args['dropout']                    # the dropout after mlp layers
    config.model_args.adj_normlize: bool = model_args['adj_normlize']                 # the edge_weight normalization for gcn conv
    config.model_args.emb_normlize: bool = model_args['emb_normlize']               # the l2 normalization after gnn layer
    config.model_args.enable_prot = model_args['enable_prot']                        # whether to enable prototype training
    config.model_args.num_prototypes_per_class = model_args['num_prototypes_per_class']              # the num_prototypes_per_class
    config.model_args.gat_dropout = model_args['gat_dropout']  # dropout in gat layer
    config.model_args.gat_heads = model_args['gat_heads']  # multi-head
    config.model_args.gat_hidden = model_args['gat_hidden']  # the hidden units for each head
    config.model_args.gat_concate = model_args['gat_concate']  # the concatenation of the multi-head feature
    config.model_args.num_gat_layer = model_args['num_gat_layer']
    
    config.train_args.learning_rate = train_args['learning_rate'] 
    config.train_args.batch_size = train_args['batch_size'] 
    config.train_args.weight_decay = train_args['weight_decay'] 
    config.train_args.max_epochs = train_args['max_epochs'] 
    config.train_args.save_epoch = train_args['save_epoch'] 
    config.train_args.early_stopping = train_args['early_stopping'] 
    config.train_args.last_layer_optimizer_lr = train_args['last_layer_optimizer_lr']             # the learning rate of the last layer
    config.train_args.joint_optimizer_lrs = train_args['joint_optimizer_lrs']     # the learning rates of the joint training optimizer
    config.train_args.warm_epochs = train_args['warm_epochs']                          # the number of warm epochs
    config.train_args.proj_epochs = train_args['proj_epochs']                          # the epoch to start mcts
    config.train_args.sampling_epochs = train_args['sampling_epochs']                     # the epoch to start sampling edges
    config.train_args.nearest_graphs = train_args['nearest_graphs']                        # number of graphs in projection
    
    return config
    

#Configuration
class ProtConfig():
    def __init__(self, datasetName, sep=0., clst=0.0, div = 0.0):
        self.data_args = DataParser(datasetName)
        self.model_args = ModelParser()
        self.train_args = TrainParser()
        self.sep = sep
        self.clst = clst
        self.clst = div


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