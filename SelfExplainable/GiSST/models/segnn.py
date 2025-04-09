import torch
import numpy as np
from models.SEGNN.models.ExplainGNN import ExplainGNN
from models.SEGNN.dataset import get_labeled, TestLoader, TrainLoader
from models.SEGNN.utils import tensor2onehot, idx_to_mask
import os.path
import global_config as gl
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_dense_adj

from torch_geometric.explain import groundtruth_metrics
from torch_geometric.utils import k_hop_subgraph, mask_to_index, to_undirected
from torch_geometric.data import Data


class Segnn():
    def __init__(self, dataset, datasetName, deviceToUse=None, config = None, _run = None):
        if deviceToUse == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"
        
        self.config = config
        
        if config == None: #Use a default config if none was set
            self.config = SegnnConfig()
        
        self.datasetName = datasetName
        self.dataset = dataset
        self.data = dataset[0].to(self.device) #Node classification only...
        self.label_nodes = get_labeled(self.data.train_mask, self.data.edge_index, self.config.hop, self.device)
        
        self._run = _run
        
        #Setup model for dataset
        self.model = ExplainGNN(self.config,
            nfeat=self.dataset[0].x.shape[1],
            device=self.device,
            _run = self._run).to(self.device)
    
    def train(self):
        print("trying to train")      
             
        data = self.data
        train_mask = data.train_mask
        val_mask = data.val_mask
        
        train_loader = TrainLoader(train_mask, data.edge_index, sample_size=self.config.batch_size, hop=self.config.hop, device=self.device)
        val_loader = TestLoader(val_mask, data.edge_index, 16, self.config.hop, self.device)
        self.model.fit(data.x, data.edge_index, data.edge_attr, data.y, self.label_nodes, train_loader,val_loader, train_iters=self.config.epochs, verbose=self.config.debug)

    def test(self, hop = None):
        if hop == None:
            hop = self.config.hop
        
        print("Trying to test")
        data = self.data
        test_mask = data.test_mask
        test_loader = TestLoader(test_mask, data.edge_index, sample_size=1, hop=hop, device=self.device)
        
        ACC, mAP = self.model.test(self.label_nodes, test_loader)
        print("Accuracy: {:.4f}, mAP: {:.4f}".format(ACC, mAP))
        
        return ACC
        
    def save(self):
        torch.save(self.model.state_dict(),os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName))
        
    def load(self):
        self.model.features = self.data.x
        self.model.labels = self.data.y
        self.model.onehot_labels =  tensor2onehot(self.data.y)
        self.model.edge_index = self.data.edge_index
        self.model.load_state_dict(torch.load(os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName)))
    
    def getPredictionVectors(self, mask, testdata, hop=None):
        if hop == None:
            hop = self.config.hop
        
        test_loader = TestLoader(mask, testdata.edge_index, sample_size=1, hop=hop, device=self.device)
        #print(test_loader[0])
        #nodes, sub_edge_index, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops=hops, edge_index=self.data.edge_index)
        
        preds = []
        for i in range(len(test_loader)):
            nodes = test_loader[i]
            preds.append(self.model.predictionVector(self.label_nodes, nodes))#, testdata.x, testdata.edge_index, testdata.edge_attr))
            
        preds = torch.cat(preds, 0)
        return preds
    
    def getPredictionVectorsWithSubGraph(self, node_indexes, testdatasets, hop=None):
        if hop == None:
            hop = self.config.hop
            
        
        preds = []
        for testdata, node_index in zip(testdatasets, node_indexes):
            nodeList = torch.unique(testdata.edge_index)
            nodes = (torch.tensor([node_index], device = self.device),
                nodeList.to(self.device),
                torch.zeros(testdata.edge_index.shape[1],  device = self.device, dtype=int),
                testdata.edge_index.to(self.device),
                torch.zeros(testdata.edge_index.shape[1], device = self.device, dtype=int))
        
            preds.append(self.model.predictionVector(self.label_nodes, nodes)) #, testdata.x, testdata.edge_index, testdata.edge_attr))
            
        preds = torch.cat(preds, 0)
        return preds
        
    def getSubGraphBasedOnExplanation(self, mask, testdata, hop=None, topk_edges_used = 20, K = 3):
        if hop == None:
           hop = self.config.hop
        
        test_loader = TestLoader(mask, testdata.edge_index, sample_size=1, hop=hop, device=self.device)
        #mask_to_index(testmask)
        
        
        preds = []
        for i in range(len(test_loader)):
            edge_mask, edge_index = self.model.explain_structure(self.label_nodes, test_loader[i], K=K) #self.model.explain_structure(self.label_nodes, test_loader[i], testdata.x, testdata.edge_index, testdata.edge_attr, K=K)
            boolean_mask = self._createBooleanMask(edge_mask, topk_edges_used)
            
            preds.append(Data(x = testdata.x, edge_index=edge_index[:, boolean_mask], y = testdata.y))
            #print(preds[i])
        
        return preds
        
    def _createBooleanMask(self, edge_mask, topk = 10):
        if(len(edge_mask) < topk):
            topk = len(edge_mask)
            
        #print(len(edge_mask))
            
        top_k = torch.topk(edge_mask, topk)
        edge_mask.scatter_(0, top_k.indices, 1)
        mask = edge_mask.eq(1)
        return mask
        #res = torch.where(edge_mask > targetValue-threshold, True, False)
        #return res
        
    # def get_edge_explanation(self, node_index, manual_hops = 2):
    #     final_hops = manual_hops
    #     if manual_hops < self.config.hop:
    #         final_hops = self.config.hop
        
    #     nodes, sub_edge_index, _, hard_edge_mask = k_hop_subgraph(int(node_index), num_hops=final_hops, edge_index=self.data.edge_index)
        
    #     #Create the equivelant of a testloader entry
    #     node = (torch.tensor([node_index], device = self.device),
    #             nodes.to(self.device),
    #             torch.zeros(sub_edge_index.shape[1],  device = self.device, dtype=int),
    #             sub_edge_index.to(self.device),
    #             torch.zeros(sub_edge_index.shape[1], device = self.device, dtype=int))
        
    #     edge_mask, _ = self.model.explain_structure(self.label_nodes, node, K=self.config.K)
    #     edge_mask = edge_mask + 1 #TODO fiqure good argument why required
        
    #     edge_importance = torch.zeros(len(self.data.edge_index[0])).to(edge_mask.get_device())
    #     edge_importance.masked_scatter_(hard_edge_mask, edge_mask) #Unmask based on the hard mask used earlier
        
    #     return edge_importance.cpu()
    
    def get_explanation(self, node_index, manual_hops = 1):
        final_hops = manual_hops
        if manual_hops < self.config.hop:
            final_hops = self.config.hop
        
        nodes, sub_edge_index, _, hard_edge_mask = k_hop_subgraph(int(node_index), num_hops=final_hops, edge_index=self.data.edge_index)
        
        #Create the equivelant of a testloader entry
        node = (torch.tensor([node_index], device = self.device),
                nodes.to(self.device),
                torch.zeros(sub_edge_index.shape[1],  device = self.device, dtype=int),
                sub_edge_index.to(self.device),
                torch.zeros(sub_edge_index.shape[1], device = self.device, dtype=int))
        
        edge_mask, _ = self.model.explain_structure(self.label_nodes, node, K=self.config.K)
        edge_mask = edge_mask + 1 #TODO fiqure good argument why required
        
        edge_importance = torch.zeros(len(self.data.edge_index[0])).to(edge_mask.get_device())
        edge_importance.masked_scatter_(hard_edge_mask, edge_mask) #Unmask based on the hard mask used earlier
        
        return None, edge_importance.cpu()
        
class SegnnConfig():
    def __init__(self, config_dict = None):
        self.modelname = 'SEGNN'
        self.debug = True
        self.seed = 12
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 128
        self.K = 10
        self.init = False
        self.nlayer = 2
        self.alpha = 0.5
        self.beta1 = 1
        self.beta2 = 1
        self.T = 1
        self.hop = 1
        self.model = 'DeGNN' #Can be 'MLP','GCN','DeGNN'
        self.epochs = 100
        self.attr_mask = 0.5
        self.batch_size = 128
        if config_dict != None:
            self.modelname = config_dict["modelname"]
            self.debug = config_dict["debug"]
            self.seed = config_dict["seed"]
            self.lr = config_dict["lr"]
            self.weight_decay = config_dict["weight_decay"]
            self.hidden = config_dict["hidden"]
            self.K = config_dict["K"]
            self.init = config_dict["init"]
            self.nlayer = config_dict["nlayer"]
            self.alpha = config_dict["alpha"]
            self.beta1 = config_dict["beta1"]
            self.beta2 = config_dict["beta2"]
            self.T = config_dict["T"]
            self.hop = config_dict["hop"]
            self.model = config_dict["model"]
            self.epochs = config_dict["epochs"]
            self.attr_mask = config_dict["attr_mask"]
            self.batch_size = config_dict["batch_size"]
        