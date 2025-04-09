import torch
import models.PIGNN.protop_gnnNets as protognn
import models.PIGNN.tes_gnnNets as tesgnn
from models.PIGNN.dataset import get_dataset, get_dataloader
from torch_geometric.loader import DataLoader
import models.PIGNN.train_protopgnns as proto
import models.PIGNN.train_tesgnns as tes
import os.path
import global_config as gl
import numpy as np
import random
from torch_geometric.utils import index_to_mask
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Subset


class Pignn():
    def __init__(self, dataset, datasetName, deviceToUse=None, config = None, _run = None):
        if deviceToUse == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"
            

        
        gc = gl.GlobalConfig()
        self.config = config
        self.datasetName = datasetName    
    
        if config == None: #Use a default config if none was set
            self.config = PignnConfig()
        
        if not self.config.param.graph_classification:
            #load data in for NC
            self.data = dataset[0].to(self.device)
            
            
            self.dataset = PignnDataset(self.data)
            num_node_features = len(self.data.x[0])
            num_classes = torch.max(self.data.y+1)
        
        else:
            #create dataloader for GC
            train = Subset(dataset, dataset.index_train)
            eval = Subset(dataset, dataset.index_val)
            test = Subset(dataset, dataset.index_test)
            
            dataloader = dict()
            dataloader['train'] = DataLoader(train, batch_size=self.config.param.batch_size, shuffle=True, worker_init_fn=None)
            dataloader['eval'] = DataLoader(eval, batch_size=self.config.param.batch_size, shuffle=False, worker_init_fn=None)
            dataloader['test'] = DataLoader(test, batch_size=self.config.param.batch_size, shuffle=False, worker_init_fn=None)
            
            self.dataset = dataloader
            num_classes = dataset.num_classes
            num_node_features = dataset.num_features
        #print(num_classes)
        
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(False)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        
        
        if self.config.param.graph_classification:
            dataloader_params = {'batch_size': self.config.param.batch_size,
                                'random_split_flag': self.config.dataConf.random_split_flag,
                                'data_split_ratio': self.config.dataConf.data_split_ratio,
                                'seed': self.config.seed}
        
        self.train_params = {'num_epochs': self.config.param.num_epochs,
                    'num_early_stop': self.config.param.num_early_stop,
                    'milestones': self.config.param.milestones,
                    'gamma': self.config.param.gamma,
                    #'use_pretrained':use_pretrained,
                    #'pretrained_dir':pretrained_dir
                    }
        self.optimizer1_params = {'lr': self.config.param.learning_rate1,
                            'weight_decay': self.config.param.weight_decay1}
        self.optimizer2_params = {'lr': self.config.param.learning_rate2,
                            'weight_decay': self.config.param.weight_decay2}
        
        
        #Setup model for dataset
        if self.config.type == "tes":
            self.model = tesgnn.get_gnnNets(num_node_features, num_classes, self.config).to(self.device)
            
            if self.config.param.graph_classification:
                self.trainer = tes.TrainModel(model=self.model,
                                    dataset=self.dataset,
                                    device=self.device,
                                    graph_classification=self.config.param.graph_classification,
                                    save_dir=os.path.join(gc.checkpoint_dir,
                                                        "PIGNNProto"),
                                    save_name=f'{self.config.gnn_name}_{self.config.param.num_basis_per_class}_{len(self.config.param.gnn_latent_dim)}l',
                                    dataloader_params=dataloader_params)
            else:
                self.trainer = tes.TrainModel(model=self.model,
                                        dataset=self.dataset,
                                        device=self.device,
                                        graph_classification=self.config.param.graph_classification,
                                        save_dir=os.path.join(gc.checkpoint_dir,
                                                        "PIGNNProto"),
                                        save_name=f'{self.config.gnn_name}_{self.config.param.num_basis_per_class}_{len(self.config.param.gnn_latent_dim)}l')

        else:
            self.model = protognn.get_gnnNets(num_node_features, num_classes, self.config).to(self.device)
            
            if self.config.param.graph_classification:
                self.trainer = proto.TrainModel(model=self.model,
                                    dataset=self.dataset,
                                    device=self.device,
                                    graph_classification=self.config.param.graph_classification,
                                    save_dir=os.path.join(gc.checkpoint_dir,
                                                        "PIGNNProto"),
                                    save_name=f'{self.config.gnn_name}_{self.config.param.num_basis_per_class}_{len(self.config.param.gnn_latent_dim)}l',
                                    dataloader_params=dataloader_params)
            else:
                self.trainer = proto.TrainModel(model=self.model,
                                        dataset=self.dataset,
                                        device=self.device,
                                        graph_classification=self.config.param.graph_classification,
                                        save_dir=os.path.join(gc.checkpoint_dir,
                                                        "PIGNNProto"),
                                        save_name=f'{self.config.gnn_name}_{self.config.param.num_basis_per_class}_{len(self.config.param.gnn_latent_dim)}l')
            
        
    
    def train(self):
        print("trying to train")      
        self.trainer.train(train_params=self.train_params, optimizer1_params=self.optimizer1_params, optimizer2_params=self.optimizer2_params)

    def test(self):
        print("trying to test")
        self.model.eval()
        if self.config.param.graph_classification:
            test_loss, test_acc, preds = self.trainer.test()
        else:
            data = self.data
            test_loss, preds = self.trainer._eval_batch(data, data.y, mask=data.test_mask)
            test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}")
        return test_acc
        
    def save(self):
        state = {'net': self.model.state_dict(), 'basis_data': self.model.basis_data}
        torch.save(state, os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName))
        #torch.save(self.trainer.model.state_dict(), os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName))
        
    def load(self):
        self.model = self.model.to("cpu")
        saved = torch.load(os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName))
        state_dict = saved['net']
        self.model.basis_data = saved['basis_data']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.trainer.model = self.model
        
    def get_edge_explanation_old(self, node_index, manual_hops = 2):
        #placeholder
        if self.config.param.graph_classification:
            print("gc")
            #gc get data from basis vectors
            #out = self.model(data=data)
            #logit, embs, l2s = out
            #x = ( - l2s[:, basis_id]).detach().numpy()
            #x = (x-x.min())/(x.max()-x.min())
            #nodeMap = [x[i] if i in nodes else 0 for i in range(data.num_nodes)]
        else:
            data = self.data
            #print("y", data.y)
            subgraph = torch_geometric.utils.k_hop_subgraph(node_index, 5, data.edge_index, relabel_nodes = True)
            #print(subgraph)
            
            
            
            #create new dataset graph out of subgraph
            sub_nodes = subgraph[0].tolist()
            sub_edge_index = subgraph[1]
            sub_hard_edge_mask = subgraph[3].to(self.device)
            
            for i in range(len(sub_nodes)):
                if sub_nodes[i] == node_index:
                    sub_node_index = i
            
            #print(subgraph)
            
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
            
            
            #subData = data #right now just running with subgraph being entire graph, very slow. But also the case for 4 hops from the middle of the graph basically
            with torch.no_grad():
                out = self.model(data=subData)
                ___, ___, l2s = out
            l2s = l2s.cpu()
            
            """
            for basis_id in range(len(self.model.basis_data)):
                mpl.rcParams['figure.dpi'] = 300
                fig, ax = plt.subplots(figsize=(6, 6))
                
                #basis_id=0 #select prototype you want to visualize
                basis_node_id = l2s[:,basis_id].argmin().item()
                subgraphBasis = torch_geometric.utils.k_hop_subgraph(basis_node_id, 2, subData.edge_index)
                nodes = subgraphBasis[0].tolist()
                x = ( - l2s[:, basis_id]).detach()            
                x = (x-x.min())/(x.max()-x.min())
                
                cmap = [x[i] if i in nodes else 0 for i in range(subData.num_nodes)]
        
                g = torch_geometric.utils.to_networkx(subData, to_undirected=True)
                pos = nx.kamada_kawai_layout(g)
                nx.draw(g, node_color=cmap, pos=pos, cmap='Blues', edgecolors='black', ax=ax)
                nx.draw(g.subgraph([node_index]), pos=pos, edgecolors='red', ax=ax)
                
                #Set common labels
                fig.tight_layout(pad=5)
                #plt.show()
                path = "checkpoints/PignnFig/"+str(node_index)+'_'+str(basis_id)+".png"
                plt.savefig(path)
            """    
            #print(l2s)
            #print(l2s[sub_node_index,:])
            #print(l2s[sub_node_index,:].argmin().item())
            
            basis_id = l2s[sub_node_index,:].argmin().item()
            basis_node_id = l2s[:,basis_id].argmin().item()
            x = ( - l2s[:, basis_id]).detach()            
            x = (x-x.min())/(x.max()-x.min())
            
            #subgraphBasis = torch_geometric.utils.k_hop_subgraph(basis_node_id, 2, subData.edge_index)
            #nodes = subgraphBasis[0].tolist()
            #x = [x[i] if i in nodes else 0 for i in range(subData.num_nodes)]
            
            sub_edge_prob = torch.zeros(len(subData.edge_index[0]))
            
            for e in range(len(subData.edge_index[0])):
                #print(x[int(data.edge_index[0][e])])
                #print(x[int(data.edge_index[1][e])])
                #sub_edge_prob[e] = (x[int(subData.edge_index[0][e])] + x[int(subData.edge_index[1][e])])/2
                sub_edge_prob[e] = torch.maximum(x[int(subData.edge_index[0][e])], x[int(subData.edge_index[1][e])])
        
        sub_edge_prob = sub_edge_prob.to(self.device)
        edge_prob = torch.zeros(len(data.edge_index[0])).to(self.device)
        edge_prob.masked_scatter_(sub_hard_edge_mask, sub_edge_prob)
        
        return edge_prob.cpu()
    
    def get_node_feat_explanation_old(self, node_index):
                #placeholder
        if self.config.param.graph_classification:
            print("gc")
            #gc get data from basis vectors
            #out = self.model(data=data)
            #logit, embs, l2s = out
            #x = ( - l2s[:, basis_id]).detach().numpy()
            #x = (x-x.min())/(x.max()-x.min())
            #nodeMap = [x[i] if i in nodes else 0 for i in range(data.num_nodes)]
        else:
            data = self.data
            #print("y", data.y)
            subgraph = torch_geometric.utils.k_hop_subgraph(node_index, 5, data.edge_index, relabel_nodes = True)
            #print(subgraph)
            
            
            
            #create new dataset graph out of subgraph
            sub_nodes = subgraph[0].tolist()
            sub_edge_index = subgraph[1]
            sub_hard_edge_mask = subgraph[3].to(self.device)
            
            for i in range(len(sub_nodes)):
                if sub_nodes[i] == node_index:
                    sub_node_index = i
            
            #print(subgraph)
            
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
            
            
            #subData = data #right now just running with subgraph being entire graph, very slow. But also the case for 4 hops from the middle of the graph basically
            with torch.no_grad():
                out = self.model(data=subData)
                ___, ___, l2s = out
            l2s = l2s
            
            """
            for basis_id in range(len(self.model.basis_data)):
                mpl.rcParams['figure.dpi'] = 300
                fig, ax = plt.subplots(figsize=(6, 6))
                
                #basis_id=0 #select prototype you want to visualize
                basis_node_id = l2s[:,basis_id].argmin().item()
                subgraphBasis = torch_geometric.utils.k_hop_subgraph(basis_node_id, 2, subData.edge_index)
                nodes = subgraphBasis[0].tolist()
                x = ( - l2s[:, basis_id]).detach()            
                x = (x-x.min())/(x.max()-x.min())
                
                cmap = [x[i] if i in nodes else 0 for i in range(subData.num_nodes)]
        
                g = torch_geometric.utils.to_networkx(subData, to_undirected=True)
                pos = nx.kamada_kawai_layout(g)
                nx.draw(g, node_color=cmap, pos=pos, cmap='Blues', edgecolors='black', ax=ax)
                nx.draw(g.subgraph([node_index]), pos=pos, edgecolors='red', ax=ax)
                
                #Set common labels
                fig.tight_layout(pad=5)
                #plt.show()
                path = "checkpoints/PignnFig/"+str(node_index)+'_'+str(basis_id)+".png"
                plt.savefig(path)
            """    
            #print(l2s)
            #print(l2s[sub_node_index,:])
            #print(l2s[sub_node_index,:].argmin().item())
            
            basis_id = l2s[sub_node_index,:].argmin().item()
            basis_node_id = l2s[:,basis_id].argmin().item()
            #subgraphBasis = torch_geometric.utils.k_hop_subgraph(basis_node_id, 2, subData.edge_index)
            #nodes = subgraphBasis[0].tolist()
            x = ( - l2s[:, basis_id]).detach()            
            x = (x-x.min())/(x.max()-x.min())
            
            #subgraphBasis = torch_geometric.utils.k_hop_subgraph(basis_node_id, 2, subData.edge_index)
            #nodes = subgraphBasis[0].tolist()
            #for i in range(len(x)):
            #    x[i] = x[i] if i in nodes else 0
            #x = [x[i] if i in nodes else 0 for i in range(subData.num_nodes)]
        
        hard_node_mask = index_to_mask(subgraph[0], data.num_nodes)
        
        node_prob = torch.zeros(data.num_nodes).to(self.device)
        node_prob = node_prob.masked_scatter(hard_node_mask, x)
        node_prob = torch.reshape(node_prob, (data.num_nodes, 1))
        #node_prob = torch.reshape()
        
        return node_prob
    
    def get_explanation(self, node_index, sparsity = 0.5, manual_hops = 2):
        nodes_to_plot, edge_index2, _, hard_edge_mask = torch_geometric.utils.k_hop_subgraph(node_index, 5, self.data.edge_index, relabel_nodes=True)
        mapping = {k.item(): i for i,k in enumerate(nodes_to_plot)}
        data = Data(x=self.data.x[nodes_to_plot], edge_index=edge_index2)
        
        node_id = mapping[node_index]
        
        with torch.no_grad():
            out, embs, similarity=self.model(data.x.float(), data.edge_index)    

            y_hat=out[node_id,:].argmax(-1).item()
            
            mask=torch.zeros(len(data.x)).to(self.device)
            to_mask = torch_geometric.utils.k_hop_subgraph(node_id, 3, data.edge_index)[0].tolist()
            to_mask.remove(node_id)
            
            for i in range(self.model.num_basis_per_class*y_hat,self.model.num_basis_per_class*(y_hat+1)):
                if self.config.type == "tes": #Uses cosine similarity for tesnet
                    cosines_i = similarity[to_mask,i].detach() 
                    mask[to_mask] += self.model.classifier_weights[i, y_hat].detach()*cosines_i
                else: #Uses l2 distance for similarity in protopnet
                    sim_i = -similarity[to_mask,i] 
                    sim_i = (sim_i-sim_i.min())/(sim_i.max()-sim_i.min())
                    mask[to_mask] += self.model.classifier_weights[i, y_hat].detach() * sim_i
                    #print(cosines_i)
                    



            mask=(mask-mask.min())/(mask.max()-mask.min()+1e-8)
            #print(mask)
            p=np.percentile(mask.cpu(), 100*sparsity)
            mask[mask<p]=0
            edge_mask=torch.zeros(data.edge_index.shape[1])
            for i, (a,b) in enumerate(data.edge_index.T):
                edge_mask[i]=(mask[a]+mask[b])/2
            edge_mask[edge_mask>0]=1
            

            
            
            edge_mask_fin = torch.zeros(len(self.data.edge_index[0]))
            edge_mask_fin.masked_scatter_(hard_edge_mask.cpu(), edge_mask.cpu())
            
            hard_node_mask = index_to_mask(nodes_to_plot.cpu(), self.data.num_nodes)
            node_prob = torch.zeros(self.data.num_nodes)
            node_prob = node_prob.masked_scatter(hard_node_mask.cpu(), mask.cpu())
            node_prob = torch.reshape(node_prob, (self.data.num_nodes, 1))
            
            return node_prob, edge_mask_fin

        
    def get_prediction_vector(self, nodes, data=None):
        self.model.eval()
        
        #print(data)
        
        if data == None:
            data = self.dataset.data.to(self.device)
        
        #print("data", data)
        #print()
        #nodes = torch.from_numpy(nodes)
        #print(nodes)
        #test_mask = index_to_mask(nodes, data.x.shape[0])
        
        """
        if self.config.param.graph_classification:
            losses, preds, accs = [], [], []
            for batch in self.loader['test']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                accs.append(batch_preds == batch.y)
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
        else:
            #
            test_loss, preds = self.trainer._eval_batch(data, data.y, mask=test_mask)
            #test_acc = (preds[test_mask] == data.y[test_mask]).float().mean().item()
        #print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}")
        
        print("preds", len(preds))
        print("preds", preds)
        """
        with torch.no_grad():
            out = self.model(data=data)
            logits, _, _ = out
        
        return logits[nodes]

class PignnDataset():
    def __init__(self, data):
        self.data = data
        self.num_node_features = len(data.x[0])
        self.num_classes = torch.max(self.data.y) + 1


class PignnConfig():
    def __init__(self, cfg = None, modelCfg = None, dataCfg = None, seed = None):
        self.type = "proto"
        self.modelname = "PignnProto"
        self.seed = 0
        self.gnn_name = "gcn"
        self.param = modelParams(modelCfg)
        self.dataConf = dataParams(dataCfg)
        
        if cfg != None:
            self.type = cfg['type']
            self.modelname = cfg['modelname']
            self.gnn_name = cfg['gnn_name']
        
        if seed != None:
            self.seed = seed
        
class modelParams():
    def __init__(self, modelCfg = None):
        self.learning_rate1 = 1e-4
        self.learning_rate2 =  0.01
        self.weight_decay1 = 0
        self.weight_decay2 = 0
        self.milestones = None
        self.gamma = None
        self.batch_size = 1
        self.num_epochs = 5
        self.num_early_stop = 0
        self.gnn_latent_dim = [1024, 1024, 1024]
        self.basis_dim = 128
        self.num_basis_per_class = 10
        self.gnn_dropout = 0.5
        self.add_self_loop = True
        self.gcn_adj_normalization = True
        self.gnn_emb_normalization = False
        self.graph_classification = False
        self.node_classification = True
        self.gnn_nonlinear = 'relu'
        self.readout = 'identity'
        self.fc_latent_dim = []
        self.fc_dropout = 0.5
        self.fc_nonlinear = 'relu'
        
        if modelCfg != None:
            self.learning_rate1 = modelCfg['learning_rate1']
            self.learning_rate2 = modelCfg['learning_rate2']
            self.weight_decay1 = modelCfg['weight_decay1']
            self.weight_decay2 = modelCfg['weight_decay2']
            self.milestones = modelCfg['milestones']
            self.gamma = modelCfg['gamma']
            self.batch_size = modelCfg['batch_size']
            self.num_epochs = modelCfg['num_epochs']
            self.num_early_stop = modelCfg['num_early_stop']
            self.gnn_latent_dim = modelCfg['gnn_latent_dim']
            self.basis_dim = modelCfg['basis_dim']
            self.num_basis_per_class = modelCfg['num_basis_per_class']
            self.gnn_dropout = modelCfg['gnn_dropout']
            self.add_self_loop = modelCfg['add_self_loop']
            self.gcn_adj_normalization = modelCfg['gcn_adj_normalization']
            self.gnn_emb_normalization = modelCfg['gnn_emb_normalization']
            self.graph_classification = modelCfg['graph_classification']
            self.node_classification = modelCfg['node_classification']
            self.gnn_nonlinear = modelCfg['gnn_nonlinear']
            self.readout = modelCfg['readout']
            self.fc_latent_dim = modelCfg['fc_latent_dim']
            self.fc_dropout = modelCfg['fc_dropout']
            self.fc_nonlinear = modelCfg['fc_nonlinear']
        
class dataParams():
    def __init__(self, dataCfg = None):
        self.random_split_flag = True
        self.data_split_ratio = [0.8, 0.1, 0.1]
        if dataCfg != None:
            self.random_split_flag = dataCfg['random_split_flag']
            self.data_split_ratio = dataCfg['data_split_ratio']