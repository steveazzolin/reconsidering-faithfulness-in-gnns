import torch
import numpy as np
from sklearn import metrics
import os
import global_config as gl
from torch_geometric.utils import mask_to_index, index_to_mask
from copy import deepcopy

from models.GISST.sig.utils.optimization_utils import init_model, run_model, batch_train, batch_evaluate
from torch_geometric.loader.dataloader import DataLoader

from models.GISST.sig.nn.models.sigcn import SIGCN
from models.GISST.sig.nn.loss.regularization_loss import reg_sig_loss
from models.GISST.sig.nn.loss.classification_loss import cross_entropy_loss
from models.GISST.sig.explainers.sig_explainer import SIGExplainer

class Gisst():
    def __init__(self, dataset, datasetName, deviceToUse=None, config = None, _run = None, dict_dataset=None):
        if deviceToUse == None:
            self.device = "cuda"
            #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"
            
        self.config = config
        if config == None: #Use a default config if none was set
            self.config = GisstConfig()
            
        #Save input values
        self.datasetName = datasetName
        self.dataset = dataset

        self.num_class = dataset.num_classes
        #Everything above is required
        
        #Metric logging info
        #self._run = _run #Access to logger
        #self.train_counter = 0 
        
        self.coeffs={
            'x_l1': self.config.x_l1_coeffs,
            'x_ent': self.config.x_ent_coeffs,
            'edge_l1': self.config.edge_l1_coeffs,
            'edge_ent': self.config.edge_ent_coeffs
        }
        
        self.fn_train = batch_train
        self.fn_eval = batch_evaluate
        if not dict_dataset is None:
            assert self.datasetName == "GOODMotif2" or self.datasetName == "GOODMotif_size"
            self.kwargs_train = {
                'loader': DataLoader(
                    dict_dataset["train"],
                    self.config.batch_size,
                    shuffle=True, 
                    num_workers=0
                ),
                'device': self.device
            }
            self.kwargs_val = {
                'loader': DataLoader(
                    dict_dataset["val"],
                    self.config.batch_size,
                    shuffle=False,
                    num_workers=0
                ),
                'device': self.device
            }
            self.kwargs_test = {
                'loader': DataLoader(
                    dict_dataset["test"],
                    self.config.batch_size,
                    shuffle=False,
                    num_workers=0
                ),
                'device': self.device
            }
        else: 
            self.kwargs_train = {
                'loader': DataLoader(
                    self.dataset[self.dataset.index_train],
                    self.config.batch_size,
                    shuffle=True, 
                    num_workers=0
                ),
                'device': self.device
            }
            self.kwargs_val = {
                'loader': DataLoader(
                    self.dataset[self.dataset.index_val],
                    self.config.batch_size,
                    shuffle=False,
                    num_workers=0
                ),
                'device': self.device
            }
            self.kwargs_test = {
                'loader': DataLoader(
                    self.dataset[self.dataset.index_test],
                    self.config.batch_size,
                    shuffle=False,
                    num_workers=0
                ),
                'device': self.device
            }
        self.model = SIGCN(
            input_size=self.dataset.num_features,
            output_size=self.num_class,
            hidden_conv_sizes=(self.config.hidden_dims, ) * self.config.num_hidden_layers, 
            hidden_dropout_probs=(self.config.dropout_rates, ) * self.config.num_hidden_layers,
            classify_graph=True,
            lin_dropout_prob=self.config.dropout_rates,
            architecture = self.config.architecture,
            mitigation=self.config.mitigation
        ).to(self.device)
        init_model(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        

        
        self.explainer = SIGExplainer(self.model)
        self.explainer.to(self.device)
                   
    def train(self):
        run_model(
            model=self.model, 
            optimizer=self.optimizer,
            sig_coeffs= self.coeffs,
            num_epochs= self.config.epochs, 
            kwargs_train=self.kwargs_train,
            kwargs_eval = self.kwargs_val,
            kwargs_test = self.kwargs_test,
            fn_train=self.fn_train,
            fn_eval=self.fn_eval,
            training=True,
        )
    
    def test(self):
        result = self.fn_eval(
            model=self.model,
            sig_coeffs=self.coeffs,
            **self.kwargs_test,
            training=False,
        )
        
        acc = result['accuracy']
        
        return acc
    
    def save(self):
        torch.save(self.model.state_dict(),os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName))
    def my_save(self,path):
        torch.save(self.model.state_dict(),path)
        
    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName)))
    def my_load(self,path):
        self.model.load_state_dict(torch.load(path))


    def get_explanation(self, data):
        #data = self.dataset[graph_index].to(self.device)
        data = data.to(self.device)
        
        node_feat_prob, edge_prob = self.explainer.explain_graph(
            x=data.x,
            edge_index=data.edge_index,
            batch = torch.zeros(data.x.size()[0], device=self.device, dtype=int),
            use_grad=False,
            y=None,
            loss_fn=None,
            take_abs=False,
            pred_for_grad=False
        )
        
        node_feat_prob = node_feat_prob.reshape((1, len(node_feat_prob)))
        
        return node_feat_prob.cpu(), edge_prob.cpu()
        
    def get_prediction_vector(self, data):  
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            out, x_prob, edge_prob = self.model(
                data.x, 
                data.edge_index,
                return_probs=True,
                batch=torch.zeros(data.x.size()[0], device=self.device, dtype=int),
            )
            
        return out
    
class GisstConfig():
    def __init__(self, config_dict = None):

        self.modelname = 'GISST_gc'
        self.num_hidden_layers = 3
        self.hidden_dims = 32
        self.lr = 0.001
        self.epochs = 1000
        self.batch_size = 32
        self.weight_decay = 0.0005 #l2_coeff in their code
        self.dropout_rates = 0.1
        self.x_l1_coeffs = 0.005
        self.x_ent_coeffs = 0.01
        self.edge_l1_coeffs = 0.005
        self.edge_ent_coeffs = 0.1
        self.verbose = True
        
        if config_dict != None:
            self.modelname = config_dict['modelname']
            self.num_hidden_layers = config_dict['num_hidden_layers']
            self.hidden_dims = config_dict['hidden_dims']
            self.lr = config_dict['lr']
            self.epochs = config_dict['epochs']
            self.batch_size = config_dict['batch_size']
            self.weight_decay = config_dict['weight_decay'] #l2_coeff in their code
            self.dropout_rates = config_dict['dropout_rates']
            self.x_l1_coeffs = config_dict['x_l1_coeffs']
            self.x_ent_coeffs = config_dict['x_ent_coeffs']
            self.edge_l1_coeffs = config_dict['edge_l1_coeffs']
            self.edge_ent_coeffs = config_dict['edge_ent_coeffs']
            self.verbose = config_dict['verbose']
            self.architecture = config_dict["architecture"]
            self.mitigation = config_dict["mitigation"]