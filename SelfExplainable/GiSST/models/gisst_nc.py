import torch
import numpy as np
from sklearn import metrics
import os
import global_config as gl
from torch_geometric.utils import mask_to_index, index_to_mask
from copy import deepcopy

from models.GISST.sig.nn.models.sigcn import SIGCN
from models.GISST.sig.utils.optimization_utils import init_model
from models.GISST.sig.nn.loss.regularization_loss import reg_sig_loss
from models.GISST.sig.nn.loss.classification_loss import cross_entropy_loss
from models.GISST.sig.explainers.sig_explainer import SIGExplainer

"""
Wrapper for the GISST model that allows for training, testing, save, load and explanations extraction
This is a good example of a "simple" integration of a model such that it could be used for testing.
"""

class Gisst():
    def __init__(self, dataset, datasetName, deviceToUse=None, config = None, _run = None):
        if deviceToUse == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = "cpu"
            
        self.config = config
        if config == None: #Use a default config if none was set
            self.config = GisstConfig()
            
        #Save input values
        self.datasetName = datasetName
        self.dataset = dataset                                              #Required property
        self.data = dataset[0].to(self.device) #Node classification only... (required for NC)
        self.num_class = int(torch.max(self.data.y) + 1)
        
        #Metric logging info
        self._run = _run #Access to logger
        self.train_counter = 0 
        
        self.model = SIGCN(                                                 #Required property
            input_size=self.data.x.size(1),
            output_size=self.num_class,
            hidden_conv_sizes=(self.config.hidden_dims, ) * self.config.num_hidden_layers, 
            hidden_dropout_probs=(self.config.dropout_rates, ) * self.config.num_hidden_layers,
            lin_dropout_prob=self.config.dropout_rates
        ).to(self.device)
        
        init_model(self.model)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        
        self.explainer = SIGExplainer(self.model)
        self.explainer.to(self.device)
        
    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()

        out, x_prob, edge_prob = self.model(
            self.data.x, 
            self.data.edge_index, 
            return_probs=True
        )
        loss_x_l1, \
        loss_x_ent, \
        loss_edge_l1, \
        loss_edge_ent = reg_sig_loss(
            x_prob, 
            edge_prob, 
            coeffs={
                'x_l1': self.config.x_l1_coeffs,
                'x_ent': self.config.x_ent_coeffs,
                'edge_l1': self.config.edge_l1_coeffs,
                'edge_ent': self.config.edge_ent_coeffs
            }
        )
    
        loss = cross_entropy_loss(out[self.data.train_mask], self.data.y[self.data.train_mask]) + loss_x_l1 + loss_x_ent + loss_edge_l1 + loss_edge_ent
        
        train_mask = self.data.train_mask.cpu() 
        pred = out.detach().max(1)[1]
        pred = pred.cpu().numpy()
        y = self.data.y.detach().cpu().numpy()
        acc_val = metrics.accuracy_score(y[train_mask], pred[train_mask])
        
        if acc_val >= self.best_acc_val:
            self.best_acc_val = acc_val
            self.best_train_weights = deepcopy(self.model.state_dict())
        
        if(self._run !=None):             
            self._run.log_scalar("training.loss", float(loss.detach().cpu()), self.train_counter)
            self._run.log_scalar("training.accuracy", float(acc_val), self.train_counter)
            self.train_counter += 1
        
        loss.backward(retain_graph=True)
        self.optimizer.step()   
            
    def train(self):
        self.best_acc_val = 0
        for epoch in range(1, self.config.epochs):
            self._train()
            if epoch % 10 == 0 and self.config.verbose: 
                print('Finished training for %d epochs' % epoch)
                
        self.model.load_state_dict(self.best_train_weights)
    
    def test(self):
        print("Trying to test GISST")
        data = self.data
        test_mask = data.test_mask.cpu()
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                data.x, 
                data.edge_index, 
                return_probs=False
            )

        pred = out.max(1)[1]
        pred = pred.detach().cpu().numpy()
        y = self.data.y.detach().cpu().numpy()

        acc = metrics.accuracy_score(y[test_mask], pred[test_mask])
        
        return acc
    
    def save(self):
        torch.save(self.model.state_dict(),os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName))
        
    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(gl.GlobalConfig.checkpoint_dir, self.config.modelname, self.datasetName)))

    def get_explanation(self, node_index, manual_hops=1):
        """Generate explanation for a specific node index

        Args:
            node_index int: Node to generate explanation for
            manual_hops (int, optional): Not used in GISST. Defaults to 1.

        Returns:
            Node feature probs on the CPU and edge probabilities on the CPU. These have to have the same shape as a mask such that they can be converted to a mask.
        """
        node_feat_prob, edge_prob = self.explainer.explain_node(
            node_index=node_index,
            x=self.data.x,
            edge_index=self.data.edge_index,
            use_grad=False,
            y=None,
            loss_fn=None,
            take_abs=False,
            pred_for_grad=False
        )
        
        node_feat_prob = node_feat_prob.reshape((1, len(node_feat_prob)))
        
        return node_feat_prob.cpu(), edge_prob.cpu()
        
    def get_prediction_vector(self, nodes, data=None):
        """Fetch the prediction vectors for a list of nodes provided

        Args:
            nodes (_type_): List of nodes
            data (_type_, optional): Option to specifiy specific data object to use instead of the bundled one. Defaults to None.

        Returns:
            _type_: _description_
        """
        if data == None:
            data = self.data #If no data provided 
            
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                data.x, 
                data.edge_index, 
                return_probs=False
            )
            
        return out[nodes]
    
class GisstConfig():
    def __init__(self, config_dict = None):

        self.modelname = 'GISST'
        self.num_hidden_layers = 2
        self.hidden_dims = 16
        self.lr = 0.001
        self.epochs = 1000
        self.weight_decay = 0.0005 #l2_coeff in their code
        self.dropout_rates = 0.0
        self.x_l1_coeffs = 0.0005
        self.x_ent_coeffs = 0.001
        self.edge_l1_coeffs = 0.005
        self.edge_ent_coeffs = 0.01
        self.verbose = True
        
        if config_dict != None:
            self.modelname = config_dict['modelname']
            self.num_hidden_layers = config_dict['num_hidden_layers']
            self.hidden_dims = config_dict['hidden_dims']
            self.lr = config_dict['lr']
            self.epochs = config_dict['epochs']
            self.weight_decay = config_dict['weight_decay'] #l2_coeff in their code
            self.dropout_rates = config_dict['dropout_rates']
            self.x_l1_coeffs = config_dict['x_l1_coeffs']
            self.x_ent_coeffs = config_dict['x_ent_coeffs']
            self.edge_l1_coeffs = config_dict['edge_l1_coeffs']
            self.edge_ent_coeffs = config_dict['edge_ent_coeffs']
            self.verbose = config_dict['verbose']
        