import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from ....sig.nn.models.base_model import BaseClassifier
from torch_geometric.nn import GCNConv,GINEConv, global_mean_pool,MLP
from torch_geometric.data import Data
from ....sig.nn.modules.attention import AttentionProb
from ....sig.nn.modules.mask import ProbMask
from torch_geometric.utils import to_undirected, coalesce
from ourutils.splitting import relabel, split_graph
from models.GINmod.GINs import GINConvAttn
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric import __version__ as pyg_v
from ourutils.pooling import GlobalMeanPool


class SIGCN(BaseClassifier):
    """
    Sparse Interpretable Graph Convolution Network.

    Args:
        input_size (int): Number of input node features.
        output_size (int): Number of output node features.
        hidden_conv_sizes (tuple of int): Output sizes for hidden convolution layers.
        hidden_dropout_probs (tuple of float): Dropout probabilities after the hidden
            convolution layers.
        activation (torch.nn.functional): Non-linear activation function after the hidden
            convolution layers.  
        classify_graph (boolean): Whether the model is a graph classifier. Default False
            for node classifier.
        lin_dropout_prob (None or float): Dropout probability after the hidden linear 
            layer for graph classifier. None for node classifier.
        **kwargs: Additoinal kwargs for GCNConv.
    """
    def __init__(
        self, 
        input_size,
        output_size,
        hidden_conv_sizes, 
        hidden_dropout_probs, 
        activation=F.relu,
        classify_graph=False,
        lin_dropout_prob=None,
        architecture = None,
        mitigation = None,
        **kwargs
    ):
        super().__init__(
            input_size,
            output_size,
            hidden_conv_sizes, 
            hidden_dropout_probs, 
            activation,
            classify_graph,
            lin_dropout_prob,
            architecture
        )
        self.architecture = architecture
        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for i in range(self.num_conv_layers):

            if self.architecture == "GINe":
                mpl = MLP(in_channels=self.conv_input_sizes[i],
                            hidden_channels = 300,
                            out_channels=self.conv_output_sizes[i],
                            num_layers=2)
                self.convs.append(
                    GINEConv(mpl,edge_dim=1)
                )
            elif self.architecture == "GCN":
                self.convs.append(
                   GCNConv(
                       self.conv_input_sizes[i], 
                       self.conv_output_sizes[i],
                       **kwargs
                   )
                )
            elif self.architecture == "GINmod":
                dim_hidden = self.conv_input_sizes[i]
                out_dim = self.conv_output_sizes[i]
                self.convs.append(
                    GINConvAttn(nn.Sequential(nn.Linear(dim_hidden, 2 * dim_hidden),
                                nn.BatchNorm1d(2 * dim_hidden, track_running_stats=True), nn.ReLU(),
                                nn.Linear(2 * dim_hidden, out_dim)),
                                emb_dim= out_dim)
                )
            self.dropouts.append(
                Dropout(self.dropout_probs[i])
            )
        if mitigation == "p2" or mitigation == "p2HM":
            print(mitigation)
            self.pooling = GlobalMeanPool()
        self.prob_mask = ProbMask(self.input_size)
        self.attention_prob = AttentionProb(self.input_size)
        self.mitigation = mitigation # added da Anon vale solo per p2
    def forward(
        self, 
        x, 
        edge_index,
        edge_weight=None,
        return_probs=False,
        batch=None,
        weight=None,
        is_ratio=None,
        training = None,
    ):
        """
        Forward pass.

        Args:
            x (torch.float): Node feature tensor with shape [num_nodes, num_node_feat].
            edge_index (torch.long): Edges in COO format with shape [2, num_edges].
            edge_weight (torch.float): Weight for each edge with shape [num_edges].
            return_probs (boolean): Whether to return x_prob and edge_prob.
            batch (None or torch.long): Node assignment for a batch of graphs with shape 
                [num_nodes] for graph classification. None for node classification.

        Return:
            x (torch.float): Final output of the network with shape 
                [num_nodes, output_size].
            x_prob (torch.float): Node feature probability with shape [input_size].
            edge_prob (torch.float): Edge probability with shape [num_edges].
        """
        x_prob = self.prob_mask()
        x_prob.requires_grad_()
        x_prob.retain_grad()
        x = x * x_prob

        edge_prob = self.attention_prob(x, edge_index)
        edge_prob.requires_grad_()
        edge_prob.retain_grad()

        # added by Anon 
        # per aver eedg prob simm.
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).to(edge_prob.device)
        else:
            edge_index_sorted, edge_weight = coalesce(edge_index, edge_weight, is_sorted=False)
        edge_index, edge_prob = to_undirected(edge_index, edge_prob, reduce="mean")


        if self.mitigation == "HM":
            edge_prob = self.sampling(edge_prob,training)
        elif self.mitigation == "p2HM":
            edge_prob = self.sampling(edge_prob,training)


        #if is_ratio, apply topK

        edge_weight = edge_weight * edge_prob

        if weight:
            if is_ratio:
                data = Data(x=x, edge_index=edge_index, batch=batch, edge_attr=edge_weight)
                (causal_edge_index, causal_edge_attr, causal_edge_weight), _ = split_graph(data, edge_prob, weight)
                causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, batch)
                x = causal_x
                batch = causal_batch
                edge_index = causal_edge_index
                edge_weight = causal_edge_attr  


        # fine added by Anon 
               
        # data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")


        #set maskera 
        mask = edge_weight

        if self.architecture == "GINmod":
            for conv in self.convs:
                if isinstance(conv,MessagePassing):
                    conv._apply_sigmoid = False
                    if pyg_v >= "2.4.0":
                        conv._fixed_explain = True
                    else:
                        conv.__explain__ = True
                        conv._explain = True    
                    conv.__edge_mask__ = mask
                    conv._edge_mask = mask

        for i in range(self.num_conv_layers):
            #print("START conv",i)
            #RICORDATI CHE L'UNIQUIZE SERVE SOLO PER LA GIN
            if self.architecture == "GINe" or self.architecture=="GINmod":
                x = self.convs[i](x, edge_index, edge_weight.unsqueeze(1))
            elif self.architecture == "GCN":
                x = self.convs[i](x, edge_index, edge_weight)
            if self.classify_graph or i < self.num_conv_layers - 1:
                # no activation for output conv layer in node classifier
                x = self.activation(x)
            x = self.dropouts[i](x)                  


        if self.classify_graph:
            if self.mitigation == "p2":
                #x = self.pooling(x, batch, edge_index=edge_index,edge_mask=edge_prob)
                x = self.pooling(x, batch, edge_index=edge_index,edge_mask=mask)
            if self.mitigation == "p2HM":
                x = self.pooling(x, batch, edge_index=edge_index,edge_mask=mask)
            else:
                x = global_mean_pool(x, batch)

            x = self.activation(self.lin1(x))
            x = self.lin_dropout(x)
            x = self.lin2(x)


        # togli maschera
        if self.architecture == "GINmod":
            for conv in self.convs:
                if isinstance(conv, MessagePassing):
                    if pyg_v >= "2.4.0":
                        conv._fixed_explain = False
                    else:
                        conv.__explain__ = False
                        conv._explain = False
                    conv.__edge_mask__ = None
                    conv._edge_mask = None

        if return_probs:
            return x, x_prob, edge_prob
        else:
            return x
        
    def sampling(self, att_log_logits, training, mitigation_expl_scores="hard"):

        temp = 1
        att = self.concrete_sample(att_log_logits, temp=temp, training=training)
        if mitigation_expl_scores == "hard":
            att_hard = torch.where(att > 0.5, 0.99999, 0.00001)
            att = att_hard
            #att_hard = (att > 0.5).float()
            #att = att_hard - att.detach() + att
        return att
    
    
    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @torch.no_grad()
    def probs(self, *args, **kwargs):
        # nodes x classes
        lc_logits = self(*args, **kwargs)
        if lc_logits.shape[-1] > 1:
            return lc_logits.softmax(dim=1)
        else:
            return lc_logits.sigmoid()
    
    @torch.no_grad()
    def log_probs(self, eval_kl=False, *args, **kwargs):
        # nodes x classes

        lc_logits = self(*args, **kwargs)

        if lc_logits.shape[-1] > 1:
            return lc_logits.log_softmax(dim=1)           

        else:
            if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                lc_logits = lc_logits.sigmoid()
                new_logits = torch.zeros((lc_logits.shape[0], lc_logits.shape[1]+1), device=lc_logits.device)
                new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                new_logits[:, 0] = 1 - new_logits[:, 1]
                new_logits[new_logits == 0.] = 1e-10
                return new_logits.log()
            else:
                return lc_logits.sigmoid().log()