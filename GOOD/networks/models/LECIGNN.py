r"""
`Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization <https://arxiv.org/abs/2306.01103>`_.
"""
import munch
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from torch_geometric import __version__ as pyg_v
from torch_geometric.data import Data
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from .Pooling import GlobalMeanPool
from munch import munchify
from .MolEncoders import AtomEncoder, BondEncoder
from GOOD.utils.fast_pytorch_kmeans import KMeans
from GOOD.utils.splitting import split_graph, relabel

import copy

@register.model_register
class LECIGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LECIGIN, self).__init__(config)

        # --- if environment inference ---
        config.environment_inference = False
        if config.environment_inference:
            self.env_infer_warning = f'#W#Expermental mode: environment inference phase.'
            config.dataset.num_envs = 3
        # --- Test environment inference ---

        self.config = copy.deepcopy(config)

        self.learn_edge_att = True
        self.LA = self.config.ood.extra_param[0]
        self.EC = self.config.ood.extra_param[1]      # Never used
        self.EA = self.config.ood.extra_param[2]
        self.EF = self.config.ood.extra_param[4]

        fe_kwargs = {'without_embed': True if self.EF else False}

        # --- Build networks ---
        self.sub_gnn = GINFeatExtractor(self.config, **fe_kwargs)
        fe_kwargs['mitigation_readout'] = config.mitigation_readout
        print(f"Using feature sampling := {self.config.mitigation_sampling}")
        print(f"self.EF = {self.EF}")

        self.extractor = ExtractorMLP(self.config)

        self.ef_mlp = EFMLP(self.config, bn=True)
        self.ef_discr_mlp = MLP([self.config.model.dim_hidden, 2 * self.config.model.dim_hidden, self.config.model.dim_hidden],
                                 dropout=self.config.model.dropout_rate, config=self.config, bn=True)
        self.ef_pool = GlobalMeanPool()
        self.ef_classifier = Classifier(munchify({'model': {'dim_hidden': self.config.model.dim_hidden},
                                                   'dataset': {'num_classes': self.config.dataset.num_envs}}))

        self.lc_gnn = GINFeatExtractor(self.config, **fe_kwargs)
        self.la_gnn = GINFeatExtractor(self.config, **fe_kwargs)
        self.ec_gnn = GINFeatExtractor(self.config, **fe_kwargs)    # Never used
        self.ea_gnn = GINFeatExtractor(self.config, **fe_kwargs)

        self.lc_classifier = Classifier(self.config)
        self.la_classifier = Classifier(self.config)
        self.ec_classifier = Classifier(munchify({'model': {'dim_hidden': self.config.model.dim_hidden},
                                                   'dataset': {'num_classes': self.config.dataset.num_envs}})) # Never used
        self.ea_classifier = Classifier(munchify({'model': {'dim_hidden': self.config.model.dim_hidden},
                                                  'dataset': {'num_classes': self.config.dataset.num_envs}}))

        self.edge_mask = None



    def forward(self, *args, **kwargs):
        r"""
        The LECIGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')

        # --- Filter environment info in features (only features) ---
        if self.EF:
            filtered_features = self.ef_mlp(data.x, data.batch)
            adversarial_features = GradientReverseLayerF.apply(filtered_features, self.EF * self.config.train.alpha)
            ef_logits = self.ef_classifier(self.ef_pool(self.ef_discr_mlp(adversarial_features, data.batch), data.batch))
            data.x = filtered_features
            kwargs['data'] = data
        else:
            ef_logits = None

        node_repr = self.sub_gnn.get_node_repr(*args, **kwargs)
        att_log_logits = self.extractor(node_repr, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, self.training, self.config.mitigation_expl_scores)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                if self.config.average_edge_attn == "default":
                    edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
                else:
                    if not data.edge_attr is None:
                        edge_index_sorted, data.edge_attr = coalesce(data.edge_index, data.edge_attr, is_sorted=False)                    
                    data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        if kwargs.get('weight', None):
            if kwargs.get('is_ratio'):
                (causal_edge_index, causal_edge_attr, causal_edge_weight), _ = split_graph(data, edge_att, kwargs.get('weight'))
                causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)
                data.x = causal_x
                data.batch = causal_batch
                data.edge_index = causal_edge_index
                if not data.edge_attr is None:
                    data.edge_attr = causal_edge_attr
                edge_att = causal_edge_weight

        if self.config.mitigation_expl_scores == "topK" or self.config.mitigation_expl_scores == "topk":
            (causal_edge_index, causal_edge_attr, edge_att), \
                _ = split_graph(data, edge_att, self.config.mitigation_expl_scores_topk)
           
            causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)

            data_topk = Data(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr, batch=causal_batch)
            kwargs['data'] = data_topk
            kwargs["batch_size"] =  data.batch[-1].item() + 1

        set_masks(edge_att, self.lc_gnn)
        lc_logits = self.lc_classifier(self.lc_gnn(*args, **kwargs))
        clear_masks(self)

        if self.LA and self.training:
            set_masks(1 - GradientReverseLayerF.apply(edge_att, self.LA * self.config.train.alpha), self.la_gnn)
            la_logits = self.la_classifier(self.la_gnn(*args, **kwargs))
            clear_masks(self)
        else:
            la_logits = None

        if self.EA and self.training:
            set_masks(GradientReverseLayerF.apply(edge_att, self.EA * self.config.train.alpha), self.ea_gnn)
            ea_readout = self.ea_gnn(*args, **kwargs)
            if self.config.environment_inference:
                if self.env_infer_warning:
                    print(self.env_infer_warning)
                    self.env_infer_warning = None
                kmeans = KMeans(n_clusters=self.config.dataset.num_envs, n_init=10, device=ea_readout.device).fit(ea_readout)
                self.E_infer = kmeans.labels_
            ea_logits = self.ea_classifier(ea_readout)
            clear_masks(self)
        else:
            ea_logits = None

        self.edge_mask = edge_att
        kwargs['data'] = data

        return (lc_logits, la_logits, None, ea_logits, ef_logits), att, edge_att

    def sampling(self, att_log_logits, training, mitigation_expl_scores):
        if (self.config.dataset.dataset_name == 'GOODMotif' and self.config.dataset.domain == 'size') or \
            (self.config.dataset.dataset_name == 'FPIIFMotif'):
            temp = (self.config.train.epoch * 0.1 + (200 - self.config.train.epoch) * 10) / 200
        else:
            temp = 1

        if mitigation_expl_scores == "anneal":
            temp = (self.config.train.epoch * 0.1 + (200 - self.config.train.epoch) * 5) / 200

        att = self.concrete_sample(att_log_logits, temp=temp, training=training)

        if mitigation_expl_scores == "hard":
            att_hard = (att > 0.5).float()
            att = att_hard - att.detach() + att
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

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
        (lc_logits, la_logits, _, ea_logits, ef_logits), att, edge_att = self(*args, **kwargs)
        if lc_logits.shape[-1] > 1:
            return lc_logits.softmax(dim=1)
        else:
            return lc_logits.sigmoid()
    
    @torch.no_grad()
    def log_probs(self, eval_kl=False, *args, **kwargs):
        # nodes x classes
        (lc_logits, la_logits, _, ea_logits, ef_logits), att, edge_att = self(*args, **kwargs)
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
        
    @torch.no_grad()
    def predict_from_subgraph(self, edge_att=False, *args, **kwargs):
        set_masks(edge_att, self.lc_gnn)
        lc_logits = self.lc_classifier(self.lc_gnn(*args, **kwargs))
        clear_masks(self)
        
        if lc_logits.shape[-1] > 1:
            return lc_logits.argmax(-1)
        else:
            return lc_logits.sigmoid()
    
    @torch.no_grad()
    def get_subgraph(self, get_pred=False, log_pred=False, ratio=None, *args, **kwargs):
        data = kwargs.get('data')
        data.ori_x = data.x

        if self.EF:
            filtered_features = self.ef_mlp(data.x, data.batch)
            data.x = filtered_features
            kwargs['data'] = data

        node_repr = self.sub_gnn.get_node_repr(*args, **kwargs)
        att_log_logits = self.extractor(node_repr, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, self.training, self.config.mitigation_expl_scores)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                if self.config.average_edge_attn == "default":
                    edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
                else:
                    # data.ori_edge_index = data.edge_index.detach().clone() #for backup and debug
                    if not data.edge_attr is None:
                        _, data.edge_attr = coalesce(data.edge_index, data.edge_attr, is_sorted=False)
                    if hasattr(data, "edge_gt") and not data.edge_gt is None:
                        _, data.edge_gt = coalesce(data.edge_index, data.edge_gt, is_sorted=False)
                    data.edge_index, edge_att = to_undirected(data.edge_index, att.squeeze(-1), reduce="mean")

                    # for i, (u,v) in enumerate(data.edge_index.T):
                    #     print((u,v), edge_att[i])
                    # exit()
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        if kwargs.get('return_attn', False):
            self.attn_distrib = self.sub_gnn.encoder.get_attn_distrib()
            self.sub_gnn.encoder.reset_attn_distrib()

        edge_att = edge_att.view(-1)
        if not ratio is None:
            assert False
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(data, edge_att, ratio)
            return (causal_edge_index, None, None, causal_edge_weight), \
                    (spu_edge_index, None, None, spu_edge_weight), edge_att
        return edge_att



@register.model_register
class LECIvGIN(LECIGIN):
    r"""
    The GIN virtual node version of LECI.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LECIvGIN, self).__init__(config)
        print("LECIvGIN")
        fe_kwargs = {'without_embed': True if self.EF else False, "mitigation_readout": config.mitigation_readout}
        self.sub_gnn = vGINFeatExtractor(self.config, **fe_kwargs)

        fe_kwargs['mitigation_readout'] = config.mitigation_readout
        self.lc_gnn = vGINFeatExtractor(self.config, **fe_kwargs)
        self.la_gnn = vGINFeatExtractor(self.config, **fe_kwargs)
        self.ec_gnn = vGINFeatExtractor(self.config, **fe_kwargs) # Never used
        self.ea_gnn = vGINFeatExtractor(self.config, **fe_kwargs)


class EFMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch], bn):
        super(EFMLP, self).__init__()
        if config.dataset.dataset_type == 'mol':
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)
            self.mlp = MLP([config.model.dim_hidden, config.model.dim_hidden, 2 * config.model.dim_hidden,
                            config.model.dim_hidden], config.model.dropout_rate, config, bn=bn)
        else:
            self.atom_encoder = nn.Identity()
            self.mlp = MLP([config.dataset.dim_node, config.model.dim_hidden, 2 * config.model.dim_hidden,
                            config.model.dim_hidden], config.model.dropout_rate, config, bn=bn)

    def forward(self, x, batch):
        return self.mlp(self.atom_encoder(x), batch)


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = True
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p,
                                         config=config, bn=config.ood.extra_param[5])
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p,
                                         config=config, bn=config.ood.extra_param[5])

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch=None):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                assert batch is not None
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, config, bias=True, bn=False):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if bn:
                    m.append(nn.BatchNorm1d(channels[i]))
                else:
                    m.append(InstanceNorm(channels[i]))

                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class GradientReverseLayerF(Function):
    r"""
    Gradient reverse layer for DANN algorithm.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        r"""
        gradient forward propagation

        Args:
            ctx (object): object of the GradientReverseLayerF class
            x (Tensor): feature representations
            alpha (float): the GRL learning rate

        Returns (Tensor):
            feature representations

        """
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        gradient backpropagation step

        Args:
            ctx (object): object of the GradientReverseLayerF class
            grad_output (Tensor): raw backpropagation gradient

        Returns (Tensor):
            backpropagation gradient

        """
        output = grad_output.neg() * ctx.alpha
        return output, None


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module._fixed_explain = True             
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._fixed_explain = False
            module.__edge_mask__ = None
            module._edge_mask = None

def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos
