
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor

from .BaseGNN import GNNBasic, BasicEncoder
from torch.nn import Identity

import torch.nn.functional as F


class GINFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config, **kwargs):
        super(GINFeatExtractor, self).__init__(config)
        print("#D#Init GINFeatExtractor")
        num_layer = config.model.model_layer

        self.encoder = GINEncoder(config, **kwargs)
        self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout

    def get_node_repr(self, *args, **kwargs):

        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            node_repr = self.encoder.get_node_repr(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            node_repr = self.encoder.get_node_repr(x, edge_index, batch, batch_size, **kwargs)
        return node_repr


class GINEncoder(BasicEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config, *args, **kwargs):

        super(GINEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model.model_layer
        self.without_readout = kwargs.get('without_readout')

        # self.atom_encoder = AtomEncoder(config.model.dim_hidden)
        self.convs = nn.ModuleList()
        if kwargs.get('without_embed'):
            self.convs.append(GINConvAttn(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden, track_running_stats=True), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)),
                                               emb_dim=config.model.dim_hidden))
        else:
            self.convs.append(GINConvAttn(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden, track_running_stats=True), nn.ReLU(),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)),
                                            emb_dim=config.dataset.dim_node))

        self.convs = self.convs.extend(
            [
                GINConvAttn(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                    nn.BatchNorm1d(2 * config.model.dim_hidden, track_running_stats=True), nn.ReLU(),
                                    nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)),
                                    emb_dim=config.model.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )

    def get_attn_distrib(self):
        ret = []
        for conv in self.convs:
            ret.append(conv.attn_distrib)
        return ret

    def reset_attn_distrib(self):
        for conv in self.convs:
            conv.attn_distrib = []

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            graph feature representations
        """

        node_repr = self.get_node_repr(x, edge_index, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        out_readout = self.readout(node_repr, batch, batch_size, edge_index=edge_index, edge_mask=self.convs[0].__edge_mask__)
        return out_readout

    def get_node_repr(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        layer_feat = [x]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(layer_feat[-1], edge_index, return_attn_distrib=kwargs.get('return_attn', False)))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
        return layer_feat[-1]
    

class GINConvAttn(gnn.MessagePassing):
    def __init__(self, mlp, emb_dim):
        super(GINConvAttn, self).__init__(aggr="add")
        
        if torch_geometric.__version__ >= "2.4.0":
            print("#D#Using the fixed _explain_ functionality")
            self._fixed_explain = False

        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.attn_distrib = []

    def forward(self, x, edge_index, return_attn_distrib=False):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, return_attn_distrib=return_attn_distrib))
        return out

    def message(self, x_i, x_j, return_attn_distrib):           
        if self._fixed_explain:
            edge_mask = self.__edge_mask__
            if self._apply_sigmoid:
                edge_mask = edge_mask.sigmoid()
            x_j = x_j * edge_mask.view([-1] + [1] * (x_j.dim() - 1))

        return x_j

    def update(self, aggr_out):
        return aggr_out

