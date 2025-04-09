r"""
The Graph Neural Network from the `"How Powerful are Graph Neural Networks?"
<https://arxiv.org/abs/1810.00826>`_ paper.
"""
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

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .MolEncoders import AtomEncoder, BondEncoder
from torch.nn import Identity

import torch.nn.functional as F


@register.model_register
class GIN(GNNBasic):
    r"""
    The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):

        super().__init__(config)
        self.feat_encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class GINFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINFeatExtractor, self).__init__(config)
        print("#D#Init GINFeatExtractor")
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = GINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
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

    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):

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
        if hasattr(self.convs[0], "__edge_mask__"):
            out_readout = self.readout(node_repr, batch, batch_size, edge_index=edge_index, edge_mask=self.convs[0].__edge_mask__)
        else:
            out_readout = self.readout(node_repr, batch, batch_size)
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
        
        self._fixed_explain = False
        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.attn_distrib = []


    def forward(self, x, edge_index, return_attn_distrib=False):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, return_attn_distrib=return_attn_distrib))
        return out

    def message(self, x_i, x_j):
        if self._fixed_explain:
            edge_mask = self.__edge_mask__
            if self._apply_sigmoid:
                edge_mask = edge_mask.sigmoid()
            x_j = x_j * edge_mask.view([-1] + [1] * (x_j.dim() - 1))
        return x_j

    def update(self, aggr_out):
        return aggr_out




class GINMolEncoder(BasicEncoder):
    r"""The GIN encoder for molecule data, using the :class:`~GINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINMolEncoder, self).__init__(config, **kwargs)
        self.without_readout = kwargs.get('without_readout')
        num_layer = config.model.model_layer
        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)

        self.convs = nn.ModuleList(
            [
                GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                       nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)), config)
            ] +
            [
                GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                       nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)), config)
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            graph feature representations
        """
        node_repr = self.get_node_repr(x, edge_index, edge_attr, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        if hasattr(self.convs[0], "__edge_mask__"):
            out_readout = self.readout(node_repr, batch, batch_size, edge_index=edge_index, edge_mask=self.convs[0].__edge_mask__)
        else:
            out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """

        layer_feat = [self.atom_encoder(x)]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            tmp = conv(layer_feat[-1], edge_index, edge_attr)
            post_conv = batch_norm(tmp)
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
        return layer_feat[-1]


class GINEConv(gnn.MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, config, eps: float = 0., train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if hasattr(self.nn[0], 'in_features'):
            in_channels = self.nn[0].in_features
        else:
            in_channels = self.nn[0].in_channels
        self.bone_encoder = BondEncoder(in_channels, config)
        # if edge_dim is not None:
        #     self.lin = Linear(edge_dim, in_channels)
        #     # self.lin = Linear(edge_dim, config.model.dim_hidden)
        # else:
        #     self.lin = None

        self._fixed_explain = False

        self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, return_attn_distrib:bool = False) -> Tensor:

        if self.bone_encoder and edge_attr is not None:
            edge_attr = self.bone_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, return_attn_distrib=return_attn_distrib)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, return_attn_distrib: bool = False) -> Tensor:
        if edge_attr is not None:
            if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionalities do not "
                                 "match. Consider setting the 'edge_dim' "
                                 "attribute of 'GINEConv'")

            if self.lin is not None:
                edge_attr = self.lin(edge_attr)

            m = x_j + edge_attr
        else:
            m = x_j

        if self._fixed_explain:
            edge_mask = self.__edge_mask__
            if self._apply_sigmoid:
                edge_mask = edge_mask.sigmoid()
            m = m * edge_mask.view([-1] + [1] * (m.dim() - 1))

        return m.relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'