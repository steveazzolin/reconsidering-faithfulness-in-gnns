r"""
The pooling classes for the use of the GNNs.
"""
import torch.nn as nn
from torch import Tensor
import torch_geometric.nn as gnn

from torch_scatter import scatter_mean

class GNNPool(nn.Module):
    r"""
    Base pooling class.
    """
    def __init__(self):
        super().__init__()


class GlobalMeanPool(GNNPool):
    r"""
    Global mean pooling
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.mitigation_readout = kwargs["mitigation_readout"] if "mitigation_readout" in kwargs.keys() else None
        print("mitigation_readout = ", self.mitigation_readout)

    def forward(self, x, batch, batch_size=None, edge_index=None, edge_mask=None):
        r"""Returns batch-wise graph-level-outputs by averaging node features
            across the node dimension, so that for a single graph
            :math:`\mathcal{G}_i` its output is computed by

            .. math::
                \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

            Args:
                x (Tensor): Node feature matrix
                    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
                batch (Tensor): Batch vector
                    :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
                    node to a specific example.
                batch_size (int): Batch size.


            Returns (Tensor):
                batch-wise graph-level-outputs by averaging node features across the node dimension.

        """
        if batch_size is None:
            batch_size = batch[-1].item() + 1
        if self.mitigation_readout == "weighted":
            node_mask = scatter_mean(edge_mask, edge_index[0], dim_size=x.shape[0])
            x = x * node_mask.unsqueeze(1)
        return gnn.global_mean_pool(x, batch, batch_size)


class GlobalAddPool(GNNPool):
    r"""
    Global add pooling
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, batch, batch_size=None, edge_index=None, edge_mask=None):
        r"""Returns batch-wise graph-level-outputs by adding node features
            across the node dimension, so that for a single graph
            :math:`\mathcal{G}_i` its output is computed by

            .. math::
                \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n

            Args:
                x (Tensor): Node feature matrix
                    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
                batch (Tensor): Batch vector
                    :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
                    node to a specific example.
                batch_size (int): Batch size.

            Returns (Tensor):
                batch-wise graph-level-outputs by adding node features across the node dimension.
        """
        if batch_size is None:
            batch_size = batch[-1].item() + 1
        if not edge_mask is None and not edge_index is None:
            node_mask = scatter_mean(edge_mask, edge_index[0], dim_size=x.shape[0])
            x = x * node_mask.unsqueeze(1)
        return gnn.global_add_pool(x, batch, batch_size)


class GlobalMaxPool(GNNPool):
    r"""
    Global max pooling
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, batch, batch_size=None):
        r"""Returns batch-wise graph-level-outputs by taking the channel-wise
            maximum across the node dimension, so that for a single graph
            :math:`\mathcal{G}_i` its output is computed by

            .. math::
                \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

            Args:
                x (Tensor): Node feature matrix
                    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
                batch (Tensor): Batch vector
                    :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
                    node to a specific example.
                batch_size (int): Batch size.

            Returns (Tensor):
                   batch-wise graph-level-outputs by taking the channel-wise maximum across the node dimension.

        """
        if batch_size is None:
            batch_size = batch[-1].item() + 1
        return gnn.global_max_pool(x, batch, batch_size)


class IdenticalPool(GNNPool):
    r"""
    Identical pooling
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, batch, batch_size=None):
        r"""Returns batch-wise graph-level-outputs by taking the node features identically.

            Args:
                x (Tensor): Node feature matrix
                batch (Tensor): Batch vector
                batch_size (int): Batch size.

            Returns (Tensor):
                   batch-wise graph-level-outputs by taking the node features identically.

        """
        return x
