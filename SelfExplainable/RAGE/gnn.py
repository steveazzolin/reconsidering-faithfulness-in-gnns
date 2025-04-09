import os
import argparse
import torch
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, PNAConv, global_max_pool, dense_diff_pool, global_sort_pool, DenseGCNConv, PNA, global_add_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader, DenseDataLoader
import torch.nn.functional as F
import numpy as np
import random
from math import floor, ceil
from tqdm import tqdm
import utils
import time

from typing import Callable, Union

from torch_geometric.nn.inits import reset

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import softmax

import data_utils
from torch_geometric.utils import to_dense_adj


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

    def forward(self, x, batch, batch_size=None, edge_index=None, edge_mask=None):
        if batch_size is None:
            batch_size = batch[-1].item() + 1
        node_mask = scatter_mean(edge_mask, edge_index[0], dim_size=x.shape[0])
        x = x * node_mask.unsqueeze(1)
        return gnn.global_mean_pool(x, batch, batch_size)



class GNN_DiffPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=True):
        super(GNN_DiffPool, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(DenseGCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            out = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](out.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class DiffPool(torch.nn.Module):
    def __init__(self, max_nodes, num_features, dim, num_classes):
        super(DiffPool, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN_DiffPool(num_features, dim, num_nodes)
        self.gnn1_embed = GNN_DiffPool(num_features, dim, dim)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN_DiffPool(dim, dim, num_nodes)
        self.gnn2_embed = GNN_DiffPool(dim, dim, dim)

        self.gnn3_embed = GNN_DiffPool(dim, dim, dim)

        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, num_classes)

    def forward(self, data):
        x = data.x
        adj = data.adj
        mask = data.mask

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return l1 + l2, e1 + e2, x


class GINWrapper(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(GINWrapper, self).__init__()

        mlp = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                  torch.nn.BatchNorm1d(out_channels),
                                  torch.nn.Linear(out_channels, out_channels),
                                  torch.nn.BatchNorm1d(out_channels))
        self.conv = GINWrapperConv(mlp, **kwargs)

    def forward(self, x, edge_index, edge_weight):
        return self.conv(x, edge_index, edge_weight)


class GINWrapperConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class PNAConvWrapper(PNAConv):
    def __init__(self, num_features, dim, aggregators, scalers, deg):
        super().__init__(num_features, dim, aggregators, scalers, deg)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight=None,
                edge_attr: OptTensor = None) -> Tensor:
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

            # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)


class ExpertAttention(torch.nn.Module):
    def __init__(self, dim, expert_number):
        super(ExpertAttention, self).__init__()

        self.expert_number = expert_number
        self.dim = dim

        self.w1 = torch.nn.Linear(dim, dim)
        self.w2 = torch.nn.Linear(dim, expert_number)

    def get_dimension(self):
        return self.dim

    def forward(self, embeddings, batch):
        return softmax(self.w2(torch.tanh(self.w1(embeddings))), batch).mean(dim=1)


class GNN(torch.nn.Module):

    def __init__(self, num_features, num_classes=2, num_layers=3, dim=20, dropout=0.0, layer='gcn', pool='max', mitigation=None, sort_pool_k=None, deg=None, expert_count=1):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.mitigation = mitigation
        self.pool = pool
        self.sort_pool_k = sort_pool_k
        self.expert_count = expert_count
        self.layer = None
        if layer == 'gcn':
            self.layer = GCNConv
        elif layer == 'gat':
            self.layer = GATConv
        elif layer == 'gin':
            self.layer = GINWrapper
        elif layer == 'sage':
            self.layer = SAGEConv
        elif layer == 'pna':
            self.layer = PNAConvWrapper
        else:
            raise NotImplementedError(f'Layer: {layer} is not implemented!')

        if layer == 'pna':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']

            # First GCN layer.
            self.convs.append(self.layer(num_features, dim, aggregators=aggregators, scalers=scalers, deg=deg))
            self.bns.append(torch.nn.BatchNorm1d(dim))

            # Follow-up GCN layers.
            for i in range(self.num_layers - 1):
                self.convs.append(self.layer(dim, dim, aggregators=aggregators, scalers=scalers, deg=deg))
                self.bns.append(torch.nn.BatchNorm1d(dim))
        else:
            # First GCN layer.
            self.convs.append(self.layer(num_features, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

            # Follow-up GCN layers.
            for i in range(self.num_layers - 1):
                self.convs.append(self.layer(dim, dim))
                self.bns.append(torch.nn.BatchNorm1d(dim))

        # Fully connected layer.
        if self.pool == 'sort':
            self.fc = torch.nn.Linear(dim * self.sort_pool_k, num_classes)
        elif self.pool == 'expert':
            self.expert_attention = ExpertAttention(dim, self.expert_count)
            self.fc = torch.nn.Linear(dim, num_classes)
        elif self.pool == "m2":
            self.pooling_layer = GlobalMeanPool()
            self.fc = torch.nn.Linear(dim, num_classes)
        elif self.mitigation == "m2HM":  
            self.pooling_layer = GlobalMeanPool()
            self.fc = torch.nn.Linear(dim, num_classes)
        else:
            self.fc = torch.nn.Linear(dim, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self.layer):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def learn_node_weight(self, node_embeddings, k):
        pass

    def forward(self, data, edge_weight=None):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # GCNs.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after every layer.

        # Pooling and FCs.
        node_embeddings = x
        if self.pool == 'max':
            graph_embedding = global_max_pool(node_embeddings, batch)
        elif self.pool == 'sort':
            graph_embedding = global_sort_pool(node_embeddings, batch, k=self.sort_pool_k)
        elif self.pool == 'mean':
            graph_embedding = global_mean_pool(node_embeddings, batch)
        elif self.pool == 'expert':
            node_weight = self.expert_attention(node_embeddings, batch)
            graph_embedding = global_add_pool(node_embeddings * node_weight[:, None], batch)
        elif self.pool == "m2":
            graph_embedding = self.pooling_layer(node_embeddings,batch,edge_index=edge_index,edge_mask=edge_weight)
        elif self.mitigation == "m2HM":
            graph_embedding = self.pooling_layer(node_embeddings,batch,edge_index=edge_index,edge_mask=edge_weight)
        else:
            raise NotImplementedError(f'Pooling: {self.pool} is not implemented!')
        out = self.fc(graph_embedding)

        return node_embeddings, graph_embedding, out


class GIBGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes=2, num_layers=3, dim=20, dropout=0.0, layer='gcn', deg=None):
        super(GIBGNN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.layer = None
        if layer == 'gcn':
            self.layer = GCNConv
        elif layer == 'gat':
            self.layer = GATConv
        elif layer == 'gin':
            self.layer = GINWrapper
        elif layer == 'sage':
            self.layer = SAGEConv
        elif layer == 'pna':
            self.layer = PNAConv
        else:
            raise NotImplementedError(f'Layer: {layer} is not implemented!')

        if layer == 'pna':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']

            # First GCN layer.
            self.convs.append(self.layer(num_features, dim, aggregators=aggregators, scalers=scalers, deg=deg))
            self.bns.append(torch.nn.BatchNorm1d(dim))

            # Follow-up GCN layers.
            for i in range(self.num_layers - 1):
                self.convs.append(self.layer(dim, dim, aggregators=aggregators, scalers=scalers, deg=deg))
                self.bns.append(torch.nn.BatchNorm1d(dim))
        else:
            # First GCN layer.
            self.convs.append(self.layer(num_features, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

            # Follow-up GCN layers.
            for i in range(self.num_layers - 1):
                self.convs.append(self.layer(dim, dim))
                self.bns.append(torch.nn.BatchNorm1d(dim))

        # Fully connected layer.
        self.fc = torch.nn.Linear(dim, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self.layer):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def aggregate(self, assignment, x, batch, edge_index):

        max_id = torch.max(batch)
        all_pos_embedding = []

        st = 0
        for i in range(int(max_id + 1)):

            j = 0
            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1

            end = st + j

            if end == len(batch) - 1:
                end += 1

            one_batch_x = x[st:end]
            one_batch_assignment = assignment[st:end]

            group_features = torch.mm(one_batch_assignment.unsqueeze(0), one_batch_x)
            # pos_embedding = group_features[0].unsqueeze(dim=0)
            all_pos_embedding.append(group_features)

            st = end

        all_pos_embedding = torch.cat(tuple(all_pos_embedding), dim=0)

        return all_pos_embedding

    def forward(self, data, edge_weight=None):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        node_weight = data.node_weight

        # GCNs.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after every layer.

        # Pooling and FCs.
        node_embeddings = x
        # graph_embedding = self.aggregate(node_weight, node_embeddings, batch, edge_index)
        graph_embedding = global_add_pool(node_embeddings * node_weight[:, None], batch)
        # assert (graph_embedding == graph_embedding2).close()
        out = self.fc(graph_embedding)

        return node_embeddings, graph_embedding, out


def train(model, optimizer, train_loader, device, method):
    model.train()
    total_loss = 0

    for train_batch in train_loader:  # , desc='Train Batch', total=len(train_loader)):
        optimizer.zero_grad()
        out = model(train_batch.to(device))[-1]
        if method == 'classification':
            logits = F.log_softmax(out, dim=-1)
            loss = F.nll_loss(logits, train_batch.y.flatten())
        else:
            loss = F.mse_loss(out.flatten(), train_batch.y.flatten())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * out.shape[0]

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def eval(model, eval_loader, device, method):
    model.eval()
    total_loss = 0

    preds = []
    grounds = []
    for eval_batch in eval_loader:  # tqdm(eval_loader, desc='Eval Batch', total=len(eval_loader)):
        out = model(eval_batch.to(device))[-1]
        if method == 'classification':
            logits = F.log_softmax(out, dim=-1)
            loss = F.nll_loss(logits, eval_batch.y.flatten())
            # pred = torch.argmax(logits, dim=-1)
            # hits = (pred == eval_batch.y).sum()
            # total_hits += hits
        else:
            loss = F.mse_loss(out.flatten(), eval_batch.y.flatten())
            # total_hits = 0
        preds.append(out)
        grounds.append(eval_batch.y.flatten())
        total_loss += loss.item() * out.shape[0]  # eval_batch.num_graphs

    if method == 'classification':
        preds = torch.softmax(torch.cat(preds, dim=0), dim=1)
        grounds = torch.cat(grounds, dim=0)
        return total_loss, utils.auc(grounds, preds), utils.ap(grounds, preds), utils.accuracy(grounds, preds), preds, grounds
    else:
        preds = torch.cat(preds, dim=0).flatten()
        grounds = torch.cat(grounds, dim=0).flatten()
        return total_loss, utils.r_squared(grounds, preds), utils.mse(grounds, preds), utils.mae(grounds, preds), preds, grounds


def load_trained_gnn(dataset_name, device, path=None, num_classes=2):
    if path is None:
        model_path = f'data/{dataset_name}/gin/best_model.pt'
    else:
        model_path = path
    dataset = data_utils.load_dataset(dataset_name)
    model = GNN(
        num_features=dataset.num_features,
        num_classes=num_classes,
        num_layers=3,
        dim=20,
        dropout=0.0,
        layer='gin'
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


@torch.no_grad()
def load_trained_prediction(dataset_name, device, paths=None, num_classes=2):
    if paths is None:
        prediction_file = f'data/{dataset_name}/gin/preds.pt'
    else:
        prediction_file = paths[1]
    if os.path.exists(prediction_file):
        return torch.load(prediction_file, map_location=device)
    else:
        dataset = data_utils.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, device, path=paths[0], num_classes=num_classes)
        model.eval()

        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        preds = []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            out = model(eval_batch.to(device))[-1]
            logits = F.log_softmax(out, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            preds.append(pred)
        preds = torch.cat(preds)
        torch.save(preds, prediction_file)
        return preds


@torch.no_grad()
def load_trained_embeddings_logits(dataset_name, device, paths=None, num_classes=2):
    if paths is None:
        node_embeddings_file = f'data/{dataset_name}/gin/node_embeddings.pt'
        graph_embeddings_file = f'data/{dataset_name}/gin/graph_embeddings.pt'
        outs_file = f'data/{dataset_name}/gin/outs.pt'
        logits_file = f'data/{dataset_name}/gin/logits.pt'
    else:
        node_embeddings_file = paths[1]
        graph_embeddings_file = paths[2]
        outs_file = paths[3]
        logits_file = paths[4]
    if os.path.exists(node_embeddings_file) and os.path.exists(graph_embeddings_file) and os.path.exists(logits_file) and os.path.exists(outs_file):
        node_embeddings = torch.load(node_embeddings_file)
        for i, node_embedding in enumerate(node_embeddings):  # every graph has different size
            node_embeddings[i] = node_embeddings[i].to(device)
        graph_embeddings = torch.load(graph_embeddings_file, map_location=device)
        outs = torch.load(outs_file, map_location=device)
        logits = torch.load(logits_file, map_location=device)
        return node_embeddings, graph_embeddings, outs, logits
    else:
        dataset = data_utils.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, device, path=paths[0], num_classes=num_classes)
        model.eval()
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        graph_embeddings, node_embeddings, outs, logits = [], [], [], []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            node_emb, graph_emb, out = model(eval_batch.to(device))
            logit = F.log_softmax(out, dim=-1)
            max_batch_number = max(eval_batch.batch)
            for i in range(max_batch_number + 1):
                idx = torch.where(eval_batch.batch == i)[0]
                node_embeddings.append(node_emb[idx])
            graph_embeddings.append(graph_emb)
            outs.append(out)
            logits.append(logit)
        graph_embeddings = torch.cat(graph_embeddings)
        outs = torch.cat(outs)
        logits = torch.cat(logits)
        torch.save([node_embedding.cpu() for node_embedding in node_embeddings], node_embeddings_file)
        torch.save(graph_embeddings, graph_embeddings_file)
        torch.save(outs, outs_file)
        torch.save(logits, logits_file)
        return node_embeddings, graph_embeddings, outs, logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mutagenicity',
                        help="Dataset. Options are ['mutagenicity', 'aids', 'nci1', 'proteins']. Default is 'mutagenicity'. ")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate. Default is 0.0. ')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size. Default is 128.')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GCN layers. Default is 3.')
    parser.add_argument('--dim', type=int, default=20,
                        help='Number of GCN dimensions. Default is 20. ')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed for training. Default is 0. ')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs. Default is 1000. ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--method', type=str, default='classification')
    parser.add_argument('--layer', type=str, default='gcn')
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start_run', type=int, default=1)
    parser.add_argument('--algorithm', type=str, default='gnn', choices=['gnn', 'gibgnn'])
    parser.add_argument('--expert_count', type=int, default=1)
    return parser.parse_args()


def calculate_sort_pool_k(num_nodes):
    num_nodes = np.array(num_nodes)
    best_i = 0
    for i in range(1, max(num_nodes) + 1):
        if (num_nodes > i).sum() < (len(num_nodes) * 0.6):
            return best_i
        else:
            best_i = i


def get_class_name(layer, pool):
    if pool == 'sort':
        return 'sortpool'
    elif pool == 'diffpool':
        return 'diffpool'
    elif pool == 'expert':
        return 'expertpool'
    else:
        return layer


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Load and split the dataset.
    dataset = data_utils.load_dataset(args.dataset)
    splits, indices = data_utils.split_data(dataset)
    train_set, valid_set, test_set = splits

    max_nodes = max([graph.num_nodes for graph in dataset])

    from torch_geometric.utils import degree

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_set:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_set:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    sort_pool_k = calculate_sort_pool_k([graph.num_nodes for graph in dataset])

    # Logging.
    gnn_folder = f'data/{args.dataset}/{get_class_name(args.layer, args.pool)}/'
    if not os.path.exists(gnn_folder):
        os.makedirs(gnn_folder)
    log_file = gnn_folder + f'log.txt'
    with open(log_file, 'w') as f:
        pass

    if args.dataset == 'IMDB-M':
        args.num_classes = 3
    elif args.dataset == 'Tree-of-Life':
        args.num_classes = 1
        args.method = 'regression'

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    train_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
    valid_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
    test_scores = {'accuracy': [], 'auc': [], 'ap': [], 'mae': [], 'mse': [], 'r2': []}
    eval_times = []
    for run in range(args.start_run, args.start_run + args.runs):
        torch.backends.cudnn.deterministic = True
        random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        np.random.seed(run)

        if args.pool == 'diffpool':
            import torch_geometric.transforms as T

            def create_dense_set(sparse_set):
                dense_set = []
                for graph in sparse_set:
                    if 'edge_attr' in graph:
                        del graph.edge_attr
                    dense_set.append(T.ToDense(max_nodes)(graph))
                return dense_set

            train_loader = DenseDataLoader(create_dense_set(train_set), batch_size=args.batch_size, shuffle=True, num_workers=0)
            valid_loader = DenseDataLoader(create_dense_set(valid_set), batch_size=args.batch_size)
            test_loader = DenseDataLoader(create_dense_set(test_set), batch_size=args.batch_size)
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # Initialize the model.
        if args.algorithm == 'gnn':
            if args.pool == 'diffpool':
                model = DiffPool(
                    max_nodes=max_nodes,
                    num_features=dataset.num_features,
                    dim=args.dim,
                    num_classes=args.num_classes).to(device)
            else:
                model = GNN(
                    num_features=dataset.num_features,
                    num_classes=args.num_classes,
                    num_layers=args.num_layers,
                    dim=args.dim,
                    dropout=args.dropout,
                    layer=args.layer,
                    pool=args.pool,
                    sort_pool_k=sort_pool_k,
                    deg=deg,
                    expert_count=args.expert_count
                ).to(device)
        else:
            from torch_geometric.data import Data
            def add_dummy_node_weights(set):
                new_set = []
                for graph in set:
                    d = Data(edge_index=graph.edge_index.clone(), x=graph.x.clone(), y=graph.y.clone(), node_weight=torch.randn(graph.x.shape[0]))
                    new_set.append(d)
                return new_set

            train_loader = DataLoader(add_dummy_node_weights(train_set), batch_size=args.batch_size, shuffle=True, num_workers=0)
            valid_loader = DataLoader(add_dummy_node_weights(valid_set), batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(add_dummy_node_weights(test_set), batch_size=args.batch_size, shuffle=True, num_workers=0)
            model = GIBGNN(
                num_features=dataset.num_features,
                num_classes=args.num_classes,
                num_layers=args.num_layers,
                dim=args.dim,
                dropout=args.dropout,
                layer=args.layer,
                deg=deg
            ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # # Training.
        start_epoch = 1
        epoch_iterator = tqdm(range(start_epoch, start_epoch + args.epochs), desc='Epoch')
        best_valid = float('inf')
        best_epoch = 0
        patience = int(args.epochs / 5)
        cur_patience = 0
        # if run not in [1, 2]:
        for epoch in epoch_iterator:
            train_loss = train(model, optimizer, train_loader, device, method=args.method)
            valid_loss = eval(model, valid_loader, device, method=args.method)[0]
            if valid_loss < best_valid:
                print(valid_loss / len(valid_set))
                cur_patience = 0
                best_valid = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), gnn_folder + f'best_model_run_{run}.pt')
                # torch.save(optimizer.state_dict(), gnn_folder + f'optimizer_checkpoint_run_{run}.pth')
            else:
                cur_patience += 1
                if cur_patience >= patience:
                    break
            # with open(log_file, 'a') as f:
            #     print(f'Epoch = {epoch}:', file=f)
            #     print(f'Train Loss = {train_loss:.4e}', file=f)
            #     print(f'Valid Loss = {valid_loss:.4e}', file=f)
        # with open(log_file, 'a') as f:
        #     print(f'Best Epoch = {best_epoch}', file=f)

        # Testing.
        model.load_state_dict(torch.load(gnn_folder + f'best_model_run_{run}.pt', map_location=device))
        # torch.save(model.state_dict(), gnn_folder + f'model_best_run_{run}.pt')
        # if run == 0:
        #     torch.save(model.state_dict(), gnn_folder + f'model_best.pth')

        # evaluation
        start_eval = time.time()
        if args.method == 'classification':
            train_loss, train_auc, train_ap, train_acc, train_preds, train_grounds = eval(model, train_loader, device, method=args.method)
            valid_loss, valid_auc, valid_ap, valid_acc, valid_preds, valid_grounds = eval(model, valid_loader, device, method=args.method)
            test_loss, test_auc, test_ap, test_acc, test_preds, test_grounds = eval(model, test_loader, device, method=args.method)
            train_scores['auc'].append(train_auc)
            train_scores['ap'].append(train_ap)
            train_scores['accuracy'].append(train_acc)
            valid_scores['auc'].append(valid_auc)
            valid_scores['ap'].append(valid_ap)
            valid_scores['accuracy'].append(valid_acc)
            test_scores['auc'].append(test_auc)
            test_scores['ap'].append(test_ap)
            test_scores['accuracy'].append(test_acc)
        else:
            train_loss, train_r2, train_mse, train_mae, train_preds, train_grounds = eval(model, train_loader, device, method=args.method)
            valid_loss, valid_r2, valid_mse, valid_mae, valid_preds, valid_grounds = eval(model, valid_loader, device, method=args.method)
            test_loss, test_r2, test_mse, test_mae, test_preds, test_grounds = eval(model, test_loader, device, method=args.method)
            train_scores['r2'].append(train_r2)
            train_scores['mse'].append(train_mse)
            train_scores['mae'].append(train_mae)
            valid_scores['r2'].append(valid_r2)
            valid_scores['mse'].append(valid_mse)
            valid_scores['mae'].append(valid_mae)
            test_scores['r2'].append(test_r2)
            test_scores['mse'].append(test_mse)
            test_scores['mae'].append(test_mae)

        torch.save((train_preds, train_grounds), gnn_folder + f'train_predictions_run_{run}.pt')
        torch.save((valid_preds, valid_grounds), gnn_folder + f'valid_predictions_run_{run}.pt')
        torch.save((test_preds, test_grounds), gnn_folder + f'test_predictions_run_{run}.pt')
        end_eval = time.time()
        eval_times.append(end_eval - start_eval)
        print(f'Eval takes: {end_eval - start_eval}s')

    print(f'Average Eval takes: {np.mean(eval_times)}s')

    print(f"Train Scores = {train_scores}", flush=True)
    print(f"Valid Scores = {valid_scores}", flush=True)
    print(f"Test Scores = {test_scores}", flush=True)
    all_scores = {'train': train_scores, 'valid': valid_scores, 'test_scores': test_scores}
    torch.save(all_scores, gnn_folder + f'all_scores_{args.start_run}_{args.runs + args.start_run - 1}.pt')
    with open(log_file, 'a') as f:
        print(file=f)
        print(f"Train Scores = {train_scores}", file=f)
        print(f"Valid Scores = {valid_scores}", file=f)
        print(f"Test Scores = {test_scores}", file=f)

        if args.method == 'classification':
            print(f"Train AUC = {np.mean(train_scores['auc'])} +- {np.std(train_scores['auc'])}", file=f)
            print(f"Valid AUC = {np.mean(valid_scores['auc'])} +- {np.std(valid_scores['auc'])}", file=f)
            print(f"Test AUC = {np.round(np.mean(test_scores['auc']), 4)} +- {np.round(np.std(test_scores['auc']), 4)}", file=f)

            print(f"Train AP = {np.mean(train_scores['ap'])} +- {np.std(train_scores['ap'])}", file=f)
            print(f"Valid AP = {np.mean(valid_scores['ap'])} +- {np.std(valid_scores['ap'])}", file=f)
            print(f"Test AP = {np.round(np.mean(test_scores['ap']), 4)} +- {np.round(np.std(test_scores['ap']), 4)}", file=f)

            print(f"Train Accuracy = {np.mean(train_scores['accuracy'])} +- {np.std(train_scores['accuracy'])}", file=f)
            print(f"Valid Accuracy = {np.mean(valid_scores['accuracy'])} +- {np.std(valid_scores['accuracy'])}", file=f)
            print(f"Test Accuracy = {np.round(np.mean(test_scores['accuracy']), 4)} +- {np.round(np.std(test_scores['accuracy']), 4)}", file=f)
        else:
            print(f"Train R2 = {np.mean(train_scores['r2'])} +- {np.std(train_scores['r2'])}", file=f)
            print(f"Valid R2 = {np.mean(valid_scores['r2'])} +- {np.std(valid_scores['r2'])}", file=f)
            print(f"Test R2 = {np.round(np.mean(test_scores['r2']), 4)} +- {np.round(np.std(test_scores['r2']), 4)}", file=f)

            print(f"Train MSE = {np.mean(train_scores['mse'])} +- {np.std(train_scores['mse'])}", file=f)
            print(f"Valid MSE = {np.mean(valid_scores['mse'])} +- {np.std(valid_scores['mse'])}", file=f)
            print(f"Test MSE = {np.round(np.mean(test_scores['mse']), 4)} +- {np.round(np.std(test_scores['mse']), 4)}", file=f)

            print(f"Train MAE = {np.mean(train_scores['mae'])} +- {np.std(train_scores['mae'])}", file=f)
            print(f"Valid MAE = {np.mean(valid_scores['mae'])} +- {np.std(valid_scores['mae'])}", file=f)
            print(f"Test MAE = {np.round(np.mean(test_scores['mae']), 4)} +- {np.round(np.std(test_scores['mae']), 4)}", file=f)


if __name__ == '__main__':
    main()
