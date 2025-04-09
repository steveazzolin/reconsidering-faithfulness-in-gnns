import os
import argparse
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import utils
import gnn
from collections import OrderedDict
import higher

import data_utils
from ourutils.splitting import relabel,split_graph


class GNNNode(torch.nn.Module):

    def __init__(self, num_features, num_layers=3, dim=20, dropout=0.0, layer='gcn'):
        super(GNNNode, self).__init__()

        self.num_features = num_features
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
            self.layer = gnn.GINWrapper
        else:
            raise NotImplementedError(f'Layer: {layer} is not implemented!')

        # First GCN layer.
        self.convs.append(self.layer(num_features, dim))
        self.bns.append(torch.nn.BatchNorm1d(dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(self.layer(dim, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, self.layer):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

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
        return node_embeddings


class EdgeSampler(torch.nn.Module):
    def __init__(self, args):
        super(EdgeSampler, self).__init__()

        self.gnn_node = GNNNode(num_features=args.num_features, num_layers=args.num_layers, dim=args.dim, dropout=args.dropout, layer=args.explainer_layer)
        self.edge_sampler = torch.nn.Sequential(
            torch.nn.Linear(args.dim * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    def generate_edge_embeddings(self, node_embeds, edge_index):
        rows = edge_index[0]
        cols = edge_index[1]
        row_embeds = node_embeds[rows]
        col_embeds = node_embeds[cols]
        min_row_col_embeds = torch.minimum(row_embeds, col_embeds)
        max_row_col_embeds = torch.maximum(row_embeds, col_embeds)
        edge_embeds = torch.cat([min_row_col_embeds, max_row_col_embeds], 1)
        return edge_embeds

    def forward(self, data, edge_weight=None):
        node_embeddings = self.gnn_node(data, edge_weight)
        edge_embeddings = self.generate_edge_embeddings(node_embeddings, data.edge_index)
        edge_explanations = self.edge_sampler(edge_embeddings)
        return edge_explanations

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()


class RAGE(torch.nn.Module):
    def __init__(self, args):
        super(RAGE, self).__init__()

        self.args = args
        self.outer = EdgeSampler(args)

        self.inner = None
        self.inner_optimizer = None
        self.inner_parameters = None

        self.reset_inner_optimizer()

    @torch.no_grad()
    def eval_all(self, data):
        # se mitigation non sigmoid
        if self.args.mitigation == "HM" or self.args.mitigation == "m2HM":
            explanation = self.outer(data).flatten()
            explanation = self.sampling(explanation,training=False)
        else:
            explanation = torch.sigmoid(self.outer(data).flatten())
        _, _, out_subgraph = self.inner(data, edge_weight=explanation)
        return explanation, out_subgraph

    def eval_inner(self, data):
        # se mitigation non sigmoid
        if self.args.mitigation == "HM" or self.args.mitigation == "m2HM":
            explanation = self.outer(data).flatten()
            explanation = self.sampling(explanation,training=False)
        else:
            explanation = torch.sigmoid(self.outer(data).flatten())

        _, _, out_subgraph = self.inner(data, edge_weight=explanation)
        return explanation, out_subgraph
    
    @torch.no_grad()
    def get_explanation(self,data):
        # se mitigation non sigmoid
        if self.args.mitigation == "HM" or self.args.mitigation == "m2HM":
            explanation = self.outer(data).flatten()
            explanation = self.sampling(explanation,training=False)
        else:
            explanation = torch.sigmoid(self.outer(data).flatten())
        return explanation
    
    @torch.no_grad()
    def probs(self, data, **kwargs):        
        _,out_subgraph = self(data,is_inference=True,**kwargs)
        if out_subgraph.shape[-1] > 1:
            return out_subgraph.softmax(dim=1)
        else:
            return out_subgraph.sigmoid()
    
    @torch.no_grad()
    def log_probs(self,data,eval_kl=False, **kwargs):
        #explanation = torch.sigmoid(self.outer(data).flatten())
        #_, _, out_subgraph = self.inner(data, edge_weight=explanation)
        _,out_subgraph = self(data,is_inference=True,**kwargs)
        if out_subgraph.shape[-1] > 1:
            return out_subgraph.softmax(dim=1)
        else:
            if eval_kl: # make the single logit a proper distribution summing to 1 to compute KL
                lc_logits = out_subgraph.sigmoid()
                new_logits = torch.zeros((lc_logits.shape[0], lc_logits.shape[1]+1), device=lc_logits.device)
                new_logits[:, 1] = new_logits[:, 1] + lc_logits.squeeze(1)
                new_logits[:, 0] = 1 - new_logits[:, 1]
                new_logits[new_logits == 0.] = 1e-10
                return new_logits.log()
            else:
                return out_subgraph.sigmoid().log()
            

    def forward(self, data,is_inference=False,weight=None,is_ratio=False):
        # se mitigation non sigmoid

        if self.args.mitigation == "HM" or self.args.mitigation == "m2HM":   
            with torch.no_grad():
                explanation = self.outer(data).flatten()
                explanation = self.sampling(explanation,training=False)
        else:
            with torch.no_grad():
                explanation = torch.sigmoid(self.outer(data).flatten())

        #presa da steve
        # edge_prob = expla
        if weight== False:
            assert False
        if weight:
            if is_ratio:
                data = Data(x=data.x, edge_index=data.edge_index, batch=data.batch)
                (causal_edge_index, causal_edge_attr, causal_edge_weight), _ = split_graph(data, explanation, weight)
                causal_x, causal_edge_index, causal_batch, _ = relabel(data.x, causal_edge_index, data.batch)
                x = causal_x
                batch = causal_batch
                edge_index = causal_edge_index
                explanation = causal_edge_weight
                
                data.x = causal_x
                data.batch = causal_batch
                data.edge_index = causal_edge_index

        _, _, out_subgraph = self.inner(data, edge_weight=explanation)
        
        if not is_inference:
            if self.args.method == 'classification':
                logits = F.log_softmax(out_subgraph, dim=-1)
                loss = F.nll_loss(logits, data.y.flatten())
            else:
                loss = F.mse_loss(out_subgraph.flatten(), data.y.flatten())
            new_parameters = self.inner_optimizer.step(loss, params=self.inner.parameters())
            self.inner_parameters = new_parameters
            self.set_inner_parameters()

        return explanation, out_subgraph
    
    def sampling(self, att_log_logits, training, mitigation_expl_scores="hard"):
        #print("faccio il sampling")
        temp = 1
        att = self.concrete_sample(att_log_logits, temp=temp, training=training)
        if mitigation_expl_scores == "hard":
            att_hard = torch.where(att > 0.5, 0.99999, 0.00001)
            att = att_hard
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
    
    def set_inner_parameters(self):
        state = OrderedDict()
        prev_state = self.inner.state_dict()
        trainable_parameter_keys = [item[0] for item in list(self.inner.named_parameters())]
        for state_id in prev_state:
            if state_id in trainable_parameter_keys:
                idx = trainable_parameter_keys.index(state_id)
                state[state_id] = self.inner_parameters[idx]
            else:
                state[state_id] = prev_state[state_id]
        self.inner.load_state_dict(state)

    def reset_inner_optimizer(self, full_reset=False):
        self.inner = gnn.GNN(num_features=self.args.num_features,
                             num_classes=self.args.num_classes,
                             num_layers=self.args.num_layers,
                             dim=self.args.dim,
                             dropout=self.args.dropout,
                             layer=self.args.gnn_layer,
                             pool=self.args.gnn_pool,
                             mitigation = self.args.mitigation,
                             sort_pool_k=None,
                             deg=None)
        if self.inner_parameters is not None and not full_reset:
            self.set_inner_parameters()
        self.inner.to(self.args.device)
        inner_optimizer = torch.optim.Adam(self.inner.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.inner_optimizer = higher.get_diff_optim(inner_optimizer, self.inner.parameters())


def train(model, optimizer, train_loader, valid_loader, device, method, args):
    model.train()
    model.reset_inner_optimizer(full_reset=True)
    total_loss = 0

    for train_batch in train_loader:
        current_batch_size = len(train_batch)
        inner_batch = Batch.from_data_list([train_batch[i] for i in range(int(current_batch_size / 2))])
        support_batch = Batch.from_data_list([train_batch[i] for i in range(int(current_batch_size / 2) + 1, current_batch_size)])

        optimizer.zero_grad()
        model.reset_inner_optimizer(full_reset=False)

        model.inner.train()

        best_inner_loss = float('inf')
        for i in range(args.inner_epoch):
            model(inner_batch.to(device))
            valid_loss = eval(model, valid_loader, device, method=args.method, args=args)[0]
            if valid_loss < best_inner_loss:
                best_inner_loss = valid_loss
                if args.gnn_pool == "m2":
                    torch.save(model.state_dict(), args.result_folder + f'best_inner_model_{args.run}_m2.pt')
                elif args.mitigation == "HM":
                    torch.save(model.state_dict(), args.result_folder + f'best_inner_model_{args.run}_HM.pt')
                elif args.mitigation == "m2HM":
                    torch.save(model.state_dict(), args.result_folder + f'best_inner_model_{args.run}_m2HM.pt')
                else:
                    torch.save(model.state_dict(), args.result_folder + f'best_inner_model_{args.run}.pt')

        if args.gnn_pool == "m2":
            model.load_state_dict(torch.load(args.result_folder + f'best_inner_model_{args.run}_m2.pt', map_location=device))
        if args.mitigation == "HM":
            model.load_state_dict(torch.load(args.result_folder + f'best_inner_model_{args.run}_HM.pt', map_location=device))
        if args.mitigation == "m2HM":
            model.load_state_dict(torch.load(args.result_folder + f'best_inner_model_{args.run}_m2HM.pt', map_location=device))
        else:
            model.load_state_dict(torch.load(args.result_folder + f'best_inner_model_{args.run}.pt', map_location=device))

        model.train()
        model.inner.eval()

        explanation, out_subgraph = model.eval_inner(support_batch.to(device))
        if method == 'classification':
            logits = F.log_softmax(out_subgraph, dim=-1)
            loss = F.nll_loss(logits, support_batch.y.flatten())
        else:
            loss = F.mse_loss(out_subgraph.flatten(), support_batch.y.flatten())

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * out_subgraph.shape[0]

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def eval(model, eval_loader, device, method, args):
    model.eval()
    total_loss = 0
    preds = []
    grounds = []
    for eval_batch in eval_loader:  # tqdm(eval_loader, desc='Eval Batch', total=len(eval_loader)):
        explanation, out_subgraph = model.eval_all(eval_batch.to(device))
        if method == 'classification':
            logits = F.log_softmax(out_subgraph, dim=-1)
            loss = F.nll_loss(logits, eval_batch.y.flatten())
        else:
            loss = F.mse_loss(out_subgraph.flatten(), eval_batch.y.flatten())
        
        preds.append(out_subgraph)
        grounds.append(eval_batch.y.flatten())
        total_loss += loss.item() * out_subgraph.shape[0]  # eval_batch.num_graphs

    if method == 'classification':
        preds = torch.softmax(torch.cat(preds, dim=0), dim=1)
        grounds = torch.cat(grounds, dim=0)
        return total_loss / len(eval_loader.dataset), utils.auc(grounds, preds), utils.ap(grounds, preds), utils.accuracy(grounds, preds), preds, grounds
    else:
        preds = torch.cat(preds, dim=0).flatten()
        grounds = torch.cat(grounds, dim=0).flatten()
        return total_loss / len(eval_loader.dataset), utils.r_squared(grounds, preds), utils.mse(grounds, preds), utils.mae(grounds, preds), preds, grounds


@torch.no_grad()
def performance(preds, grounds, method):
    if method == 'classification':
        return utils.auc(grounds, preds), utils.ap(grounds, preds), utils.accuracy(grounds, preds)
    else:
        return utils.r_squared(grounds, preds), utils.mse(grounds, preds), utils.mae(grounds, preds)


@torch.no_grad()
def get_explanation(model, graph, device):
    model.eval()
    explanation = model(graph.to(device))[0]
    return explanation

