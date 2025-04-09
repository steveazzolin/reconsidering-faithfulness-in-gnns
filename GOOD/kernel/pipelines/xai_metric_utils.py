import networkx as nx
import torch
from random import randint, shuffle
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np
import os

from torch_geometric.utils import remove_isolated_nodes, dropout_edge

edge_colors = {
    "inv": "green",
    "spu": "blue",
    "added": "red"
}
node_colors = {
    True: "red",
    False: "#1f78b4"
}


def remove_from_graph(G, edge_index_to_remove=None, what_to_remove=None):
    if edge_index_to_remove is None and what_to_remove:
        G = G.copy()
        edge_remove = []
        for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
            if val == what_to_remove:
                edge_remove.append((u,v))
        G.remove_edges_from(edge_remove)
        G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
        G.remove_nodes_from(list(nx.isolates(G)))
    elif not edge_index_to_remove is None:
        G = G.copy()
        G.remove_edges_from([(u.item(), v.item()) for u,v in edge_index_to_remove.T])
        G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
        G.remove_nodes_from(list(nx.isolates(G)))
    else:
        raise ValueError(f"what_to_remove can not be None with edge_index None")
    return G

def mark_edges(G, inv_edge_index, spu_edge_index, inv_edge_w=None, spu_edge_w=None):
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "inv" for u,v in inv_edge_index.T}
    )
    if not inv_edge_w is None:
        d = {(u.item(), v.item()): round(inv_edge_w[i].item(),2) for i, (u,v) in enumerate(inv_edge_index.T)}
        assert np.all([d[u,v] == d[v,u] for u,v in d.keys()])
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values=d
        )
    nx.set_edge_attributes(
        G,
        name="origin",
        values={(u.item(), v.item()): "spu" for u,v in spu_edge_index.T}
    )
    if not spu_edge_w is None:
        d = {(u.item(), v.item()): round(spu_edge_w[i].item(),2) for i, (u,v) in enumerate(spu_edge_index.T)}
        assert np.all([d[u,v] == d[v,u] for u,v in d.keys()])
        nx.set_edge_attributes(
            G,
            name="attn_weight",
            values=d
        )

def mark_frontier(G, G_filt):
    # mark frontier nodes as nodes attached to both inv and spu parts
    # to mark nodes check which nodes have a change in the degree between original and filtered graph           
    
    frontier = list(filter(lambda n: G.degree[n] != G_filt.degree[n], G_filt.nodes()))

    nx.set_node_attributes(G_filt, name="frontier", values=False)
    nx.set_node_attributes(G_filt, name="frontier", values={n: True for n in frontier})
    return len(frontier)

def random_attach_no_target_frontier(S, T):
    # random attach frontier nodes in S and T
    # avoid selecting target nodes that are in the frontier
    
    edge_attrs = list(nx.get_edge_attributes(S, "edge_attr").values())
    S_frontier = list(filter(lambda x: nx.get_node_attributes(S,'frontier').get(x, False), S.nodes()))

    ret = nx.union(S, T, rename=("", "T"))
    for n in S_frontier:
        # pick random node v in G_t_spu
        # add edge (u,v) and (v,u)
        idx = randint(0, len(T.nodes()) - 1)
        v = "T" + str(list(T)[idx])

        if edge_attrs != []:
            attr = randint(0, len(edge_attrs) - 1)
            ret.add_edge(str(n), v, edge_attr=edge_attrs[attr])
            ret.add_edge(v, str(n), edge_attr=edge_attrs[attr])
        else:
            ret.add_edge(str(n), v)
            ret.add_edge(v, str(n))
    return ret

def expl_acc(expl, data, expl_weight=None):
    edge_gt = {(u.item(),v.item()): data.edge_gt[i] for i, (u,v) in enumerate(data.edge_index.T)} 
    edge_expl = set([(u.item(),v.item()) for u,v in expl.T])
    
    tp = int(sum([edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    fp = int(sum([not edge_gt[(u.item(),v.item())] for u,v in expl.T]))
    tn = int(sum([not (u.item(),v.item()) in edge_expl and not edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    fn = int(sum([not (u.item(),v.item()) in edge_expl and edge_gt[(u.item(),v.item())] for u,v in data.edge_index.T]))
    
    f1 = 2*tp / (2*tp + fp + fn)

    wiou, den = 0, 1e-12
    for i, (u,v) in enumerate((data.edge_index.T)):
        u, v = u.item(), v.item()
        if edge_gt[(u,v)]:
            if (u,v) in edge_expl:
                wiou += expl_weight[i].item()
                den += expl_weight[i].item()
            else:
                den += expl_weight[i].item()
        elif (u,v) in edge_expl:
            den += expl_weight[i].item()
    wiou = wiou / den
    return round(wiou, 3), round(f1, 3)

def sample_edges(G_ori, alpha, deconfounded, edge_index_to_remove):
    # keep each spu/inv edge with probability alpha
    G = G_ori.copy()
    if not deconfounded:
        # bernoulli sampling
        # edges = set()
        # for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
            # pass
            # if val == where_to_sample:
            #     edges.add((u,v))                
                # if where_to_sample == "spu" and np.random.binomial(1, alpha, 1)[0] == 0:
                #     edge_remove.append((u,v))
        edges = list(edges)
    else:
        edges = [(u.item(), v.item()) for u, v in edge_index_to_remove.T]    
    
    shuffle(edges)
    edge_remove = edges[:int(len(G.edges()) * (1-alpha))] #remove the 1-alpha% of the undirected edges
    # edge_remove = edges[:1]
    G.remove_edges_from(edge_remove)
    G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def sample_edges_tensorized(data, nec_number_samples, sampling_type, nec_alpha_1, avg_graph_size, edge_index_to_remove=None, force_undirected=True):
    if sampling_type == "bernoulli":
        assert not nec_alpha_1 is None

        # customization of dropout_edge from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html#dropout_edge
        data = data.clone()

        row, col = data.edge_index
        mask_noncausal = torch.ones(row.size(0), dtype=bool)
        mask_noncausal[edge_index_to_remove] = False
        
        edge_mask = torch.rand(row.size(0), device=data.edge_index.device) >= nec_alpha_1
        edge_mask[mask_noncausal] = True
        if force_undirected:
            edge_mask[row > col] = False

        edge_index = data.edge_index[:, edge_mask]

        if force_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_id_kept = edge_mask.nonzero().repeat((2, 1)).squeeze()
        else:
            edge_id_kept = edge_mask

        if hasattr(data, "edge_attr") and not data.edge_attr is None:
            data.edge_attr = data.edge_attr[edge_id_kept]
        
        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask]
        data.num_nodes = data.x.shape[0]
        return data
    elif sampling_type == "deconfounded":
        if nec_number_samples == "prop_G_dataset":
            k = max(1, int(nec_alpha_1 * avg_graph_size))
        elif nec_number_samples == "prop_R":
            k = max(1, int(nec_alpha_1 * edge_index_to_remove.sum()))
        elif nec_number_samples == "alwaysK":
            k = nec_alpha_1
        else:
            raise ValueError(f"value for nec_number_samples ({nec_number_samples}) not supported")
        
        k = min(k, int(data.edge_index.shape[1]/2)-2)
        assert k > 0, k

        data = data.clone()
        row, col = data.edge_index
        undirected = data.edge_index[:, row <= col]

        candidate_mask = edge_index_to_remove[row <= col]
        candidate_idxs = torch.argwhere(candidate_mask)
        perm = torch.randperm(candidate_idxs.shape[0])
        to_keep = perm[:-k]
        removed = perm[-k:]
        causal_idxs_keep = candidate_idxs[to_keep].view(-1)
        causal_idxs_remove = candidate_idxs[removed].view(-1)

        to_keep = torch.zeros(undirected.shape[1], dtype=torch.bool)
        to_keep[candidate_mask == 0] = 1
        to_keep[causal_idxs_keep] = 1

        data.edge_index = torch.cat((undirected[:, to_keep], undirected[:, to_keep].flip(0)), dim=1)
        
        if hasattr(data, "edge_attr") and not data.edge_attr is None:
            undirected_edge_attr = data.edge_attr[row <= col]
            data.edge_attr = torch.cat((undirected_edge_attr[to_keep, :], undirected_edge_attr[to_keep, :]), dim=0)

        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask]
        data.num_nodes = data.x.shape[0]
        return data
    else:
        raise ValueError(f"sampling_type {sampling_type} not valid")
