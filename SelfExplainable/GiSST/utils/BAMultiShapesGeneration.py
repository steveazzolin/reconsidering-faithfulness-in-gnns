from networkx.generators import random_graphs, lattice, small, classic
import networkx as nx
import pickle as pkl
import numpy as np
from networkx.algorithms.operators.binary import compose, union
import torch
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt
from torch_geometric.utils import mask_to_index
from torch_geometric.data import Data


# Originally From
# https://github.com/steveazzolin/gnn_logic_global_expl/blob/master/datasets/BAMultiShapes/generate_dataset.py
# Pyg had the dataset by default, but edge and node masks were missing
# It has been heavily modified, but is still based on the original method of generating the dataset

def combine_graphs(baGraph, motif_list):
    #print(torch.tensor(list(baGraph.edges())).long().transpose(0,1))
    edge_indices = [torch.from_numpy(nx.adjacency_matrix(baGraph).A).nonzero().t()]
    node_masks = [torch.zeros(baGraph.number_of_nodes()).long()]
    edge_masks = [torch.zeros(baGraph.number_of_edges()*2).long()] #Temp
    num_nodes = baGraph.number_of_nodes()
    
    for motif in motif_list:
        motif_edge_index = torch.from_numpy(nx.adjacency_matrix(motif).A).nonzero().t()

        # Add motif to the graph.
        edge_indices.append(motif_edge_index + num_nodes)
        node_masks.append(torch.ones(motif.number_of_nodes()))
        edge_masks.append(torch.ones(motif.number_of_edges()*2))
        
        i = int(torch.randint(0, num_nodes, (1, )))
        
        j = int(torch.randint(0, motif.number_of_nodes(), (1, ))) + num_nodes
        edge_indices.append(torch.tensor([[i, j], [j, i]]))
        edge_masks.append(torch.zeros(2))
        
        num_nodes += motif.number_of_nodes()
        
    edge_index=torch.cat(edge_indices, dim=1),
    edge_mask=torch.cat(edge_masks, dim=0),
    node_mask=torch.cat(node_masks, dim=0),
    
    return edge_index[0], node_mask[0], edge_mask[0]

def generate_class1(nb_node_ba=40):
    r = np.random.randint(3)
    
    if r == 0: # W + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-9, 1)
        g2 = classic.wheel_graph(6)
        g3 = get_grid_graph()
    elif r == 1: # W + H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-5, 1)
        g2 = classic.wheel_graph(6)
        g3 = small.house_graph()
    elif r == 2: # H + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5-9, 1)
        g2 = small.house_graph()
        g3 = get_grid_graph()
        
    edge_index, node_mask, edge_mask = combine_graphs(g1, [g2, g3])
    return edge_index, node_mask, edge_mask

def generate_class0(nb_node_ba=40):
    r = np.random.randint(10)

    
    if r > 3:
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba, 1) 
        motifs = []
        
    if r == 0: # W
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6, 1)
        g2 = classic.wheel_graph(6)
        motifs = [g2]
    if r == 1: # H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5, 1)
        g2 = small.house_graph()
        motifs = [g2]
    if r == 2: # G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9, 1)
        g2 = get_grid_graph()
        motifs = [g2]          
    if r == 3: # All
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9-5-6, 1)
        g2 = small.house_graph()
        g3 = get_grid_graph()
        g4 =  classic.wheel_graph(6)
        
        motifs = [g2, g3, g4]
        
    edge_index, node_mask, edge_mask = combine_graphs(g1, motifs)
    return edge_index, node_mask, edge_mask

def get_grid_graph():
    gridgraph = lattice.grid_2d_graph(int(3), int(3))
    i = 0
    mapping = dict()
    for n in gridgraph.nodes():
        mapping[n] = i
        i = i + 1
    gridgraph = nx.relabel_nodes(gridgraph,mapping)
    
    return gridgraph

def generate(num_samples):
    assert num_samples % 2 == 0
    edge_indexes = []
    labels = []
    feats = []
    node_masks = []
    edge_masks = []
    nb_node_ba = 40

    for _ in range(int(num_samples/2)):
        edge_index, node_mask, edge_mask = generate_class1(nb_node_ba=nb_node_ba)
        edge_indexes.append(edge_index)
        node_masks.append(node_mask)
        edge_masks.append(edge_mask)
        labels.append(0)
        feats.append(list((np.ones((nb_node_ba,10)))/10))

    for _ in range(int(num_samples/2)):
        edge_index, node_mask, edge_mask = generate_class0(nb_node_ba=nb_node_ba)
        edge_indexes.append(edge_index)
        node_masks.append(node_mask)
        edge_masks.append(edge_mask)
        labels.append(1)
        feats.append(list((np.ones((nb_node_ba,10)))/10))
    return edge_indexes, feats, labels, node_masks, edge_masks 


def generate_data_list(num_graphs = 1000):
    edge_indexes, feats, labels, node_masks, edge_masks = generate(num_graphs)
    
    data_list = []
    
    for i in range(0, len(labels)):
        x = torch.from_numpy(np.array(feats[i])).to(torch.float)
        data = Data(x=x, edge_index=edge_indexes[i], y=torch.tensor([labels[i]]), node_mask = node_masks[i], edge_mask = edge_masks[i])
        data_list.append(data)
        
    return data_list    

if __name__ == "__main__":
    seed_everything(123)
    print(generate_data_list(1000)[0])
    
    
    