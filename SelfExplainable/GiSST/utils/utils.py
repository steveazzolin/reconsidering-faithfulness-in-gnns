import torch
from torch_geometric.utils import mask_to_index, index_to_mask
from torch_geometric.utils import k_hop_subgraph
import time
    
def get_node_and_edge_mask(data, model, id, threshold = 0.5):
    #if random:
        #return get_random_explanation(data, threshold)
    
    #Edge mask creation
    expl_feat, masked_edge_importance = model.get_explanation(int(id))
    
    if masked_edge_importance != None:   
        masked_edge_importance = masked_edge_importance.detach().cpu()
        edge_mask = masked_edge_importance.gt(threshold).detach().cpu()
    else:
        edge_mask = None
    
    if expl_feat != None:
        expl_feat = expl_feat.detach().cpu()
        node_mask = expl_feat.gt(threshold).long().detach().cpu()
    else:
        node_mask = None
    
    permuted_edge_mask, permuted_node_mask = get_permuted_masks(edge_mask, node_mask)
       
    return edge_mask, node_mask, permuted_edge_mask, permuted_node_mask, masked_edge_importance, expl_feat,

def get_permuted_masks(edge_mask, node_mask):
    #permute edge mask
    permuted_edge_mask = None
    if edge_mask != None:
        perm=torch.randperm(len(edge_mask))
        permuted_edge_mask = edge_mask[perm]
        permuted_edge_mask = permuted_edge_mask.detach().cpu()
    
    #Permute node mask
    permuted_node_mask = None
    if node_mask != None:
        r=torch.randperm(node_mask.size()[0])
        c=torch.randperm(node_mask.size()[1])
        permuted_node_mask=node_mask[r[:, None], c]

        # With view
        idx = torch.randperm(permuted_node_mask.nelement())
        permuted_node_mask = permuted_node_mask.view(-1)[idx].view(permuted_node_mask.size())
        permuted_node_mask = permuted_node_mask.detach().cpu()
    
    return permuted_edge_mask, permuted_node_mask
    

def get_random_explanation(data, threshold):
    edge_mask = torch.randn(len(data.edge_index[0])).gt(threshold).long()
    node_mask = torch.randn_like(data.x).gt(threshold).long()
    
    return edge_mask, node_mask

def train_model_with_time(model):
    t1 = time.time()
    model.train()
    model.save()
    t2 = time.time()
    
    return t2-t1