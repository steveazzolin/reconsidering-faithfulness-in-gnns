import torch
from torch_geometric.utils import mask_to_index, index_to_mask
import torch.nn.functional as F

from torch_geometric.explain import groundtruth_metrics
from torch_geometric.utils import k_hop_subgraph
import math
from utils.utils import *

def test_feature_unfaithfulness(model, cache, top_k = 0.25):
    """Test unfaithfulness based on features
    
    Not used in the thesis and might be bugged

    Args:
        model (_type_): Model to test
        cache (_type_): Cache to use
        top_k (float, optional): The percentage of features to keep. Defaults to 0.25.

    Returns:
        An unfaithfulness score
    """
    dataset = model.dataset

    K = math.ceil(top_k * dataset.num_features)
    
    test_idx = dataset.index_test #TODO Fiqure if test mask is best to use...
    
    org_preds = torch.zeros((test_idx.shape[0], dataset.num_classes))    
    topk_feature_preds = torch.zeros(org_preds.shape)
    for elemnum, graph_id in enumerate(test_idx):
        data = dataset[graph_id].to(model.device)
        tempData = data.clone().to(model.device)
        
        org_preds[elemnum] = model.get_prediction_vector(data)
        
        _, _, _, _, _ , expl_feat = get_explanation(cache, elemnum, model, data, graph_id)
        _, top_k_index = torch.topk(expl_feat, k = K)
        node_mask = torch.zeros_like(data.x)
        
        #Zero features TODO Gaussian noise option
        node_mask[:, top_k_index] = 1.0
        tempData.x = data.x.clone() * node_mask
        topk_feature_preds[elemnum] = (model.get_prediction_vector(tempData))
    
    org_preds = org_preds.cpu()
    topk_feature_preds = topk_feature_preds.cpu()
        
    org_softmax = F.softmax(org_preds, dim=-1)
    pert_softmax = F.softmax(topk_feature_preds, dim=-1)
    
    eps = eps=1e-7
    kl_div = F.kl_div((org_softmax+eps).log() , pert_softmax, reduction='batchmean')
    result = 1 - float(torch.exp(-kl_div))
    
    return result

def test_full_unfaithfulness_old(model, cache):
    """Test unfaithfulness that uses the entire explanation. 
    
    The old implementation had a bug such that all prediction vectors were combined and KL div 
    was then used instead of many KL divs and then averages...

    Args:
        model (_type_): Model to use
        cache (_type_): Cache to use

    Returns:
        _type_: The unfaithfulness calculation with random explanations
    """
    dataset = model.dataset

    test_idx = dataset.index_test #TODO Fiqure if test mask is best to use...
    
    org_preds = torch.zeros((test_idx.shape[0], dataset.num_classes)) 
    masked_preds = torch.zeros(org_preds.shape)
    random_preds = torch.zeros(org_preds.shape)
    for elemnum, graph_id in enumerate(test_idx):
        data = dataset[graph_id].to(model.device)
        tempData = data.clone().to(model.device)
        
        org_preds[elemnum] = model.get_prediction_vector(data)
        
        edge_mask, node_mask, permuted_edge_mask, permuted_node_mask, _ ,_ = get_explanation(cache, elemnum, model, data, graph_id)
        edge_mask = edge_mask.to(model.device)
        node_mask = node_mask.to(model.device)
        permuted_edge_mask = permuted_edge_mask.to(model.device)
        permuted_node_mask = permuted_node_mask.to(model.device)
        
        #Keep only subgraph
        mask_data_into_temp_data(tempData, data, node_mask, edge_mask)
        masked_preds[elemnum] = (model.get_prediction_vector(data = tempData))
        
        #Keep only subgraph
        mask_data_into_temp_data(tempData, data, permuted_node_mask, permuted_edge_mask)
        random_preds[elemnum] = (model.get_prediction_vector(data = tempData))
    
    org_preds = org_preds.cpu()
    masked_preds = masked_preds.cpu()
    random_preds = random_preds.cpu()
        
    org_softmax = F.softmax(org_preds, dim=-1)
    pert_softmax = F.softmax(masked_preds, dim=-1)
    random_softmax = F.softmax(random_preds, dim=-1)
    
    eps = eps=1e-7 #To float issues where softmax returns numbers that are seen as zero
    kl_div = F.kl_div((org_softmax+eps).log() , pert_softmax, reduction='batchmean')
    unfaith = 1 - float(torch.exp(-kl_div))
    
    kl_div = F.kl_div((org_softmax+eps).log() , random_softmax, reduction='batchmean')
    unfaith_random = 1 - float(torch.exp(-kl_div))
    

    
    ratio = unfaith/unfaith_random
    
    return unfaith, unfaith_random, ratio

def test_full_unfaithfulness(model, cache):
    """Test unfaithfulness that uses the entire explanation

    Args:
        model (_type_): Model to use
        cache (_type_): Cache to use

    Returns:
        _type_: The unfaithfulness calculation with random explanations
    """
    dataset = model.dataset

    test_idx = dataset.index_test #TODO Fiqure if test mask is best to use...
    
    org_preds = torch.zeros((test_idx.shape[0], dataset.num_classes)) 
    masked_preds = torch.zeros(org_preds.shape)
    random_preds = torch.zeros(org_preds.shape)
    for elemnum, graph_id in enumerate(test_idx):
        data = dataset[graph_id].to(model.device)
        tempData = data.clone().to(model.device)
        
        org_preds[elemnum] = model.get_prediction_vector(data)
        
        edge_mask, node_mask, permuted_edge_mask, permuted_node_mask, _ ,_ = get_explanation(cache, elemnum, model, data, graph_id)
        edge_mask = edge_mask.to(model.device)
        node_mask = node_mask.to(model.device)
        permuted_edge_mask = permuted_edge_mask.to(model.device)
        permuted_node_mask = permuted_node_mask.to(model.device)
        
        #Keep only subgraph
        mask_data_into_temp_data(tempData, data, node_mask, edge_mask)
        masked_preds[elemnum] = (model.get_prediction_vector(data = tempData))
        
        #Keep only subgraph
        mask_data_into_temp_data(tempData, data, permuted_node_mask, permuted_edge_mask)
        random_preds[elemnum] = (model.get_prediction_vector(data = tempData))
    
    org_preds = org_preds.cpu()
    masked_preds = masked_preds.cpu()
    random_preds = random_preds.cpu()
        
    unfaiths = torch.zeros(len(org_preds))
    unfaith_randoms = torch.zeros(len(org_preds))    
    
    for i in range(len(org_preds)):        
        org_softmax = F.softmax(org_preds[i], dim=-1)
        pert_softmax = F.softmax(masked_preds[i], dim=-1)
        random_softmax = F.softmax(random_preds[i], dim=-1)
        
        eps = eps=1e-7 #To float issues where softmax returns numbers that are seen as zero
        kl_div = F.kl_div((org_softmax+eps).log() , pert_softmax, reduction='batchmean')
        unfaiths[i] =  1 - float(torch.exp(-kl_div))
        
        kl_div = F.kl_div((org_softmax+eps).log() , random_softmax, reduction='batchmean')
        unfaith_randoms[i] = 1 - float(torch.exp(-kl_div))

    unfaith = unfaiths.mean()
    unfaith_random = unfaith_randoms.mean()
    
    ratio = unfaith/unfaith_random
    
    return unfaith, unfaith_random, ratio, unfaiths, unfaith_randoms
    
def test_explainer_edge_groundtruth(model, cache, manual_node_indicies = None, hops = 2):
    """ Compare the generated edge structure explanation to the ground truth explanation provided by the data.
        This requires that the data contains an edge_mask that specifices what edges are important

    Args:
        model (_type_): What model to use
        node_indicies (_type_): What nodes to test
        hops (int, optional): Size of the compared subgraph in k hops. Defaults to 2.

    Returns:
        - acuraccy
        - recall
        - precision
        - f1_score
        - auroc
    """
    dataset = model.dataset
    
    test_idx = dataset.index_test #TODO Fiqure if test mask is best to use...
        
    predictions = []
    targets = []
    for elemnum, graph_id in enumerate(test_idx):
        data = dataset[graph_id]
        
        _, _, _, _, predicted_edges ,_ = get_explanation(cache, elemnum, model, data, graph_id)
        predicted_edges = predicted_edges.cpu()
        
        #Apllying the hard mask is left to the model itself
        predictions.append(predicted_edges)
        targets.append(data.edge_mask.cpu())
    
    predictions = torch.concatenate((predictions), axis=0) 
    targets = torch.concatenate((targets), axis=0)
    
    accuracy, recall, precision, f1_score, auroc = groundtruth_metrics(predictions, targets)
    return accuracy, recall, precision, f1_score, auroc #Result of function could be returned directly, but this is easier to see
    
def test_fidelity_metric(model, cache):
    """Test fidelity+ and -

    Args:
        model (_type_): Model to test
        cache (_type_): Explanation cache
        manual_node_indicies (_type_, optional): Option to only use specific nodes, otherwise use test set. Defaults to None.

    Returns:
        _type_: _description_
    """
    dataset = model.dataset
    
    test_idx = dataset.index_test
        
    org_preds = torch.zeros((test_idx.shape[0], dataset.num_classes)) 
    subgraph_preds_neg = torch.zeros(org_preds.shape)
    subgraph_preds_pos = torch.zeros(org_preds.shape)
    subgraph_preds_neg_ran = torch.zeros(org_preds.shape)
    subgraph_preds_pos_ran = torch.zeros(org_preds.shape)
    
    for elemnum, graph_id in enumerate(test_idx):
        data = dataset[graph_id].to(model.device)
        tempData = data.clone().to(model.device)
        
        org_preds[elemnum] = model.get_prediction_vector(data)
        
        edge_mask, node_mask, permuted_edge_mask, permuted_node_mask, _ ,_ = get_explanation(cache, elemnum, model, data, graph_id)
        edge_mask = edge_mask.to(model.device)
        node_mask = node_mask.to(model.device)
        permuted_edge_mask = permuted_edge_mask.to(model.device)
        permuted_node_mask = permuted_node_mask.to(model.device)
        #edge_mask, node_mask, permuted_edge_mask, permuted_node_mask, _, _ = get_node_and_edge_mask(data, model, node_id)
        
        #Fidelity+ (Remove subgraph subgraph)
        #if node_mask !=None:
        #    tempData.x = data.x.clone() * (1-node_mask)
        #if edge_mask !=None:
        #    tempData.edge_index = data.edge_index.clone()[:, (edge_mask.logical_not())]

        mask_data_into_temp_data(tempData, data, (1-node_mask), edge_mask.logical_not())
        subgraph_preds_pos[elemnum] = (model.get_prediction_vector(data = tempData))
        
        mask_data_into_temp_data(tempData, data, (1-permuted_node_mask), permuted_edge_mask.logical_not())
        subgraph_preds_pos_ran[elemnum] = (model.get_prediction_vector(data = tempData))
        
        #Fidelity- (Keep only subgraph)
        #if node_mask !=None:
        #    tempData.x = data.x.clone() * node_mask
        #if edge_mask !=None:
        #    tempData.edge_index = data.edge_index.clone()[:, edge_mask]
        mask_data_into_temp_data(tempData, data, node_mask, edge_mask)
        subgraph_preds_neg[elemnum] = (model.get_prediction_vector(data = tempData))
        
        mask_data_into_temp_data(tempData, data, permuted_node_mask, permuted_edge_mask)
        subgraph_preds_neg_ran[elemnum] = (model.get_prediction_vector(data = tempData))
    
    ground_truth = dataset.y[test_idx].detach().cpu()    
    
    #Predictions using org graph
    preds = org_preds.max(1)[1]
    preds = preds.detach().cpu()
    
    #Predictions when using only important subgraph
    preds_subgraph_pos = subgraph_preds_pos.max(1)[1]
    preds_subgraph_pos = preds_subgraph_pos.detach().cpu()
    preds_subgraph_pos_ran = subgraph_preds_pos_ran.max(1)[1]
    preds_subgraph_pos_ran = preds_subgraph_pos_ran.detach().cpu()
    
    #Predictions when using only important subgraph
    preds_subgraph_neg = subgraph_preds_neg.max(1)[1]
    preds_subgraph_neg = preds_subgraph_neg.detach().cpu()
    preds_subgraph_neg_ran = subgraph_preds_neg_ran.max(1)[1]
    preds_subgraph_neg_ran = preds_subgraph_neg_ran.detach().cpu()
    
        
    neg_fidelity = ((preds == ground_truth).float() -
                        (preds_subgraph_neg == ground_truth).float()).abs().mean()
    
    neg_fidelity_ran = ((preds == ground_truth).float() -
                        (preds_subgraph_neg_ran == ground_truth).float()).abs().mean()    
    
    pos_fidelity = ((preds == ground_truth).float() -
                    (preds_subgraph_pos == ground_truth).float()).abs().mean()
    
    pos_fidelity_ran = ((preds == ground_truth).float() -
                    (preds_subgraph_pos_ran == ground_truth).float()).abs().mean()
    
    eps = eps=1e-7
    
    neg_fidelity_ratio = neg_fidelity / (neg_fidelity_ran+eps)
    pos_fidelity_ratio = pos_fidelity / (pos_fidelity_ran+eps)
    
    return pos_fidelity, neg_fidelity, pos_fidelity_ran, neg_fidelity_ran, pos_fidelity_ratio, neg_fidelity_ratio

def get_explanation(cache, cacheNum, model, data, graph_id):
    if cache != None:
        return cache[cacheNum]
    else:
        return get_node_and_edge_mask(data, model, graph_id) 
            
        
def mask_data_into_temp_data(tempData, data, node_mask, edge_mask):
    if node_mask !=None:
        tempData.x = data.x.clone() * node_mask
    if edge_mask !=None:
        tempData.edge_index = data.edge_index.clone()[:, edge_mask]
    