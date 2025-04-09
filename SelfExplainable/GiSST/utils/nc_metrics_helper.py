from metrics.nc_metrics import test_feature_unfaithfulness, test_explainer_edge_groundtruth, test_fidelity_metric, test_full_unfaithfulness, test_full_unfaithfulness_old
from utils.utils import get_node_and_edge_mask, train_model_with_time
from torch_geometric.utils import mask_to_index
import global_config as gl
import time

def run_metrics(model, test_explanation = False, 
                test_faithfulness = False,
                test_full_faithfullness = False,
                test_fidelity = False,
                include_random = True,
                cache = True):
    """Helper method to run tests for NC

    Args:
        model (_type_): The model to use already combined with a dataset
        test_explanation (bool, optional): Test explanation ground truth. Requires that the dataset has an edge ground truth. Defaults to False.
        test_faithfulness (bool, optional): Test top-k features in terms of faithfulness. Not used in the thesis but is more alligned with the original implementation. Defaults to False.
        test_full_faithfullness (bool, optional): The unfaithfulness metric used in the thesis. Uses full explanations. Defaults to False.
        test_fidelity (bool, optional): Test fidelity. Defaults to False.
        include_random (bool, optional): Include random explanations in fidelity and full unfaithfulness tests. Defaults to True.
        cache (bool, optional): If the method should cache explanation results to avoid recalculating them. Can save a lot of time if explanations take a while to calculate. Defaults to True.

    Returns:
        Dictionary: Dictionary with results using keys from the global config.
    """
    results = {}
    
    #Also trains the model
    results[gl.GlobalConfig.training_time] = train_model_with_time(model)
    
    #Cache all explanations to avoid calculating them multiple times
    cached = None
    if cache:
      t1 = time.time()
      cached = []
      data = model.data
      test_idx = mask_to_index(data.test_mask).cpu().numpy()
      for elemnum, node_id in enumerate(test_idx):
        cached.append(get_node_and_edge_mask(data, model, node_id))
      t2 = time.time()
      results[gl.GlobalConfig.test_set_explanation_time] = t2-t1
  
    results[gl.GlobalConfig.test_accuracy] = model.test()
    
    if(test_explanation and model.data.get("edge_mask") != None):
      accuracy, recall, precision, f1_score, auroc = test_explainer_edge_groundtruth(model, cached) 
      results[gl.GlobalConfig.explain_accuracy] = accuracy
      results[gl.GlobalConfig.explain_recall] = recall
      results[gl.GlobalConfig.explain_precision] = precision
      results[gl.GlobalConfig.explain_f1_score] = f1_score
      results[gl.GlobalConfig.explain_auroc] = auroc
      
    if(test_faithfulness == True):
      results[gl.GlobalConfig.feature_unfaithfulness + '-0.10'] = test_feature_unfaithfulness(model, cached, top_k = 0.1)
      #We define 25% as the standard feature unfaithfullness test
      results[gl.GlobalConfig.feature_unfaithfulness + '-0.25'] = results[gl.GlobalConfig.feature_unfaithfulness] = test_feature_unfaithfulness(model, cached, top_k = 0.25)
      results[gl.GlobalConfig.feature_unfaithfulness + '-0.50'] = test_feature_unfaithfulness(model, cached, top_k = 0.5)
      results[gl.GlobalConfig.feature_unfaithfulness + '-0.75'] = test_feature_unfaithfulness(model, cached, top_k = 0.75)
      results[gl.GlobalConfig.feature_unfaithfulness + '-1'] = test_feature_unfaithfulness(model, cached, top_k=1.0)
      
    if(test_full_faithfullness):
      (results[gl.GlobalConfig.full_unfaithfulness],
      results[gl.GlobalConfig.random_full_unfaithfulness],
      results[gl.GlobalConfig.random_ratio_full_unfaithfulness],
      results[gl.GlobalConfig.full_unfaithfulness_data],
      results[gl.GlobalConfig.random_full_unfaithfulness_data]) = test_full_unfaithfulness(model, cached)
      
      (results[gl.GlobalConfig.full_unfaithfulness_old],
      results[gl.GlobalConfig.random_full_unfaithfulness_old],
      results[gl.GlobalConfig.random_ratio_full_unfaithfulness_old]) = test_full_unfaithfulness_old(model, cached)
      #if(include_random):
      #   = test_full_unfaithfulness(model, random=True)
      
    if(test_fidelity):
      (results[gl.GlobalConfig.fidelity_plus],
      results[gl.GlobalConfig.fidelity_minus],
      results[gl.GlobalConfig.random_fidelity_plus],
      results[gl.GlobalConfig.random_fidelity_minus],
      results[gl.GlobalConfig.fidelity_plus_ratio],
      results[gl.GlobalConfig.fidelity_minus_ratio]) = test_fidelity_metric(model, cached)

    return results
  