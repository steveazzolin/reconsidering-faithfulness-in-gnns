from models.gisst_gc import Gisst, GisstConfig
from dataset_loader_gc import load_dataset
from sacred import Experiment
from sacred.observers import MongoObserver
from torch_geometric.seed import seed_everything
from utils.gc_metrics_helper import run_metrics
from global_config import GlobalConfig as gl


ex = Experiment('GISST')

if gl.use_omniboard:
  ex.observers.append(MongoObserver.create(url=gl.mongo_url,
                      db_name='db'))

@ex.config
def cfg():
  dataset = "bbbp"
  config = {
            'modelname' : 'GISST_gc',
            'num_hidden_layers' : 3,
            'hidden_dims' : 32,
            'lr' : 0.001,
            'epochs' : 250,
            'batch_size' : 32,
            'weight_decay' : 0.0005,
            'dropout_rates' : 0,
            'x_l1_coeffs' : 0.005,
            'x_ent_coeffs' : 0.01,
            'edge_l1_coeffs' : 0.005,
            'edge_ent_coeffs' : 0.01,
            'verbose' : False,
        }
  test_explanation = False
  test_faithfulness = True
  test_full_faithfullness = True
  test_fidelity = True

@ex.named_config
def bbbp_optimized():
  dataset = "bbbp"
  config = {
            'modelname' : 'GISST_gc',
            'num_hidden_layers' : 3,
            'hidden_dims' : 32,
            'lr' : 0.001,
            'epochs' : 500,
            'batch_size' : 32,
            'weight_decay' : 0.0005,
            'dropout_rates' : 0.1,
            'x_l1_coeffs' : 0.005,
            'x_ent_coeffs' : 0.01,
            'edge_l1_coeffs' : 0.005,
            'edge_ent_coeffs' : 0.1,
            'verbose' : False,
        }
  test_explanation = False
  test_faithfulness = True
  test_full_faithfullness = True
  test_fidelity = True

@ex.named_config
def ba2motif_optimized():
  dataset = "ba2motif"
  config = {
            'modelname' : 'GISST_gc',
            'num_hidden_layers' : 3,
            'hidden_dims' : 16,
            'lr' : 0.001,
            'epochs' : 500,
            'weight_decay' : 0.0000,
            'dropout_rates' : 0.0,
            'x_l1_coeffs' : 0.000,
            'x_ent_coeffs' : 0.00,
            'edge_l1_coeffs' : 0.00,
            'edge_ent_coeffs' : 0.0,
            'verbose' : False 
        }
  test_explanation = True
  test_faithfulness = True
  test_full_faithfullness = True
  test_fidelity = True
  
@ex.named_config
def bamultishapes_optimized():
  dataset = "bamultishapes"
  config = {
            'modelname' : 'GISST_gc',
            'num_hidden_layers' : 3,
            'hidden_dims' : 16,
            'lr' : 0.001,
            'epochs' : 500,
            'weight_decay' : 0.0000,
            'dropout_rates' : 0.0,
            'x_l1_coeffs' : 0.0,
            'x_ent_coeffs' : 0.0,
            'edge_l1_coeffs' : 0.0,
            'edge_ent_coeffs' : 0.0,
            'verbose' : False 
        }
  test_explanation = True
  test_faithfulness = True
  test_full_faithfullness = True
  test_fidelity = True
  
@ex.named_config
def mutag_optimized():
  dataset = "mutag"
  config = {
            'modelname' : 'GISST_gc',
            'num_hidden_layers' : 3,
            'hidden_dims' : 32,
            'lr' : 0.001,
            'epochs' : 500,
            'batch_size' : 32,
            'weight_decay' : 0.0005,
            'dropout_rates' : 0.1,
            'x_l1_coeffs' : 0.005,
            'x_ent_coeffs' : 0.01,
            'edge_l1_coeffs' : 0.005,
            'edge_ent_coeffs' : 0.1,
            'verbose' : False,
        }
  test_explanation = False
  test_faithfulness = True
  test_full_faithfullness = True
  test_fidelity = True
  
@ex.named_config
def enzymes_optimized():
  dataset = "enzymes"
  config = {
            'modelname' : 'GISST_gc',
            'num_hidden_layers' : 3,
            'hidden_dims' : 32,
            'lr' : 0.005,
            'epochs' : 2000,
            'batch_size' : 32,
            'weight_decay' : 0.0005,
            'dropout_rates' : 0,
            'x_l1_coeffs' : 0.005,
            'x_ent_coeffs' : 0.01,
            'edge_l1_coeffs' : 0.005,
            'edge_ent_coeffs' : 0.01,
            'verbose' : False,
        }
  test_explanation = False
  test_faithfulness = True
  test_full_faithfullness = True
  test_fidelity = True
  

@ex.automain
def run(dataset, config, seed, _run, test_explanation, test_faithfulness, test_full_faithfullness, test_fidelity):
  #Load dataset before seeding as dataset loading changes seed
  #Only required for GC
  _dataset = load_dataset(dataset) 
  print(ex)
  print(dataset)
  print(seed)
  seed_everything(seed)
  
  configClass = GisstConfig(config_dict=config)
  print(config)
  print(_run)
  gisst = Gisst(_dataset, dataset, config = configClass, _run = _run)

  return run_metrics(gisst, test_explanation, test_faithfulness, test_full_faithfullness, test_fidelity)