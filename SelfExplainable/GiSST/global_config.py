# Global config file for shared settings for all models
class GlobalConfig:
        #Omniboard
        use_omniboard = False #Disabled as sacred will crash if the url does not work.
        mongo_url = f'mongodb://mongouser:password@11.111.11.111:37017/?authMechanism=SCRAM-SHA-1'
        
        #Folder names
        dataset_dir = 'datasets'
        checkpoint_dir = 'checkpoints'
        results_dir = 'results'
        benchmarks_results_dir = 'benchmarks'
        
        #Metric names - used in dictionaries
        test_accuracy = 'accuracy'

        explain_accuracy= 'explain accuracy'
        explain_recall = 'explain recall'
        explain_precision = 'explain precision'
        explain_f1_score = 'explain f1_score'
        explain_auroc = 'explain auroc'
        
        feature_unfaithfulness = 'unfaithfulness'
        full_unfaithfulness = 'full unfaithfulness'
        random_full_unfaithfulness = 'random full unfaithfulness'
        random_ratio_full_unfaithfulness = 'relative full unfaithfulness'
        full_unfaithfulness_data = 'full unfaithfulness data'
        random_full_unfaithfulness_data = 'random full unfaithfulness data'
        
        full_unfaithfulness_old = 'full unfaithfulness_old'
        random_full_unfaithfulness_old = 'random full unfaithfulness_old'
        random_ratio_full_unfaithfulness_old = 'relative full unfaithfulness_old'
        
         
        fidelity_plus = 'fidelity+'
        fidelity_minus = 'fidelity-'
        random_fidelity_plus = 'random fidelity+'
        random_fidelity_minus = 'random fidelity-'
        fidelity_plus_ratio = 'fidelity+ ratio'
        fidelity_minus_ratio = 'fidelity- ratio'
        
        training_time = "training time"
        test_set_explanation_time = "explanation time"
        
        
        
 