from flaml import AutoML 
from src.utils import logger 
from pathlib import Path 
import yaml

# Define root_path
root_path = Path(__file__).resolve().parent.parent 

def model_build(X_train_pca,X_test_pca,y_train_resample,y_test):
    try:
        # Define configuration file
        config_path = root_path/'config.yaml'
        with open(config_path,'r') as file:
            config = yaml.safe_load(file) 
            
        # Build Model with AutoML Pipeline
        automl = AutoML()
        
        automl_settings = {
            'time_budget': config['model']['time_budget'],
            'metric': config['model']['metric'],
            'task':config['model']['task'],
            'estimator_list':config['model']['estimator_list'],
            'eval_method':config['model']['eval_method'],
            'n_splits': config['model']['n_splits'],
            'seed':config['data']['random_state'],
            'verbose': config['model']['verbose']     
        }
     
        logger.success('Machine Learning Automation Pipeline Excecuted....')
        
        return automl,automl_settings
    except Exception as e:
        logger.exception(f'Machine Learning Automation Pipeline Failed.... {e}') 
        raise 
    



