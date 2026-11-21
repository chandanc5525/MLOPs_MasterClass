import pandas as pd 
from src.utils import logger 
from pathlib import Path 
import os 
import sys
import yaml 

# Define root_path 
root_path = Path(__file__).resolve().parent.parent

def data_ingestion():
    try: 
        logger.info('Data Ingestion Started....')
        # Define config_path 
        config_path = root_path/"config.yaml" 
        
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)
            
        # Define data_path 
        data_path = config['data']['path']
        
        # Load the dataset using pandas functionality        
        df = pd.read_csv(data_path) 
        
        output_path = root_path / "data/interim/ingested.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        logger.success("Data ingestion completed")
        
        return df 

    except FileNotFoundError as e:
        logger.exception(f'File Not Found Error please check configuration....{e}')
        raise
    
        
