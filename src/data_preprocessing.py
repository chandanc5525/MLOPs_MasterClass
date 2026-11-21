from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from imblearn.over_sampling import SMOTE 
from src.utils import logger 
from pathlib import Path
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.decomposition import PCA
import yaml 


# Define root_path 
root_path = Path(__file__).resolve().parent.parent 
def data_preprocessing(df):
    try:
        logger.info('Data Preprocessing Started....')
        # Define config_path 
        config_path = root_path/'config.yaml'
        
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)
            
        # Define target and drop_cols 
        target = config['data']['target']
        drop_col = config['data']['drop_cols']
        
        # Step1: Drop the duplicated values from the dataset
        df.drop_duplicates(inplace = True)
        
        # Step2: Split the dataset into X {Indpendent} and y {Dependent}
        X = df.drop(columns = [target] + drop_col)
        y = df[target]
        
        # Step3:Split the data into Seen and Unseen i.e. train and test 
        X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                        test_size = config['data']['test_size'],
                                                        random_state= config['data']['random_state']
                                                        )
        # Step4: Segregate the data into numerical columns and categorical columns 
        numerical_col = X_train.select_dtypes(exclude = 'object').columns 
        categorical_col = X_train.select_dtypes(include = 'object').columns 
        
        logger.success('Data splitting completed successfully. The transformation pipeline is now starting.')
        
        # Step5: Build numerical_pipeline and categorical_pipeline
        numerical_pipeline = Pipeline(steps=[('imputation',KNNImputer(n_neighbors=5)),
                                            ('Scaling',MinMaxScaler())
                                            ]
                                    )
        
        categorical_pipeline = Pipeline(steps = [('imputation',SimpleImputer(strategy = 'most_frequent')),
                                                ('encoding',OneHotEncoder(handle_unknown='ignore'))
                                                ]
                                        )
        # Step6: Use ColumnTransformer to concat numerical and categorical pipeline
        
        transformer = ColumnTransformer(
            [
            ('Numerical_Pipe',numerical_pipeline,numerical_col),
            ('Categorical_Pipe',categorical_pipeline,categorical_col)
            ]
            )       
       
        # Step7: Build another pipeline for SMOTE and Dimensional Reductionality 
       
        X_train_preprocess = transformer.fit_transform(X_train) # Seen Data Transformation
        X_test_preprocess  = transformer.transform(X_test)  # Unseen Data Transformation 

        # Step8: Use SMOTE Technique to balance the target column
        sm = SMOTE(random_state = config['data']['random_state'])
        X_train_resample,y_train_resample = sm.fit_resample(X_train_preprocess,
                                                   y_train)
        
        # Step9: Use PCA (Dimensional Reduction Technique)
        pca = PCA(n_components = config['data']['n_components'])
        
        X_train_pca = pca.fit_transform(X_train_resample) # Seen Data Transformation
        X_test_pca = pca.transform(X_test_preprocess)
        
        logger.success('Dimensional Reductionality Method Completed...')
        return X_train_pca,X_test_pca,y_train_resample,y_test
        
    except Exception as e:
        logger.exception(f'Exception Occured please check code {e}')
        raise             
