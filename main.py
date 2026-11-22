
from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_build
from src.utils import logger


def main():
    try:
        logger.info("MLOps Pipeline Execution Started...")

        # Step 1: Data Ingestion
        df = data_ingestion()

        logger.info(f"Original Dataset Shape: {df.shape}")

        # Step 2: Data Preprocessing
        (
            X_train_pca,
            X_test_pca,
            y_train_resample,
            y_test
        ) = data_preprocessing(df)

        # Display Dataset Shapes
        logger.info(f"After PCA - X_train Shape: {X_train_pca.shape}")
        logger.info(f"After PCA - X_test Shape: {X_test_pca.shape}")

        logger.info(f"After SMOTE - y_train Shape: {y_train_resample.shape}")
        logger.info(f"Original y_test Shape: {y_test.shape}")

        # logger.success("MLOps Pipeline Execution Completed Successfully.")
        
        # Step 3: Model Building
        automl,automl_settings = model_build(X_train_pca,X_test_pca,y_train_resample,y_test)
        


    except Exception as e:
        logger.exception(f"Pipeline Execution Failed: {e}")
        raise


if __name__ == "__main__":
    main()
