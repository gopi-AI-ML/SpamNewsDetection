import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    logging.info("Running main2.py")
    logging.info("Starting the end-to-end machine learning pipeline...")

    try:
        logging.info("Initialating Data Ingestion Process..........")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")

        logging.info("Data Tranformation started.........")
        data_transform = DataTransformation()
        train_data, train_labels, test_data, test_labels, vectorizer_path = data_transform.initiate_data_transformation("artifacts/train.csv", "artifacts/test.csv")
        logging.info(f"Data Tranformation completed. Vectorizer path: {vectorizer_path}")

        logging.info("Model Training started.......")
        model_trainer = ModelTrainer()
        best_model = model_trainer.train_and_evaluate(train_data, train_labels, test_data, test_labels)
        logging.info(f"Model training completed. model_path: {vectorizer_path}")

    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    main()