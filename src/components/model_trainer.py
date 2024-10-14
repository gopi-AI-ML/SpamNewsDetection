import os
import sys

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("pickle_files", "fake_news_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Running Model_trainer file....")
            # Initialize models
            models = {
                "MultinomialNB": MultinomialNB(),
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier()
            }

            best_model = None
            best_accuracy = 0
            
            for model_name, model in models.items():
                logging.info(f"Training the {model.__class__.__name__} Model.....")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)

                print(" ")
                print(report)
                print(" ")
                print(conf_matrix)
                print(" ")

                logging.info(f"{model_name} achieved an accuracy of {accuracy:.2f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info(f"Best model: {best_model.__class__.__name__} saved with accuracy of : {best_accuracy:.2f} and confusion matrix = {conf_matrix}")
            logging.info("Model Training Completed Successfully!!...")
            return best_model

        except Exception as e:
            raise CustomException(e, sys)