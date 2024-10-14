import sys
import pandas as pd
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class SpamClassifier:
    def __init__(self, data_path, save_dataset_path="artifacts/mail_data.csv"):
        logging.info("Running spamMail.py file..")
        try:
            self.data_path = data_path
            self.save_dataset_path = save_dataset_path
            self.model = None
            self.vectorizer = None
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        try:
            # Load the dataset
            data = pd.read_csv(self.data_path, encoding="ISO-8859-1", usecols=["v1", "v2"])
            data["label"] = data['v1'].apply(lambda x: 1 if x == "spam" else 0)  # Fix: 0 for spam, 1 for ham
            data["text"] = data["v2"]
            data = data.drop(columns=["v1", "v2"], axis=1)
            data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset and reset index
            save_object(file_path=self.save_dataset_path, obj=data)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, data):
        try:
            # Define features and target variable
            X = data["text"]
            Y = data["label"]

            # Split the data into training and testing sets
            return train_test_split(X, Y, test_size=0.2, random_state=42)
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, Y_train):
        try:
            # Transform the text data to feature vectors
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True)
            x_train = self.vectorizer.fit_transform(X_train)

            # Create and train the model
            self.model = LogisticRegression()
            self.model.fit(x_train, Y_train)

            # Save the model and vectorizer
            save_object(file_path="pickle_files/spamMail_model.pkl", obj=self.model)
            save_object(file_path="pickle_files/spamMail_vectorizer.pkl", obj=self.vectorizer)
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, X_train, Y_train, X_test, Y_test):
        try:
            x_train = self.vectorizer.transform(X_train)
            x_test = self.vectorizer.transform(X_test)

            # Predictions and accuracy on training data
            train_prediction = self.model.predict(x_train)
            train_accuracy = accuracy_score(Y_train, train_prediction)
            print('Accuracy on training data: ', train_accuracy)

            # Predictions and accuracy on test data
            test_prediction = self.model.predict(x_test)
            test_accuracy = accuracy_score(Y_test, test_prediction)
            print('Accuracy on test data: ', test_accuracy)
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self):
        try:
            # Save the trained model and vectorizer
            save_object(obj=self.model, file_path="pickle_files/SpamMail_model.pkl")
            save_object(obj=self.vectorizer, file_path="pickle_files/spamMail_vectorizer.pkl")
            print("SpamMail Model and vectorizer saved successfully!")
        except Exception as e:
            raise CustomException(e, sys)
    
    def run():
        try:
            data_path = r"C:\Users\krish\Downloads\spam.csv"
            spam_classifier = SpamClassifier(data_path)

            data = spam_classifier.load_data()
            X_train, X_test, Y_train, Y_test = spam_classifier.preprocess_data(data)
            spam_classifier.train_model(X_train, Y_train)
            spam_classifier.evaluate_model(X_train, Y_train, X_test, Y_test)
            spam_classifier.save_model()
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    try:
        SpamClassifier.run()
    except Exception as e:
        raise CustomException(e, sys)