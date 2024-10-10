import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.utils import save_object  # Import the save_object function

class SpamClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.vectorizer = None

    def load_data(self):
        # Load the dataset
        data = pd.read_csv(self.data_path, encoding="ISO-8859-1", usecols=["v1", "v2"])
        data['label'] = np.where(data['v1'] == 'spam', 1, 0)
        data = data.drop(columns=["v1"], axis=1)
        return data.sample(frac=True)  # Shuffle the dataset

    def preprocess_data(self, data):
        # Define features and target variable
        X = data["v2"]
        Y = data["label"]

        # Split the data into training and testing sets
        return train_test_split(X, Y, test_size=0.2, random_state=42)

    def train_model(self, X_train, Y_train):
        # Transform the text data to feature vectors
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        x_train = self.vectorizer.fit_transform(X_train)

        # Create and train the model
        self.model = LogisticRegression()
        self.model.fit(x_train, Y_train)

    def evaluate_model(self, X_train, Y_train, X_test, Y_test):
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

    def save_model(self):
        # Save the trained model and vectorizer
        save_object(self.model, "pickle_files/Spam_model.pkl")
        save_object(self.vectorizer, "pickle_files/spam_preprocessor.pkl")
        print("Model and vectorizer saved successfully!")

if __name__ == "__main__":
    data_path = r"C:\Users\krish\Downloads\spam.csv"
    spam_classifier = SpamClassifier(data_path)

    data = spam_classifier.load_data()
    X_train, X_test, Y_train, Y_test = spam_classifier.preprocess_data(data)
    spam_classifier.train_model(X_train, Y_train)
    spam_classifier.evaluate_model(X_train, Y_train, X_test, Y_test)
    spam_classifier.save_model()