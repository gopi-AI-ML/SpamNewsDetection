import os
import sys
import re
import pandas as pd

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Check if wordnet is already downloaded
try:
    nltk.data.find('corpora/wordnet.zip')
    print("Wordnet is already downloaded.")
except LookupError:
    print("Downloading wordnet...")
    nltk.download('wordnet')

# Check if stopwords are already downloaded
try:
    nltk.data.find('corpora/stopwords.zip')
    print("Stopwords are already downloaded.")
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("pickle_files", "fakeNews_Vectorizer.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_text(self, data):
        try:
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words("english"))
            data = data.lower()
            data = re.sub(r'[^\w\s]', " ", data)  # Remove punctuation
            data = re.sub("[^a-z]", " ", data)
            data = re.sub(r'\s+', " ", data).strip()
            words = data.split()    # Remove extra spaces
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            return " ".join(lemmatized_words)
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Running Data Trasnformation File...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Clean text data
            train_df["text"] = train_df["text"].apply(self.clean_text)
            test_df["text"] = test_df["text"].apply(self.clean_text)

            X_train = train_df["text"]
            y_train = train_df["label"]
            X_test = test_df["text"]
            y_test = test_df["label"]

            vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            X_train_transformed = vectorizer.fit_transform(X_train).toarray()
            X_test_transformed = vectorizer.transform(X_test).toarray()

            save_object(file_path= self.data_transformation_config.preprocessor_obj_file_path,  obj= vectorizer)
            logging.info("Data Transformation completed successfully!")

            return X_train_transformed, y_train, X_test_transformed, y_test, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    dt = DataTransformation()
    train_data, train_labels, test_data, test_labels, _ = dt.initiate_data_transformation("artifacts/train.csv", "artifacts/test.csv")