import re
import tkinter as tk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tkinter import messagebox
from src.utils import load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from user_interface import SpamDetector  # Make sure this matches your actual file structure

class SpamDetector:
    def __init__(self):
        # Load the vectorizer and model
        self.vectorizer = load_object("pickle_files/preprocessor.pkl")
        self.model = load_object("pickle_files/best_model.pkl")

    def clean_text(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub("[^a-z]", " ", text)
        text = re.sub(r'\s+', " ", text).strip()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(lemmatized_words)

    def predict(self, text):
        cleaned_text = self.clean_text(text)
        text_vectorized = self.vectorizer.transform([cleaned_text]).toarray()
        prediction = self.model.predict(text_vectorized)
        return prediction[0]

class SpamDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam News Detector")
        self.root.geometry("600x600")
        self.root.configure(bg="#F0F4C3")  # Light yellow background color

        # Initialize the spam detector
        self.spam_detector = SpamDetector()

        # Title Frame
        self.title_frame = tk.Frame(root, bg="#81C784", bd=2)
        self.title_frame.pack(fill="x")

        self.title_label = tk.Label(self.title_frame, text="Spam News Detector", bg="#81C784", fg="white", font=("Helvetica", 20, "bold"))
        self.title_label.pack(pady=10)

        # Button Frame for Training and Testing
        self.button_frame = tk.Frame(root, bg="#F0F4C3", bd=2)
        self.button_frame.pack(pady=20)

        self.train_button = tk.Button(self.button_frame, text="Train Model", command=self.train_model, bg="#FF9800", fg="white", font=("Helvetica", 12, "bold"), bd=0, padx=10, pady=5)
        self.train_button.pack(side="left", padx=10)

        self.test_button = tk.Button(self.button_frame, text="Test for Spam", command=self.show_test_frame, bg="#388E3C", fg="white", font=("Helvetica", 12, "bold"), bd=0, padx=10, pady=5)
        self.test_button.pack(side="right", padx=10)

        # Input Frame for Testing
        self.input_frame = tk.Frame(root, bg="#F0F4C3", bd=2)
        self.input_frame.pack(pady=20)

        self.label = tk.Label(self.input_frame, text="Enter News Article Text:", bg="#F0F4C3", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.text_entry = tk.Text(self.input_frame, height=10, width=50, bg="#ffffff", fg="#333333", font=("Helvetica", 12), bd=2, relief="groove")
        self.text_entry.pack(pady=10)

        # Predict Button for Testing
        self.predict_button = tk.Button(self.input_frame, text="Predict", command=self.predict, bg="#388E3C", fg="white", font=("Helvetica", 12, "bold"), bd=0, padx=10, pady=5)
        self.predict_button.pack()

        # Result Frame
        self.result_frame = tk.Frame(root, bg="#F0F4C3", bd=2)
        self.result_frame.pack(pady=20)

        self.result_label = tk.Label(self.result_frame, text="", font=("Helvetica", 14, "bold"), bg="#F0F4C3")
        self.result_label.pack(pady=10)

        # Footer Frame
        self.footer_frame = tk.Frame(root, bg="#F0F4C3", bd=2)
        self.footer_frame.pack(side="bottom", pady=5)

        self.footer_label = tk.Label(self.footer_frame, text="Spam News Detection App", bg="#F0F4C3", font=("Helvetica", 10, "italic"))
        self.footer_label.pack()

        # Initially hide the input and result frames
        self.input_frame.pack_forget()
        self.result_frame.pack_forget()

    def train_model(self):
        try:
            # Initialize data ingestion to load the datasets
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()  # Load your data
            
            # Initialize data transformation
            data_transform = DataTransformation()
            X_train, y_train, X_test, y_test, vectorizer_path = data_transform.initiate_data_transformation(train_data_path, test_data_path)
            
            # Now that we have our datasets, we can train the model
            model_trainer = ModelTrainer()
            best_model = model_trainer.train_and_evaluate(X_train, y_train, X_test, y_test)  # Pass the datasets here
            
            messagebox.showinfo("Training Complete", "The model has been trained successfully!")
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred while training: {str(e)}")

    def show_test_frame(self):
        # Show the testing input frame
        self.input_frame.pack(pady=20)
        self.result_frame.pack(pady=20)

    def predict(self):
        user_input = self.text_entry.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter valid text!")
            return

        try:
            prediction = self.spam_detector.predict(user_input)  # Use the instance variable
            result_message = "The Given Text is SPAM !!!" if prediction == 1 else "Not Spam"
            self.result_label.config(text=result_message, fg="#D32F2F" if prediction == 1 else "#388E3C")  # Red for spam, green for not spam
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()