import os,sys
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
import threading
from src.components.model_trainer import ModelTrainer
from src.utils import load_object, save_object
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion  # Importing DataIngestion
from src.components.data_transformation import DataTransformation  # Importing DataTransformation

class SpamDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Spam and Fake News Detector")
        self.master.geometry("900x600")
        self.master.resizable(True, True)

        # Create main frame
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Create frames for spam and fake news models
        self.spam_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED, padx=10, pady=10)
        self.spam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fake_news_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED, padx=10, pady=10)
        self.fake_news_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Ensure necessary directories exist
        self.check_and_create_directories()

        self.model_trainer = ModelTrainer()
        self.fake_news_model_path = "pickle_files/fake_news_model.pkl"
        self.fake_news_model_vectorizer = "pickle_files/fakeNews_Vectorizer.pkl"
        self.spam_mail_model_path = "pickle_files/spamMail_model.pkl"
        self.spam_mail_vectorizer_path = "pickle_files/spamMail_Vectorizer.pkl"

        # Label for spam model frame
        self.spam_label = tk.Label(self.spam_frame, text="SPAM MAIL Model", font=("Helvetica", 16))
        self.spam_label.pack(pady=10)

        # Buttons for spam model
        self.train_spam_mail_button = tk.Button(self.spam_frame, text="MAIL DETECTION", command=self.start_training_spam_mail, width=20, bg='lightgreen')
        self.train_spam_mail_button.pack(pady=10)

        self.test_spam_mail_button = tk.Button(self.spam_frame, text="Enter Your mail..", command=self.test_spam_mail_model, width=20, bg='lightcoral')
        self.test_spam_mail_button.pack(pady=10)

        # Label for fake news model frame
        self.fake_news_label = tk.Label(self.fake_news_frame, text="NEWS DETECTION", font=("Helvetica", 16))
        self.fake_news_label.pack(pady=10)

        # Buttons for fake news model
        self.train_fake_news_button = tk.Button(self.fake_news_frame, text="Train FAKE NEWS Model", command=self.start_training_fake_news, width=20, bg='lightblue')
        self.train_fake_news_button.pack(pady=10)

        self.test_fake_news_button = tk.Button(self.fake_news_frame, text="Enter Your News..", command=self.test_fake_news_model, width=20, bg='lightyellow')
        self.test_fake_news_button.pack(pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)

        # Text widget for displaying training logs
        self.log_area = scrolledtext.ScrolledText(self.main_frame, width=80, height=20, state='normal', wrap=tk.WORD)
        self.log_area.pack(pady=10)

    def check_and_create_directories(self):
        # Create required directories if they don't exist
        os.makedirs('artifacts', exist_ok=True)
        os.makedirs('pickle_files', exist_ok=True)

    def start_training_spam_mail(self):
        # Start the training for the spam mail model
        self.train_spam_mail_button.config(state='disabled')  # Disable button to prevent multiple clicks
        thread = threading.Thread(target=self.train_spam_mail_model)
        thread.start()

    def start_training_fake_news(self):
        # Start the training for the fake news model
        self.train_fake_news_button.config(state='disabled')  # Disable button to prevent multiple clicks
        thread = threading.Thread(target=self.train_fake_news_model)
        thread.start()

    def train_spam_mail_model(self):
        self.log_area.delete(1.0, tk.END)  # Clear the log area
        self.progress_bar.start()  # Start the progress bar
        self.log_area.insert(tk.END, "Training SPAM MAIL model started...\n")
        self.master.update()  # Update the GUI

        try:
            # Load and train the spam mail model
            from src.spamMail import SpamClassifier  # Ensure SpamClassifier is in spamMail.py
            spam_classifier = SpamClassifier(r"C:\Users\krish\Downloads\spam.csv")
            data = spam_classifier.load_data()
            X_train, X_test, Y_train, Y_test = spam_classifier.preprocess_data(data)
            
            # Train model and save it
            spam_classifier.train_model(X_train, Y_train)

            # Save the trained model and vectorizer
            save_object(self.spam_mail_model_path, spam_classifier.model)
            save_object(self.spam_mail_vectorizer_path, spam_classifier.vectorizer)

            spam_classifier.evaluate_model(X_train, Y_train, X_test, Y_test)  # Evaluation step
            self.log_area.insert(tk.END, "SPAM MAIL Model trained successfully!\n")
            messagebox.showinfo("Success", "SPAM MAIL Model trained successfully!")
        except Exception as e:
            self.log_area.insert(tk.END, f"An error occurred during training: {str(e)}\n")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")
        finally:
            self.progress_bar.stop()  # Stop the progress bar
            self.train_spam_mail_button.config(state='normal')  # Re-enable button

    def train_fake_news_model(self):
        self.log_area.delete(1.0, tk.END)  # Clear the log area
        self.progress_bar.start()  # Start the progress bar
        self.log_area.insert(tk.END, "Training FAKE NEWS model started...\n")
        self.master.update()  # Update the GUI

        try:
            # Load data for fake news
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()
            
            # Transform data            
            data_transformation = DataTransformation()
            X_train, X_test, Y_train, Y_test,_ = data_transformation.initiate_data_transformation("artifacts/train.csv", "artifacts/test.csv")
            
            # Train the model 
            trainer = ModelTrainer()
            trainer.train_and_evaluate(X_train, X_test, Y_train, Y_test)

            self.log_area.insert(tk.END, "FAKE NEWS Model trained successfully!\n")
            messagebox.showinfo("Success", "FAKE NEWS Model trained successfully!")
        except Exception as e:
            self.log_area.insert(tk.END, f"An error occurred during training: {str(e)}\n")
            raise CustomException(e,sys)
            messagebox.showerror(e)
        finally:
            self.progress_bar.stop()  # Stop the progress bar
            self.train_fake_news_button.config(state='normal')  # Re-enable button


    def test_spam_mail_model(self):
        # Check if the spam mail model exists
        if not os.path.exists(self.spam_mail_model_path) or not os.path.exists(self.spam_mail_vectorizer_path):
            messagebox.showwarning("Warning", "SPAM MAIL model is not trained yet. Please train the model first!")
            return

        try:
            # Prompt user for input text
            user_input = simpledialog.askstring("Input", "Enter the message to test:")
            if user_input:
                # Load the vectorizer and model
                vectorizer = load_object(self.spam_mail_vectorizer_path)
                trained_model = load_object(self.spam_mail_model_path)

                # Transform the input data
                input_data = vectorizer.transform([user_input]).toarray()

                # Predict using the trained model
                prediction = trained_model.predict(input_data)

                # Show the prediction result
                result = "SPAM" if prediction[0] == 1 else "NOT SPAM"
                messagebox.showinfo("Prediction Result", f"The message is: {result}")

        except CustomException as e:
            self.log_area.insert(tk.END, f"An error occurred during testing: {str(e)}\n")
            self.master.update()

    def test_fake_news_model(self):
        # Check if the fake news model exists
        if not os.path.exists(self.fake_news_model_path):
            messagebox.showwarning("Warning", "FAKE NEWS model is not trained yet. Please train the model first!")
            return

        try:
            # Prompt user for input text
            user_input = simpledialog.askstring("Input", "Enter the news to test:")
            if user_input:
                # Load the vectorizer and model
                vectorizer = load_object(self.fake_news_model_vectorizer)
                trained_model = load_object(self.fake_news_model_path)

                # Transform the input data
                input_data = vectorizer.transform([user_input]).toarray()

                # Predict using the trained model
                prediction = trained_model.predict(input_data)

                # Show the prediction result
                result = "FAKE" if prediction[0] == 1 else "REAL"
                messagebox.showinfo("Prediction Result", f"The news is: {result}")

        except CustomException as e:
            self.log_area.insert(tk.END, f"An error occurred during testing: {str(e)}\n")
            self.master.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()