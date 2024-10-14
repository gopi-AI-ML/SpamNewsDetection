import os
import sys
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
import threading
from src.components.model_trainer import ModelTrainer
from src.utils import load_object
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

class SpamDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Spam Mail and Fake News Detector Application")
        self.master.geometry("900x600")
        self.master.resizable(True, True)

        # Main frame with deep background color
        self.main_frame = tk.Frame(master, bg="#20B2AA")
        self.main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # Main heading label
        self.main_heading = tk.Label(self.main_frame, text="Spam Mail and Fake News Detection Model", font=("Helvetica", 20, 'bold'), bg="#20B2AA", fg="#FFFFFF")
        self.main_heading.pack(pady=10)

        # Spam frame and Fake news frame with green and blue color themes
        self.spam_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED, padx=10, pady=10, bg="#00FA9A")
        self.spam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fake_news_frame = tk.Frame(self.main_frame, bd=2, relief=tk.RAISED, padx=10, pady=10, bg="#007BFF")
        self.fake_news_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Ensure necessary directories exist
        self.check_and_create_directories()

        self.model_trainer = ModelTrainer()
        self.fake_news_model_path = "pickle_files/fake_news_model.pkl"
        self.fake_news_model_vectorizer_path = "pickle_files/fakeNews_Vectorizer.pkl"
        self.spam_mail_model_path = "pickle_files/spamMail_model.pkl"
        self.spam_mail_vectorizer_path = "pickle_files/spamMail_Vectorizer.pkl"

        # Label for spam model frame
        self.spam_label = tk.Label(self.spam_frame, text="SPAM MAIL DETECTION", font=("Helvetica", 16, 'bold'), bg="#00FA9A", fg="#000000")
        self.spam_label.pack(pady=10)

        # Buttons for spam model
        self.train_spam_mail_button = tk.Button(self.spam_frame, text="Train Spam MAIL Model", command=self.start_training_spam_mail, width=20, bg="white", fg="black", font=("Helvetica", 12, 'bold'))
        self.train_spam_mail_button.pack(pady=10)

        self.test_spam_mail_button = tk.Button(self.spam_frame, text="Enter Your Mail", command=self.test_spam_mail_model, width=20, bg="white", fg="black", font=("Helvetica", 12, 'bold'))
        self.test_spam_mail_button.pack(pady=10)

        # Label for fake news model frame
        self.fake_news_label = tk.Label(self.fake_news_frame, text="NEWS DETECTION", font=("Helvetica", 16, 'bold'), bg="#007BFF", fg="#FFFFFF")
        self.fake_news_label.pack(pady=10)

        # Buttons for fake news model
        self.train_fake_news_button = tk.Button(self.fake_news_frame, text="Train Fake NEWS Model", command=self.start_training_fake_news, width=20, bg="white", fg="black", font=("Helvetica", 12, 'bold'))
        self.train_fake_news_button.pack(pady=10)

        self.test_fake_news_button = tk.Button(self.fake_news_frame, text="Enter Your News", command=self.test_fake_news_model, width=20, bg="white", fg="black", font=("Helvetica", 12, 'bold'))
        self.test_fake_news_button.pack(pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)

        # Text widget for displaying training logs
        self.log_area = scrolledtext.ScrolledText(
            self.main_frame,
            width=80,
            height=20,
            state='normal',
            wrap=tk.WORD,
            bg="#CCCCFF",  # Dark background color
            fg="black",  # Lavender text color
            font=("Courier", 10)
        )
        self.log_area.pack(pady=10)

    def check_and_create_directories(self):
        os.makedirs('artifacts', exist_ok=True)
        os.makedirs('pickle_files', exist_ok=True)

    def start_training_spam_mail(self):
        self.train_spam_mail_button.config(state='disabled')
        thread = threading.Thread(target=self.train_spam_mail_model)
        thread.start()

    def start_training_fake_news(self):
        self.train_fake_news_button.config(state='disabled')
        thread = threading.Thread(target=self.train_fake_news_model)
        thread.start()

    def train_spam_mail_model(self):
        self.log_area.delete(1.0, tk.END)  # Clear log area
        self.progress_bar.start()
        self.log_area.insert(tk.END, "Training of SPAM MAIL model started...\n")
        self.master.update()

        try:
            from src.components.spamMail import SpamClassifier
            SpamClassifier.run()

            self.log_area.insert(tk.END, "SPAM MAIL Model trained successfully!\n")
            messagebox.showinfo("Success", "SPAM MAIL Model trained successfully!")
        except Exception as e:
            self.log_area.insert(tk.END, f"ERROR: {str(e)}")  # Provide specific error message
            messagebox.showerror("Error", f"ERROR: {str(e)}")  # Provide specific error message
        finally:
            self.progress_bar.stop()
            self.train_spam_mail_button.config(state='normal')

    def train_fake_news_model(self):
        self.log_area.delete(1.0, tk.END)  # Clear log area
        self.progress_bar.start()
        self.log_area.insert(tk.END, "Training FAKE NEWS model started...\n")
        self.master.update()

        try:
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            X_train, X_test, Y_train, Y_test, _ = data_transformation.initiate_data_transformation("artifacts/train.csv", "artifacts/test.csv")

            trainer = ModelTrainer()
            trainer.train_and_evaluate(X_train, X_test, Y_train, Y_test)

            self.log_area.insert(tk.END, "FAKE NEWS Detection Model trained successfully!\n")
            messagebox.showinfo("Success", "FAKE NEWS Model trained successfully!")
        except Exception as e:
            self.log_area.insert(tk.END, f"ERROR: {str(e)}")  # Provide specific error message
            messagebox.showerror("Error", f"ERROR: {str(e)}")  # Provide specific error message
        finally:
            self.progress_bar.stop()
            self.train_fake_news_button.config(state='normal')

    def test_spam_mail_model(self):
        if not os.path.exists(self.spam_mail_model_path) or not os.path.exists(self.spam_mail_vectorizer_path):
            messagebox.showwarning("Warning", "SPAM MAIL model is not trained yet. Please train the model first!")
            return

        try:
            user_input = simpledialog.askstring("Input", "Enter the message to test:")
            if user_input:
                vectorizer = load_object(self.spam_mail_vectorizer_path)
                trained_model = load_object(self.spam_mail_model_path)

                input_data = vectorizer.transform([user_input]).toarray()
                prediction = trained_model.predict(input_data)

                result = "The given text is SPAM!!" if prediction[0] == 1 else "the text is NOT SPAM"
                messagebox.showinfo("Prediction Result", f"The message is: {result}")
        except CustomException as e:
            self.log_area.insert(tk.END, f"ERROR: {str(e)}")
            self.master.update()

    def test_fake_news_model(self):
        if not os.path.exists(self.fake_news_model_path):
            messagebox.showwarning("Warning", "FAKE NEWS model is not trained yet. Please train the model first!")
            return

        try:
            user_input = simpledialog.askstring("Input", "Enter the news to test:")
            if user_input:
                vectorizer = load_object(self.fake_news_model_vectorizer_path)  # Corrected variable name
                trained_model = load_object(self.fake_news_model_path)

                input_data = vectorizer.transform([user_input]).toarray()
                prediction = trained_model.predict(input_data)

                result = "FAKE" if prediction[0] == 1 else "REAL"
                messagebox.showinfo("Prediction Result", f"The given news is: {result}")
        except CustomException as e:
            self.log_area.insert(tk.END, f"ERROR: {str(e)}")
            self.master.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()