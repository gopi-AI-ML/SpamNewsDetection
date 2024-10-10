import os
import sys
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
import threading
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_object, save_object
from src.exception import CustomException

class SpamDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Spam Detector")
        self.master.geometry("600x500")
        
        # Create main frame
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(padx=10, pady=10)

        # Ensure necessary directories exist
        self.check_and_create_directories()

        self.model_trainer = ModelTrainer()
        self.trained_model_path = "pickle_files/best_model.pkl"  # Path for the trained model
        self.vectorizer_path = "pickle_files/preprocessor.pkl"  # Path for the preprocessor

        # Buttons for training and testing
        self.train_button = tk.Button(self.main_frame, text="Train SPAM Model", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5, pady=10)

        self.test_button = tk.Button(self.main_frame, text="Test SPAM Model", command=self.test_model)
        self.test_button.grid(row=0, column=1, padx=5, pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.main_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=2, padx=5, pady=10)

        # Text widget for displaying training logs
        self.log_area = scrolledtext.ScrolledText(self.main_frame, width=60, height=15, state='normal', wrap=tk.WORD)
        self.log_area.grid(row=2, column=0, columnspan=2, padx=5, pady=10)

    def check_and_create_directories(self):
        # Create required directories if they don't exist
        os.makedirs('artifacts', exist_ok=True)
        os.makedirs('pickle_files', exist_ok=True)

    def start_training(self):
        # Start the training in a separate thread
        self.train_button.config(state='disabled')  # Disable button to prevent multiple clicks
        thread = threading.Thread(target=self.train_model)
        thread.start()

    def train_model(self):
        self.log_area.delete(1.0, tk.END)  # Clear the log area
        self.progress_bar.start()  # Start the progress bar
        try:
            # Data ingestion
            self.log_area.insert(tk.END, "Starting data ingestion...\n")
            self.master.update()  # Update the GUI
            
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Data transformation
            self.log_area.insert(tk.END, "Starting data transformation...\n")
            self.master.update()
            
            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            # Model training
            self.log_area.insert(tk.END, "Training the model...\n")
            self.master.update()
            
            best_model = self.model_trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
            self.log_area.insert(tk.END, "SPAM Model trained successfully!\n")
            self.master.update()
            messagebox.showinfo("Success", "SPAM Model application trained successfully!")

        except CustomException as e:
            # Show error message only during training
            self.log_area.insert(tk.END, f"An error occurred during training: {str(e)}\n")
            self.master.update()
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")
        finally:
            self.progress_bar.stop()  # Stop the progress bar
            self.train_button.config(state='normal')  # Re-enable button

    def test_model(self):
        # Check if the model exists
        if not os.path.exists(self.trained_model_path) or not os.path.exists(self.vectorizer_path):
            messagebox.showwarning("Warning", "SPAM model is not trained yet. Please train the model first!")
            return

        try:
            # Prompt user for input text
            user_input = simpledialog.askstring("Input", "Enter the message to test:")
            if user_input:
                # Load the vectorizer
                vectorizer = load_object(self.vectorizer_path)

                # Transform the input data
                input_data = vectorizer.transform([user_input]).toarray()

                # Load the trained model
                trained_model = load_object(self.trained_model_path)

                # Predict using the trained model
                prediction = trained_model.predict(input_data)

                # Show the prediction result
                result = "SPAM" if prediction[0] == 1 else "NOT SPAM"
                messagebox.showinfo("Prediction Result", f"The message is: {result}")

        except CustomException as e:
            # No message box shown here
            self.log_area.insert(tk.END, f"An error occurred during testing: {str(e)}\n")
            self.master.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamDetectorApp(root)
    root.mainloop()