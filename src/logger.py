import logging
import os

# Define the logs folder and log file name
logs_folder = "logs"
LOG_FILE = "info.log"  # Ensure the log file name is a string with a .log extension

# Create the full path for the log file
LOG_FILE_PATH = os.path.join(logs_folder, LOG_FILE)

# Ensure the logs directory exists
os.makedirs(logs_folder, exist_ok=True)

# Set up the logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    filemode="w"
)

if __name__ == "__main__":
    # Log messages to test
    logging.info("bye bye ...")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
