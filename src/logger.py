import logging
import logging
import os

# Define the logs folder and log file name
logs_folder = "logs"
LOG_FILE = "info.log" 

# Create the full path for the log file
LOG_FILE_PATH = os.path.join(logs_folder, LOG_FILE)

# Ensure the logs directory exists
os.makedirs(logs_folder, exist_ok=True)

# Set up the logging configuration
def setup_logging():
    logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    filemode="w"
)
    
setup_logging()

if __name__=="__main__":
    logging.info("bye bye...")