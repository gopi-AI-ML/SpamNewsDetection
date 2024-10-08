import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler("log_files.log"),
            logging.StreamHandler()
        ]
    )
    
setup_logging()

if __name__=="__main__":
    logging.info("bye bye...")