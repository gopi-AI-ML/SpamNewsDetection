import os

def list_files_in_directory(directory):
    try:
        # List all files in the specified directory
        files = os.listdir(directory)

        # Filter out only files (ignore subdirectories)
        file_names = [file for file in files ]
        return file_names
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage
directory_path = 'C:\GitHub\Spam_news_detection'
file_names = list_files_in_directory(directory_path)

print("Files in directory:")
for file_name in file_names:
    print(file_name)
