import os

def read_reuters_files(directory_path='../data/reuters/training'):
    """
    Read all files from the Reuters training directory
    
    Args:
        directory_path (str): Path to the Reuters training directory
        
    Returns:
        list: List of tuples containing (filename, content)
    """
    documents = []
    
    try:
        # Get list of all files in directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                try:
                    # Read the content of each file
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        documents.append((filename, content))
                except Exception as e:
                    print(f"Error reading file {filename}: {str(e)}")
                    
    except Exception as e:
        print(f"Error accessing directory: {str(e)}")
        
    return documents

# Example usage
if __name__ == "__main__":
    files_content = read_reuters_files()
    print(f"Total documents read: {len(files_content)}")