import gzip
import pickle

file_path = '/home/ubuntu/workspace/SignLanguage/data/phoenix-2014/phoenix-2014.train'

# Attempt to read as a gzip-compressed pickle
try:
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
        print("File successfully read as a gzip-compressed pickle.")
        # Print the first few entries to inspect
        print(data[:5])
except Exception as e:
    print(f"Failed to read the file as gzip-compressed pickle: {e}")
