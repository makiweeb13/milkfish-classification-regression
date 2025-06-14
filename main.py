# main.py
from classify import (
    extract_features,
    classify
)

def main():
    
    print("Starting feature extraction process...")
    # extract_features()
    print("Feature extraction process completed.")

    print("Starting classification process...")
    classify()
    print("Classification process completed.")

if __name__ == "__main__":
    main()