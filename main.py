# main.py
from classify import (
    extract_features,
    train,
    classify
)

def main():
    # Comment/Uncomment functions as needed
    
    print("Starting feature extraction process...")
    # extract_features()
    print("Feature extraction process completed.")

    print("Starting training process...")
    # train()
    print("Training process completed.")

    print("Starting classification process...")
    classify()
    print("Classification process completed.")

if __name__ == "__main__":
    main()