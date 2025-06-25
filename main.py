# main.py
from classify import (
    extract_features,
    train,
    classify,
    train_random_forest,
    classify_random_forest
)

def main():
    # Comment/Uncomment functions as needed
    
    print("Starting feature extraction process...")
    # extract_features()
    print("Feature extraction process completed.")

    print("Starting training process...")
    # train()
    train_random_forest()
    print("Training process completed.")

    print("Starting classification process...")
    # classify()
    classify_random_forest()  # Test Random Forest
    print("Classification process completed.")

if __name__ == "__main__":
    main()