# main.py
from classify import (
    extract_features,
    train_gradient_boosting,
    classify_gradient_boosting,
    train_random_forest,
    classify_random_forest,
    classify_ensemble_soft_voting
)

def main():
    # Comment/Uncomment functions as needed
    
    print("Starting feature extraction process...")
    # extract_features()
    print("Feature extraction process completed.")

    print("Starting training process...")
    train_gradient_boosting()
    train_random_forest()
    print("Training process completed.")

    print("Starting classification process...")
    classify_gradient_boosting()
    classify_random_forest()
    classify_ensemble_soft_voting()  
    print("Classification process completed.")

if __name__ == "__main__":
    main()