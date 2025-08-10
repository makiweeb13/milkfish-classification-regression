# main.py
from classify import (
    extract_features,
    train_gradient_boosting,
    classify_gradient_boosting,
    train_random_forest,
    classify_random_forest,
    classify_ensemble_soft_voting,
    classify_ensemble_svm
)

from regress import (
    extract_features_weight
)

def main():
    # Comment/Uncomment functions as needed
    
    print("Starting size feature extraction for process...")
    # extract_features()
    print("Size feature extraction process completed.")

    print("Starting classification training process...")
    # train_gradient_boosting()
    # train_random_forest()
    print("Classification training process completed.")

    print("Starting classification process...")
    # classify_gradient_boosting()
    # classify_random_forest()
    # classify_ensemble_svm()
    print("Classification process completed.")

    print("Starting weight feature extraction process...")
    # extract_features_weight()
    print("Weight feature extraction process completed.")


if __name__ == "__main__":
    main()