# main.py
from classify import (
    extract_features,
    classify_fish_with_gradient_boosting,
    gradientBoostingClassifier,
    classify_fish_with_random_forest,
    randomForestClassifier,
    classify_with_ensemble,
    ensemble_with_svm
)

from regress import (
    extract_features_weight,
    regress_fish_with_gradient_boosting,
    gradientBoostingRegressor
)

def main():
    # Comment/Uncomment functions as needed
    
    print("Starting size feature extraction for process...")
    # extract_features()
    print("Size feature extraction process completed.")

    print("Starting classification training process...")
    # classify_fish_with_gradient_boosting()
    # classify_fish_with_random_forest()
    print("Classification training process completed.")

    print("Starting classification process...")
    # gradientBoostingClassifier()
    # randomForestClassifier()
    # ensemble_with_svm()
    print("Classification process completed.")

    print("Starting weight feature extraction process...")
    # extract_features_weight()
    print("Weight feature extraction process completed.")

    print("Starting regression training process...")
    regress_fish_with_gradient_boosting()
    print("Regression training process completed.")

    print("Starting regression process...")
    gradientBoostingRegressor()
    print("Regression process completed.")


if __name__ == "__main__":
    main()