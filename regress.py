import os
from data.loader import load_weight_dataset
from utils.directories_utils import (
    train_weight_images, train_weight_labels, train_weight_output,
    valid_weight_images, valid_weight_labels, valid_weight_output,
    test_weight_images, test_weight_labels, test_weight_output,
    weight_train_data, weight_valid_data, weight_test_data, data_output
)
from utils.image_utils import segment_fish_u2net
from utils.extractor_utils import merge_features_with_csv

# Model imports
from models.gradient_boosting.train_gradient_boosting import regress_fish_with_gradient_boosting
from models.gradient_boosting.test_gradient_boosting import gradientBoostingRegressor

from models.random_forest.train_random_forest import regress_fish_with_random_forest
from models.random_forest.test_random_forest import randomForestRegressor

from models.train_regress_ensemble import regress_with_ensemble

def extract_features_weight():
    # Load YOLO dataset
    load_weight_dataset(train_weight_images, train_weight_labels, f"{data_output}{weight_train_data}")
    load_weight_dataset(valid_weight_images, valid_weight_labels, f"{data_output}{weight_valid_data}")
    load_weight_dataset(test_weight_images, test_weight_labels, f"{data_output}{weight_test_data}")

    # # Preprocess and segment train_images
    # print("Segmenting images...")
    # segment_fish_u2net(train_weight_images, train_weight_output)
    # segment_fish_u2net(valid_weight_images, valid_weight_output)
    # segment_fish_u2net(test_weight_images, test_weight_output)

    # Extract features and merge with existing CSV
    print("Extracting features...")
    features_csv_path = f"{data_output}{weight_train_data}"
    valid_features_csv_path = f"{data_output}{weight_valid_data}"
    test_features_csv_path = f"{data_output}{weight_test_data}"

    if not exists(features_csv_path) or not exists(valid_features_csv_path) or not exists(test_features_csv_path):
        return

    merge_features_with_csv(features_csv_path, train_weight_output, features_csv_path, "regress")
    merge_features_with_csv(valid_features_csv_path, valid_weight_output, valid_features_csv_path, "regress")
    merge_features_with_csv(test_features_csv_path, test_weight_output, test_features_csv_path, "regress")


def exists(path):
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        return False
    return True