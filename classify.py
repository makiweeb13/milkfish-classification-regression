import os
import cv2
from data.loader import load_yolo_dataset
from utils.image_utils import segment_fish_u2net, load_and_preprocess_images, resize_images_in_directory
from utils.directories_utils import (
    train_images, train_labels, train_output,
    valid_images, valid_labels, valid_output,
    test_images, test_labels, test_output,
    size_train_data, size_valid_data, size_test_data, data_output,
    size_train_cropped, size_valid_cropped, size_test_cropped
)
from utils.extractor_utils import merge_features_with_csv

# Model imports
from models.gradient_boosting.train_gradient_boosting import classify_fish_with_gradient_boosting
from models.gradient_boosting.test_gradient_boosting import gradientBoostingClassifier

from models.random_forest.train_random_forest import classify_fish_with_random_forest
from models.random_forest.test_random_forest import randomForestClassifier

from models.svm.ensemble_svm import ensemble_with_svm
from models.train_classify_ensemble import classify_with_ensemble

os.makedirs(train_output, exist_ok=True)
os.makedirs(valid_output, exist_ok=True)
os.makedirs(test_output, exist_ok=True)


def extract_features():
    # Load YOLO dataset
    train_df = load_yolo_dataset(train_images, train_labels, f"{data_output}{size_train_data}")
    valid_df = load_yolo_dataset(valid_images, valid_labels, f"{data_output}{size_valid_data}")
    test_df = load_yolo_dataset(test_images, test_labels, f"{data_output}{size_test_data}")

    # # Crop and preprocess train_images
    # print("Loading and preprocessing train images...")
    # load_and_preprocess_images(train_df, train_images, size_train_cropped, f"{data_output}{size_train_data}")
    # load_and_preprocess_images(valid_df, valid_images, size_valid_cropped, f"{data_output}{size_valid_data}")
    # load_and_preprocess_images(test_df, test_images, size_test_cropped, f"{data_output}{size_test_data}")

    # # Preprocess and segment train_images
    # print("Segmenting images...")
    # segment_fish_u2net(size_train_cropped, train_output)
    # segment_fish_u2net(size_valid_cropped, valid_output)
    # segment_fish_u2net(size_test_cropped, test_output)

    # # Resize images
    # print("Resizing images...")
    # resize_images_in_directory(train_output, train_output)
    # resize_images_in_directory(valid_output, valid_output)
    # resize_images_in_directory(test_output, test_output)

    # Extract features and merge with existing CSV
    print("Extracting features...")
    features_csv_path = f"{data_output}{size_train_data}"
    valid_features_csv_path = f"{data_output}{size_valid_data}"
    test_features_csv_path = f"{data_output}{size_test_data}"

    if not exists(features_csv_path) or not exists(valid_features_csv_path) or not exists(test_features_csv_path):
        return

    merge_features_with_csv(features_csv_path, train_output, features_csv_path, "classify")
    merge_features_with_csv(valid_features_csv_path, valid_output, valid_features_csv_path, "classify")
    merge_features_with_csv(test_features_csv_path, test_output, test_features_csv_path, "classify")


def exists(path):
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        return False
    return True