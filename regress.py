import os
import cv2
import pandas as pd
from utils.directories_utils import (
    train_weight_images, train_weight_labels, train_weight_output,
    valid_weight_images, valid_weight_labels, valid_weight_output,
    test_weight_images, test_weight_labels, test_weight_output,
    # weight_train_data, weight_valid_data, weight_test_data, data_weight_output
)
from utils.image_utils import load_image, crop_roi, normalize_image, denoise_image, get_final_binary_mask

def load_and_preprocess_train_weight_images(df):
    for index, row in df.iterrows():
        image_path = os.path.join(train_weight_images, row["image_id"] + ".jpg")

        # Load the full image
        img = load_image(image_path)
        if img is None:
            print(f"Could not load image {image_path}")
            continue

        # Crop the fish ROI
        crop = crop_roi(img, row["bbox_x"], row["bbox_y"], row["bbox_width"], row["bbox_height"])

        # Apply normalization, denoising, contrast enhancement
        crop = normalize_image(crop)
        crop = denoise_image(crop)
        # crop = enhance_contrast(crop)
        crop = get_final_binary_mask(crop)

        print(f"Processed {row['image_id']}")

        # Save processed image crop
        save_path = os.path.join(
            train_weight_output,
            f"{row['image_id']}_processed.jpg"
        )
        cv2.imwrite(save_path, crop)

def exists(path):
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        return False
    return True