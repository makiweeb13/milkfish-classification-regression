import os
import cv2
import joblib
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
from .directories_utils import (
    saved_class_scaler, saved_regress_scaler
)

def extract_morphometrics(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No fish found in mask.")

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull_area = cv2.contourArea(hull)

    aspect_ratio = float(w) / h
    solidity = float(area) / hull_area if hull_area != 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    extent = float(area) / (w * h) if w * h != 0 else 0
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area != 0 else 0
    compactness = area / (perimeter ** 2) if perimeter != 0 else 0

    try:
        (x_, y_), (MA, ma), angle = cv2.fitEllipse(cnt)
        eccentricity = np.sqrt(1 - (MA ** 2 / ma ** 2)) if ma != 0 else 0
        elongation = ma / MA if MA != 0 else 0
    except:
        eccentricity = 0
        elongation = 0

    max_dist = 0
    for i in range(len(cnt)):
        for j in range(i + 1, len(cnt)):
            dist = np.linalg.norm(cnt[i][0] - cnt[j][0])
            if dist > max_dist:
                max_dist = dist
    feret_diameter = max_dist

    return {
        "length": h,
        "width": w,
        "area": area,
        "perimeter": perimeter,
        "solidity": solidity,
        "convex_hull_area": hull_area,
        "equivalent_diameter": equivalent_diameter,
        "feret_diameter": feret_diameter,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity
    }

def normalize_features(df, mode):
    feature_cols = df.columns.drop(["mapped_class"], errors='ignore')
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    if mode == "classify":
        joblib.dump(scaler, saved_class_scaler)
    elif mode == "regress":
        joblib.dump(scaler, saved_regress_scaler)
    return df

def merge_features_with_csv(existing_csv_path, mask_dir, output_csv_path, mode):
    df = pd.read_csv(existing_csv_path)
    features_list = []

    for idx, row in df.iterrows():
        image_id = row['image_id']
        print(f"[{idx + 1}/{len(df)}] Extracting features from: {image_id}.png")
        # Use glob to find the mask file with possible suffixes
        mask_pattern = os.path.join(mask_dir, f"{image_id}*.png")
        mask_files = glob.glob(mask_pattern)
        if mask_files:
            mask_path = mask_files[0]
        else:
            mask_path = None
        try:
            features = extract_morphometrics(mask_path) if mask_path else {k: np.nan for k in [
                "length", "width", "area", "perimeter", "solidity",
                "convex_hull_area", "equivalent_diameter", "feret_diameter",
                "aspect_ratio", "circularity"
            ]}
        except:
            features = {k: np.nan for k in [
                "length", "width", "area", "perimeter", "solidity",
                "convex_hull_area", "equivalent_diameter", "feret_diameter",
                "aspect_ratio", "circularity"
            ]}
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    features_df = normalize_features(features_df, mode)
    merged_df = pd.concat([df, features_df], axis=1)
    merged_df = merged_df.drop(columns=['image_id', 'fish_id', 'original_class', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'], errors='ignore')
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged CSV saved to: {output_csv_path}")
