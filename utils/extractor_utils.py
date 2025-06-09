import os
import cv2
import numpy as np
import pandas as pd

def extract_morphometrics(mask_path):
    # Load mask (binary image)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Threshold to ensure binary (just in case)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours (external only, assuming one fish per mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No fish found in mask.")

    cnt = max(contours, key=cv2.contourArea)  # Pick the largest contour (the fish)

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # Convex hull (optional)
    hull = cv2.convexHull(cnt)

    # Area
    area = cv2.contourArea(cnt)

    # Perimeter
    perimeter = cv2.arcLength(cnt, True)

    # Convex hull area
    hull_area = cv2.contourArea(hull)

    # Aspect ratio
    aspect_ratio = float(w) / h

    # Solidity: contour area / convex hull area
    solidity = float(area) / hull_area if hull_area != 0 else 0

    # Circularity: 4π × area / perimeter²
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

    return {
        "length": h,  # height of bounding box
        "width": w,   # width of bounding box
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "circularity": circularity,          
        "convex_hull_area": hull_area
    }

def merge_features_with_csv(existing_csv_path, mask_dir, output_csv_path):
    df = pd.read_csv(existing_csv_path)

    # Extract features for each image_id in the CSV
    features_list = []
    for _, row in df.iterrows():
        image_id = row['image_id']
        mask_path = os.path.join(mask_dir, f"{image_id}.png")
        features = extract_morphometrics(mask_path)

        if features:
            features_list.append(features)
        else:
            # Add NaNs if features could not be extracted
            features_list.append({k: np.nan for k in [
                "length", "width", "area", "perimeter",
                "aspect_ratio", "solidity", "circularity", "convex_hull_area"
            ]})

    features_df = pd.DataFrame(features_list)
    merged_df = pd.concat([df, features_df], axis=1)
    
     # Drop unnecessary columns
    merged_df = merged_df.drop(columns=['image_id', 'fish_id', 'original_class'])

    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged CSV saved to: {output_csv_path}")