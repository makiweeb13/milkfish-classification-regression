# Milkfish Size Classification and Weight Estimation using Ensemble Learning

---

## Objective
To analyze individual milkfish from aquaculture images, classify their size class, and estimate their weight through image processing and ensemble learning techniques.

---

## Installation

Run the following command to install required libraries:

```
pip install opencv-python numpy pandas scikit-learn scikit_image matplotlib torch torchvision
```

## Dataset Preparation
- **Format**: Pre-labeled YOLOv5 format dataset with .txt annotation files and YAML configuration.
- **Mapping**:
  - Class 0: "adult" → Class A (Large)
  - Class 1: "semi-adult" → Class B (Medium)
  - Class 2: "juvenile" → Class C (Small)

**Processing Logic**:
- Parse each YOLO annotation.
- Convert relative coordinates to absolute pixel values using image dimensions.
- Generate a Pandas DataFrame with fields:
  - image_id, fish_id, original_class, mapped_class, bbox_x, bbox_y, bbox_width, bbox_height

---

## Image Utilities (utils/image_utils.py)
Implemented functions:

1. **Image Loading**
```python
load_image(path: str) -> np.ndarray
```

2. **ROI Cropping**
```python
crop_roi(image: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray
```

3. **Normalization**
```python
normalize_image(image: np.ndarray) -> np.ndarray
```

4. **Noise Reduction**
```python
denoise_image(image: np.ndarray) -> np.ndarray
```

5. **Contrast Enhancement (CLAHE)**(to remove)
```python
enhance_contrast(image: np.ndarray) -> np.ndarray
```

6. **Background Removal (Otsu)**
```python
remove_background_otsu(image: np.ndarray) -> np.ndarray
```

7. **Background Removal (GrabCut)**
```python
remove_background_grabcut(image: np.ndarray, iterations: int) -> np.ndarray
```

8. **Hybrid Segmentation Function**
```python
hybrid_segmentation(image: np.ndarray, iterations: int = 5) -> np.ndarray
```
- Combines denoising, CLAHE, Otsu thresholding, and GrabCut segmentation with morphological refinement.
- Addresses shape accuracy (Otsu) and hole reduction (GrabCut).

9. **U²-Net Pre-trained Segmentation Model**
```python
segment_fish_u2net(image_path: str, output_path: str)
```
- U²-Net is an open-source pre-trained model used for segmentation.
- Achieves much higher segmentation accuracy.
---

## Feature Extraction Utilities (utils/extractor_utils.py)
Implemented functions:

1. **Extract Morphometric Features**
```python
extract_morphometrics(path: str) -> dict
```
- Features extracted per fish include:
  - Length
  - Width
  - Area
  - Perimeter
  - Aspect Ratio
  - Solidity
  - Circularity
  - Convex Hull Area

2. **Merge Into Existing DataFrame**
```python
merge_features_with_csv(existing_csv_path: str, mask_dir: str, output_csv_path: str)
```

3. **Normalize Features**
```python
normalize_features(df: pd.DataFrame) -> pd.DataFrame
```

## Image Processing Pipeline
For each fish:
- Load image and save image details into (outputs/data/):
  - test_fish_size_dataframe.csv
  - train_fish_size_dataframe.csv
  - valid_fish_size_dataframe.csv
- Run image through U²-Net for automatic segmentation.
  - The model outputs a binary mask highlighting the fish
- Save the segmented mask for morphometric feature extraction into (outputs/masks/):
  - size_test_masks
  - size_train_masks
  - size_valid_masks
- Extract morphometric and normalize features
- Overwrite existing csv files with extracted data

## Classification Process (classify.py)
- Train ensemble learning models, **Gradient Boosting(GB) and Random Forest(RF)**:
  **Gradient Boosting**
  - Train and validate GB using train and valid sets (train_gradient_boosting.py)
  - Save model and label encoder (gradient_boosting_classifier.pkl, label_encoder.pkl)
  - Test model and output predictions and their probabilities (test_gradient_boosting.py)
