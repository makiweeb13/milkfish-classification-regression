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

## Image Processing and Segmentation Pipeline (U²-Net)
For each fish:
- Load image and save image details into ```outputs/data/```:
  - ```test_fish_size_dataframe.csv```
  - ```train_fish_size_dataframe.csv```
  - ```valid_fish_size_dataframe.csv```
- Run image through **U²-Net** for automatic segmentation.
  - The model outputs a binary mask highlighting the fish
- Save the segmented mask for morphometric feature extraction into ```outputs/masks/```:
  - ```size_test_masks```
  - ```size_train_masks```
  - ```size_valid_masks```
- Extract morphometric and normalize features
- Overwrite existing csv files with extracted data

## Classification Process (classify.py)
- Train ensemble learning models, **Gradient Boosting(GB) and Random Forest(RF)**:
  - **Gradient Boosting**
    - Train and validate GB using train and valid sets ```train_gradient_boosting.py```
    - Save model and label encoder ```gradient_boosting_classifier.pkl, label_encoder.pkl```
    - Test model and output predictions and their probabilities ```test_gradient_boosting.py```
  - **Random Forest**
    - Train and validate RF using train and valid sets `train_random_forest.py`
    - Save model and label encoder `random_forest_classifier.pkl`
    - Test model and output predictions and their probabilities `test_random_forest.py`

## Ensemble Methods

### Classification Ensembles
1. **Support Vector Machine (SVM)**
   - Trained on combined predictions from GB and RF
   - Uses probability outputs as feature vectors
   - Helps in finding optimal decision boundary between classes

2. **Soft Voting Classifier**
   - Combines GB and RF predictions using weighted average of probabilities
   - Weights determined through cross-validation
   - Final prediction based on highest weighted probability

### Regression Ensemble
1. **Weighted Averaging**
   - Combines weight predictions from GB and RF models
   - Weights determined by model performance on validation set
   - Final weight estimate calculated as:
     ```
     final_weight = (α × GB_prediction) + (β × RF_prediction)
     ```
     where α and β are optimized weights summing to 1

## Model Performance
- Classification metrics include:
  - Accuracy: 95%
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

- Regression metrics include:
  - Root Mean Square Error (RMSE): ~140g
  - Mean Absolute Error (MAE)
  - R-squared (R²)

## API Implementation
The trained models are hosted using FastAPI, providing endpoints for:
- Size classification prediction
- Weight estimation
- Model performance metrics

API endpoints:
```
POST /predict
```

## Mobile Application
The models are integrated into a mobile application for easy access and visualization:
- Repository: [thesis-app](https://github.com/makiweeb13/thesis-app)
- Features:
  - Capture fish images
  - Results visualization

## Output Structure
```
outputs/
├── data/
│   ├── test_fish_size_dataframe.csv
│   ├── train_fish_size_dataframe.csv
│   └── valid_fish_size_dataframe.csv
├── masks/
│   ├── size_test_masks/
│   ├── size_train_masks/
│   └── size_valid_masks/
└── models/
    ├── gradient_boosting_classifier.pkl
    ├── random_forest_classifier.pkl
    ├── label_encoder.pkl
    ├── svm_ensemble.pkl
    └── voting_ensemble.pkl
```
