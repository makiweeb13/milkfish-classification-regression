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
segment_fish_u2net(image_path, output_path)
```
- U²-Net is an open-source pre-trained model used for segmentation.
- Achieves much higher segmentation accuracy.
---

## Image Processing Pipeline (classify.py)
For each fish:
- Load image.
- Run image through U²-Net for automatic segmentation.
  - The model outputs a binary mask highlighting the fish
- Save the segmented mask for visualization or morphometric feature extraction.
