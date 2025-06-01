import cv2
import numpy as np

# --- Image Processing Utilities ---

# Load an image from file
def load_image(image_path):
    """
    Load an image from the given path using OpenCV.
    Returns the image as a NumPy array in BGR format.
    """
    return cv2.imread(image_path)


# Crop the Region of Interest (ROI) using bounding box coordinates
def crop_roi(image, bbox_x, bbox_y, bbox_width, bbox_height):
    """
    Crop an image using pixel bounding box coordinates.
    """
    x1 = int(bbox_x - bbox_width / 2)
    y1 = int(bbox_y - bbox_height / 2)
    x2 = int(bbox_x + bbox_width / 2)
    y2 = int(bbox_y + bbox_height / 2)

    return image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]


# Resize the image to standard dimensions (640x640)
def resize_image(image, size=(640, 640)):
    """
    Resize the image to the given size (default 640x640).
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# Normalize pixel values to [0, 255]
def normalize_image(image):
    """
    Normalize the image pixel values using cv2.normalize to the full 0–255 range.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


# Apply Gaussian blur to reduce noise
def denoise_image(image, kernel_size=(3, 3)):
    """
    Apply Gaussian blur to the image to reduce noise.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


# Apply CLAHE to enhance local contrast
def enhance_contrast(image):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Works only on grayscale or LAB channels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


# --- Background Removal ---

def remove_background_otsu(image):
    """
    Apply Otsu's thresholding to convert the image to binary.
    Works best on grayscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def remove_background_grabcut(image, iterations=5):
    """
    Use GrabCut algorithm for background removal.
    Assumes foreground is roughly centered.
    Returns a binary mask of the foreground.
    """
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define a rectangle around the likely foreground
    height, width = image.shape[:2]
    rect = (int(0.1*width), int(0.1*height), int(0.8*width), int(0.8*height))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)

    # Where sure or likely foreground, set to 1
    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    return binary_mask


def hybrid_segmentation(image):
    # Step 1: Convert to grayscale and apply Otsu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Prepare initial GrabCut mask
    grabcut_mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[otsu_mask == 255] = cv2.GC_PR_FGD  # Probable foreground from Otsu
    grabcut_mask[otsu_mask == 0] = cv2.GC_BGD       # Definite background from Otsu

    # Step 3: Run GrabCut with mask initialization
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, grabcut_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Step 4: Final binary mask
    result_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    return result_mask


# --- Morphological Operations ---

def apply_opening(mask, kernel_size=(3, 3)):
    """
    Apply morphological opening to remove noise (erosion followed by dilation).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def apply_closing(mask, kernel_size=(5, 5)):
    """
    Apply morphological closing to fill small holes (dilation followed by erosion).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# --- Final Mask Utility ---

def get_final_binary_mask(image):
    """
    Generates a clean binary silhouette mask of the fish using:
    - Background removal
    - Morphological opening and closing
    """
    mask = hybrid_segmentation(image)

    mask = apply_opening(mask)
    mask = apply_closing(mask)
    return mask

def fill_holes(mask):
    im_floodfill = mask.copy()
    h, w = mask.shape[:2]
    mask_flood = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask_flood, (0, 0), 0)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    filled = mask | im_floodfill_inv
    return filled
