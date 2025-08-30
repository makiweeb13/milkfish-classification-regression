import os
import cv2
import numpy as np
from u2net.u2net_test import segment_with_u2net

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


def resize_with_padding(image, target_size=(640, 640), pad_color=0):
    """
    Resize the image to the target size with padding to maintain aspect ratio.

    Args:
        image (numpy.ndarray): Input image.
        target_size (tuple): Desired output size (width, height).
        pad_color (int): Padding color (default is 0 for black).

    Returns:
        numpy.ndarray: Resized image with padding.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale and new size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Add border
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded


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

def adaptive_threshold(image, block_size=11, C=2):
    """
    Apply adaptive thresholding to create a binary mask.
    Uses mean of neighborhood pixels minus a constant C.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, block_size, C)

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


# --- U2Net Segmentation ---
def segment_fish_u2net(image_path, output_dir):
    """
    Applies U2-Net segmentation on a single image or a folder of images.

    Args:
        image_path (str): Path to an image file or folder of images.
        output_path (str): Directory to save the predicted masks.
    """
    print(f"Segmenting: {image_path} -> {output_dir}")
    segment_with_u2net(
        image_dir=os.path.dirname(image_path),
        prediction_dir=output_dir,
        model_dir="u2net/saved_models/u2netp/u2netp.pth"
    )
    print("Segmentation complete.")


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
    mask = remove_background_otsu(image)

    # Apply morphological operations to clean up the mask
    mask = apply_opening(mask)
    mask = apply_closing(mask)
    return mask

def load_and_preprocess_images(df, images_input, images_output,  image_path):

    for index, row in df.iterrows():
        image_path = os.path.join(images_input, row["image_id"] + ".jpg")

        # Load the full image
        img = load_image(image_path)
        if img is None:
            print(f"Could not load image {image_path}")
            continue

        # Crop the fish ROI
        crop = crop_roi(img, row["bbox_x"], row["bbox_y"], row["bbox_width"], row["bbox_height"])

        # Apply normalization, denoising, contrast enhancement
        # crop = normalize_image(crop)
        # crop = denoise_image(crop)
        # crop = enhance_contrast(crop)
        # crop = get_final_binary_mask(crop)

        print(f"Processed {row['image_id']} fish {row['fish_id']} of class {row['mapped_class']}")

        # Save processed image crop
        save_path = os.path.join(
            images_output,
            f"{row['image_id']}_fish{row['fish_id']}_{row['mapped_class'].replace(' ', '')}.jpg"
        )
        cv2.imwrite(save_path, crop)


def resize_images_in_directory(input_dir, output_dir, size=(640, 640), pad_color=0):
    """
    Resize all images in the input directory to the specified size with padding,
    and save them to the output directory.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save resized images.
        size (tuple): Desired output size (width, height).
        pad_color (int or tuple): Padding color (default is 0 for black).
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            image = load_image(image_path)
            if image is None:
                print(f"Could not load image {image_path}")
                continue

            resized = resize_with_padding(image, target_size=size, pad_color=pad_color)

            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, resized)
            print(f"Resized and saved: {save_path}")