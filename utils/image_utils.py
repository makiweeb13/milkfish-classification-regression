import cv2

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
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge and convert back to BGR
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)