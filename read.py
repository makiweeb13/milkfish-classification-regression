import cv2 as cv


def read_image(image_path):
    """
    Reads an image from the specified path and returns it as a numpy array.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        numpy.ndarray: The image read from the file.
    """
    image = cv.imread(image_path)
    cv.imshow("Milkfish", image)
    cv.waitKey(0)

    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image


read_image("dataset/train/images/IMG_20230503_104019_jpg.rf.5ca1e2ff0f801a6c40fe25b3b032867c.jpg")