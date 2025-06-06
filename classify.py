import os
import cv2
from data.loader import load_yolo_dataset
from utils.image_utils import (
    load_image,
    crop_roi,
    normalize_image,
    denoise_image,
    get_final_binary_mask,
    segment_fish_u2net
)


images = "./dataset/train/images/"
labels = "./dataset/train/labels/"
output_dir = "./outputs/fish_crops/"

os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_images(df, image_path):
    
    for index, row in df.iterrows():
        image_path = os.path.join(images, row["image_id"] + ".jpg")

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

        print(f"Processed {row['image_id']} fish {row['fish_id']} of class {row['mapped_class']}")

        # Save processed image crop
        save_path = os.path.join(
            output_dir,
            f"{row['image_id']}_fish{row['fish_id']}_{row['mapped_class'].replace(' ', '')}.jpg"
        )
        cv2.imwrite(save_path, crop)


if __name__ == '__main__':
    # All code that creates DataLoaders or runs multiprocessing
    df = load_yolo_dataset(images, labels)
    # load_and_preprocess_images(df, images)
    segment_fish_u2net(images, output_dir)
    print("All fish crops processed and saved.")