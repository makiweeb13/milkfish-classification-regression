# data/loader.py

import os
import pandas as pd
from PIL import Image
from config.class_map import YOLO_CLASS_MAP

def load_yolo_dataset(images_dir, labels_dir):
    data = []

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        image_id = label_file.replace(".txt", "")
        label_path = os.path.join(labels_dir, label_file)
        image_path = os.path.join(images_dir, image_id + ".jpg")

        with Image.open(image_path) as img:
            img_width, img_height = img.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        for fish_id, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            bbox_width = width * img_width
            bbox_height = height * img_height
            bbox_x = x_center * img_width
            bbox_y = y_center * img_height

            original_class, mapped_class = YOLO_CLASS_MAP[class_id]

            data.append({
                "image_id": image_id,
                "fish_id": fish_id,
                "original_class": original_class,
                "mapped_class": mapped_class,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height
            })

    return pd.DataFrame(data)
