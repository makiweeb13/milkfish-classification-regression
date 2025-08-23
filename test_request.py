import requests
import os
import time

url = "http://localhost:8000/predict"
image_dir = "dataset/test/images"

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(image_dir, filename)
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            print(f"Result for {filename}: {response.json()}")
        time.sleep(0.5)
