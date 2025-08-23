import requests

url = "http://localhost:8000/predict"

files = {
    'file': open('weight_dataset/test/images/IMG_2455_jpg.rf.86a65051965901a67ba987b92d87aae8.jpg', 'rb')
}

response = requests.post(url, files=files)

print(response.json())
