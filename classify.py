from data.loader import load_yolo_dataset

if __name__ == "__main__":
    images = "./dataset/train/images/"
    labels = "./dataset/train/labels/"

    df = load_yolo_dataset(images, labels)

    print(df.head())

    # Optionally save
    df.to_csv("./outputs/fish_size_dataframe.csv", index=False)