# main.py
from classify import (
    extract_features
)

def main():
    # This function can be used to call the classify function
    print("Starting feature extraction process...")
    extract_features()
    print("Feature extraction process completed.")

if __name__ == "__main__":
    main()