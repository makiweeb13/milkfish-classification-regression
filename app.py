import joblib
import os
import pandas as pd
import numpy as np
import shutil
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils.image_utils import segment_fish_u2net
from utils.extractor_utils import extract_morphometrics, normalize_features
from utils.directories_utils import (
    save_gradient_boosting_model, save_random_forest_model,
    regressor_gradient_boosting_model, regressor_random_forest_model,
    classify_svm_meta, regress_ensemble, save_label_encoder
)

app = FastAPI(title="Milkfish Class and Weight Prediction API")

# Load models and encoders
gb_classifier = joblib.load(save_gradient_boosting_model)
rf_classifier = joblib.load(save_random_forest_model)
ensemble_svm_classifier = joblib.load(classify_svm_meta)
gb_regressor = joblib.load(regressor_gradient_boosting_model)
rf_regressor = joblib.load(regressor_random_forest_model)
ensemble_regressor = joblib.load(regress_ensemble)
label_encoder = joblib.load(save_label_encoder)

# Create a new folder for the image
folder_path = "uploaded_image"
output_folder = "segmented_image"
os.makedirs(folder_path, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "success", "message": "Welcome to the Milkfish Class and Weight Prediction API"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        # Step 1: Save uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_input:
            shutil.copyfileobj(file.file, temp_input)
            input_path = temp_input.name

        # Move or copy the image into the new folder
        os.makedirs(folder_path, exist_ok=True)
        input_folder = shutil.copy(input_path, folder_path)

        # Step 2: Prepare output directory for segmented image
        output_folder = "segmented_image/"
        os.makedirs(output_folder, exist_ok=True)

        # Step 3: Segment the fish from the image
        segment_fish_u2net(input_folder, output_folder)

        # Step 4: Find the segmented image inside output_folder
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        segmented_image_path = os.path.join(output_folder, input_basename + '.png')

        if not os.path.exists(segmented_image_path) or os.path.getsize(segmented_image_path) == 0:
            raise Exception(f"Segmentation failed: Output file not created or empty: {segmented_image_path}")

        # Step 5: Extract morphometric features from segmented image
        features = extract_morphometrics(segmented_image_path)
        features_df = pd.DataFrame([features])

        # Step 6: Normalize features
        normalized_features = normalize_features(features_df)

        # Step 7: Classify fish species
        gb_class_pred = gb_classifier.predict(normalized_features)
        rf_class_pred = rf_classifier.predict(normalized_features)

        meta_features = np.column_stack([rf_class_pred, gb_class_pred])

        fish_class = ensemble_svm_classifier.predict(meta_features)
        fish_class = label_encoder.inverse_transform(fish_class)

        print("Decoded Fish Class:", fish_class)

        # Step 8: Predict fish weight
        fish_weight = ensemble_regressor.predict(normalized_features)

        # Optional: If you want to send the segmented image back
        # with open(output_path, "rb") as img_file:
        #     segmented_image_bytes = img_file.read()

        # Step 9: Clean up temporary files
        os.remove(input_folder)
        os.remove(segmented_image_path)  # Don't remove output_path as it contains the segmented image

        # Step 10: Return the predictions
        return JSONResponse(content={
            "status": "success",
            "fish_class": fish_class.tolist(),
            "fish_weight": fish_weight.tolist(),
            # "segmented_image": base64.b64encode(segmented_image_bytes).decode()  # Optional
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)