import numpy as np
import pandas as pd
import joblib
from utils.directories_utils import (
    data_output, size_test_data,
    save_random_forest_model, save_label_encoder_rf,
    weight_test_data, regressor_random_forest_model
)

def randomForestClassifier():
    # Load test data
    test_df = pd.read_csv(f"{data_output}{size_test_data}")
    X_test = test_df.drop(columns=["mapped_class"])
    y_test = test_df["mapped_class"]

    # Load model and label encoder
    rf_model = joblib.load(f"{save_random_forest_model}")
    le = joblib.load(f"{save_label_encoder_rf}")

    # Predict probabilities
    probs = rf_model.predict_proba(X_test)
    preds = rf_model.predict(X_test)

    # Convert numeric predictions back to labels
    pred_labels = le.inverse_transform(preds)

    # Output example
    for i, prob in enumerate(probs[:]):
        print(f"True: {y_test.iloc[i]} | Pred: {pred_labels[i]} | Probabilities: {prob}")

def randomForestRegressor():
    # Load test data
    test_df = pd.read_csv(f"{data_output}{weight_test_data}")
    X_test = test_df.drop(columns=["weight"])
    y_test = test_df["weight"].astype(float)

    # Load regressor model
    rf_regressor = joblib.load(regressor_random_forest_model)

    # Predict on test set
    y_test_pred = rf_regressor.predict(X_test)

    # Output example
    for i, pred in enumerate(y_test_pred[:]):
        print(f"True: {y_test.iloc[i]} | Predicted: {pred:.2f}")

    # Evaluate on test set
    rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    print(f"Test RMSE: {rmse:.4f}")