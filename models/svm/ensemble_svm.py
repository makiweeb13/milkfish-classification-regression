import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils.directories_utils import (
    size_valid_data, size_test_data, data_output, save_gradient_boosting_model, save_label_encoder, save_random_forest_model
)

def ensemble_with_svm():
    # Load your trained models
    gb_model = joblib.load(save_gradient_boosting_model)
    rf_model = joblib.load(save_random_forest_model)

    # Load test data
    valid_df = pd.read_csv(f"{data_output}{size_valid_data}")

    X_valid = valid_df.drop(columns=["mapped_class"])
    y_valid = valid_df['mapped_class'] 

    le = joblib.load(save_label_encoder)
    y_valid_encoded = le.transform(y_valid)

    # Get predictions from Base Models
    gb_preds = gb_model.predict(X_valid)
    rf_preds = rf_model.predict(X_valid)

    # Stack predictions as 2D features: shape (n_samples, 2)
    meta_X_val = np.column_stack([rf_preds, gb_preds])

    # Normalize the meta-features using StandardScaler
    scaler = StandardScaler()
    meta_X_val = scaler.fit_transform(meta_X_val)

    # Meta-model: SVM
    svm_meta = SVC(probability=True, kernel='rbf', C=1.0)
    svm_meta.fit(meta_X_val, y_valid_encoded)

    # Load test data
    test_df = pd.read_csv(f"{data_output}{size_test_data}")
    X_test = test_df.drop(columns=["mapped_class"])
    y_test = test_df['mapped_class']

    y_test_encoded = le.transform(y_test)

    # Predict on test set
    gb_test_preds = gb_model.predict(X_test)
    rf_test_preds = rf_model.predict(X_test)

    # Stack predictions as 2D features: shape (n_samples, 2)
    meta_X_test = np.column_stack([rf_test_preds, gb_test_preds])

    # Normalize the meta-features using StandardScaler
    meta_X_test = scaler.transform(meta_X_test)

    # Predict with the SVM meta-model
    svm_meta_preds = svm_meta.predict(meta_X_test)

    accuracy = accuracy_score(y_test_encoded, svm_meta_preds)
    report = classification_report(y_test_encoded, svm_meta_preds, target_names=le.classes_)

    print(f"SVM Ensemble Accuracy: {accuracy:.4f}")
    print("Classification Report (SVM):")
    print(report)