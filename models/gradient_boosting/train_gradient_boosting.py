import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils.directories_utils import (
    data_output, size_train_data, size_valid_data,
    save_gradient_boosting_model, save_label_encoder
)

def classify_fish_with_gradient_boosting():
    # Load training and validation data
    train_df = pd.read_csv(f"{data_output}{size_train_data}")
    valid_df = pd.read_csv(f"{data_output}{size_valid_data}")

    # Split features and labels
    X_train = train_df.drop(columns=["mapped_class"])
    y_train = train_df["mapped_class"]

    X_valid = valid_df.drop(columns=["mapped_class"])
    y_valid = valid_df["mapped_class"]

    # Encode class labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_valid_encoded = le.transform(y_valid)

    # Train Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train_encoded)

    # Predict on validation set
    y_valid_proba = gb_model.predict_proba(X_valid)
    y_valid_pred = np.argmax(y_valid_proba, axis=1)

    # Evaluate
    print("Validation Accuracy:", accuracy_score(y_valid_encoded, y_valid_pred))
    print("Classification Report (Validation):")
    print(classification_report(y_valid_encoded, y_valid_pred, target_names=le.classes_))

    # Save model and label encoder
    joblib.dump(gb_model, save_gradient_boosting_model)
    joblib.dump(le, save_label_encoder)
