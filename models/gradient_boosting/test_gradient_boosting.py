import pandas as pd
import joblib
from utils.directories_utils import (
    data_output, size_test_data,
    save_gradient_boosting_model, save_label_encoder
)

def gradientBoostingClassifier():
    # Load test data
    test_df = pd.read_csv(f"{data_output}{size_test_data}")
    X_test = test_df.drop(columns=["mapped_class"])
    y_test = test_df["mapped_class"]

    # Load model and label encoder
    gb_model = joblib.load(f"{save_gradient_boosting_model}")
    le = joblib.load(f"{save_label_encoder}")

    # Predict probabilities
    probs = gb_model.predict_proba(X_test)
    preds = gb_model.predict(X_test)

    # Convert numeric predictions back to labels
    pred_labels = le.inverse_transform(preds)

    # Output example
    for i, prob in enumerate(probs[:]):
        print(f"True: {y_test.iloc[i]} | Pred: {pred_labels[i]} | Probabilities: {prob}")
