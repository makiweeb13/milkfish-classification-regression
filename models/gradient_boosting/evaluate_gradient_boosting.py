import pandas as pd
import joblib

# Load test data
test_df = pd.read_csv("data/test_features.csv")
X_test = test_df.drop(columns=["mapped_class"])
y_test = test_df["mapped_class"]

# Load model and label encoder
gb_model = joblib.load("models/gradient_boosting_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Predict probabilities
probs = gb_model.predict_proba(X_test)
preds = gb_model.predict(X_test)

# Convert numeric predictions back to labels
pred_labels = le.inverse_transform(preds)

# Output example
for i, prob in enumerate(probs[:5]):
    print(f"True: {y_test.iloc[i]} | Pred: {pred_labels[i]} | Probabilities: {prob}")
