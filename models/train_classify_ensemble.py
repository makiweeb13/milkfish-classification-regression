import pandas as pd
from models.classify_ensemble import SoftVotingEnsemble
import joblib
from utils.directories_utils import (
    size_test_data, data_output, save_gradient_boosting_model, save_label_encoder, save_random_forest_model, classify_ensemble
)
from sklearn.metrics import classification_report, accuracy_score

# Load your trained models
gb_model = joblib.load(save_gradient_boosting_model)
rf_model = joblib.load(save_random_forest_model)

# Create model
ensemble = SoftVotingEnsemble([gb_model, rf_model])
joblib.dump(ensemble, classify_ensemble)
print('Ensemble model saved successfully')

def classify_with_ensemble():
    test_df = pd.read_csv(f"{data_output}{size_test_data}")

    X_test = test_df.drop(columns=["mapped_class"])
    y_test = test_df['mapped_class'] 
    
    le = joblib.load(save_label_encoder)
    y_test_encoded = le.transform(y_test)
    
    ensemble_model = joblib.load(classify_ensemble)
    ensemble_preds = ensemble_model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, ensemble_preds)
    report = classification_report(y_test_encoded, ensemble_preds, target_names=le.classes_)

    print(f"Ensemble Test Accuracy: {accuracy:.4f}")
    print("Classification Report (Ensemble):")
    print(report)