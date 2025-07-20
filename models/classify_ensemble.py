import joblib
import numpy as np

# Load your trained models
gb_model = joblib.load("models/saved_models/gradient_boosting_classifier.pkl")
rf_model = joblib.load("models/saved_models/random_forest_classifier.pkl")

# Soft Voting: average the predicted probabilities from both models
def soft_voting_predict(X):
    gb_probs = gb_model.predict_proba(X)
    rf_probs = rf_model.predict_proba(X)
    avg_probs = (gb_probs + rf_probs) / 2
    return avg_probs

# Create a simple wrapper object to act as your ensemble model
class SoftVotingEnsemble:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for model in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)