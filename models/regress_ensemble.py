import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class WeightedAverageEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        # Get predictions from each model
        preds = np.array([model.predict(X) for model in self.models])
        # Weighted average
        return np.average(preds, axis=0, weights=self.weights)
