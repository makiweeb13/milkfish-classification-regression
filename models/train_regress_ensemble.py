from sklearn.metrics import root_mean_squared_error
import pandas as pd
import joblib
from utils.directories_utils import (
    weight_test_data, data_output, regressor_gradient_boosting_model, regressor_random_forest_model, regress_ensemble
)

from models.regress_ensemble import WeightedAverageEnsemble

def regress_with_ensemble():
    # Load your trained models
    gb_model = joblib.load(regressor_gradient_boosting_model)
    rf_model = joblib.load(regressor_random_forest_model)

    # Combine the models into an ensemble
    ensemble_model = WeightedAverageEnsemble(models=[gb_model, rf_model], weights=[0.6, 0.4])
    joblib.dump(ensemble_model, regress_ensemble)

    test_df = pd.read_csv(f"{data_output}{weight_test_data}")

    X_test = test_df.drop(columns=["weight"])
    y_test = test_df['weight']

    y_pred = ensemble_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Ensemble Test RMSE: {rmse:.4f}")