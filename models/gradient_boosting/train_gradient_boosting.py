import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, classification_report, root_mean_squared_error
import joblib
from utils.directories_utils import (
    data_output, size_train_data, size_valid_data,
    save_gradient_boosting_model, save_label_encoder,
    weight_train_data, weight_valid_data, regressor_gradient_boosting_model
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
    gb_model = GradientBoostingClassifier()

    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 6],
        'subsample': [0.5, 0.6, 0.8, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        gb_model,
        param_distributions=param_dist,
        n_iter=25,
        scoring='accuracy',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model with hyperparameter tuning
    random_search.fit(X_train, y_train_encoded)

    # Print best parameters from RandomizedSearchCV
    print("Best Parameters:", random_search.best_params_)
    print("Best CV Score:", random_search.best_score_)

    # Get the best model from RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Print feature importance
    feature_names = X_train.columns
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Top 10):")
    print(importance_df.head(10))

    # Calibrate the model using validation data (cv='prefit' means base_model is already trained)
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_valid, y_valid_encoded)

    best_model = calibrated_model 

    # Predict on validation set
    y_valid_proba = best_model.predict_proba(X_valid)
    y_valid_pred = np.argmax(y_valid_proba, axis=1)

    # Predict on training set
    y_train_proba = best_model.predict_proba(X_train)
    y_train_pred = np.argmax(y_train_proba, axis=1)

    # Evaluate on training set
    print("Training Accuracy:", accuracy_score(y_train_encoded, y_train_pred))
    print("Classification Report (Training):")
    print(classification_report(y_train_encoded, y_train_pred, target_names=le.classes_))

    # Evaluate on validation set           
    print("Validation Accuracy:", accuracy_score(y_valid_encoded, y_valid_pred))
    print("Classification Report (Validation):")
    print(classification_report(y_valid_encoded, y_valid_pred, target_names=le.classes_))

    # Save model and label encoder
    joblib.dump(best_model, save_gradient_boosting_model)
    joblib.dump(le, save_label_encoder)

    print(f"\nModel saved to: {save_gradient_boosting_model}")
    print(f"Label encoder saved to: {save_label_encoder}")


def regress_fish_with_gradient_boosting():
    # Load training and validation data
    train_df = pd.read_csv(f"{data_output}{weight_train_data}")
    valid_df = pd.read_csv(f"{data_output}{weight_valid_data}")

    # Split features and labels
    X_train = train_df.drop(columns=["weight"])
    y_train = train_df["weight"].astype(float)

    X_valid = valid_df.drop(columns=["weight"])
    y_valid = valid_df["weight"].astype(float)

    # Train Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    gb_regressor.fit(X_train, y_train)

    # Predict on validation set
    y_valid_pred = gb_regressor.predict(X_valid)

    # Evaluate on validation set
    rmse = root_mean_squared_error(y_valid, y_valid_pred)
    print(f"Validation RMSE: {rmse:.4f}")

    # Save the regressor model
    joblib.dump(gb_regressor, regressor_gradient_boosting_model)

    print(f"Regressor model saved to: {regressor_gradient_boosting_model}")
