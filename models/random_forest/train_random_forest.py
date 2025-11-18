import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint
from sklearn.metrics import accuracy_score, classification_report, root_mean_squared_error, mean_squared_error
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import joblib
from utils.directories_utils import (
    data_output, size_train_data, size_valid_data,
    save_random_forest_model, save_label_encoder_rf,
    weight_train_data, weight_valid_data, regressor_random_forest_model
)

def classify_fish_with_random_forest():
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
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [None, 3, 5, 7, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        rf_model,
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

    # Save feature importances
    feature_names = X_train.columns.tolist()
    rf_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    rf_importance_df.to_csv(f"{data_output}rf_feature_importances.csv", index=False)
    
    print(f"\nModel saved to: {save_random_forest_model}")
    print(f"Label encoder saved to: {save_label_encoder_rf}")
    print(f"Feature importances saved to: {data_output}rf_feature_importances.csv")

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

    # --- Learning curve (training vs validation accuracy) ---
    # Use the base estimator from random_search for learning curve (un-calibrated)
    base_estimator = random_search.best_estimator_
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            base_estimator,
            X_train, y_train_encoded,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            shuffle=True,
            random_state=42
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(7,5))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='0.2', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 's-', color='0.6', label='Validation score')
        plt.fill_between(train_sizes, train_scores_mean - np.std(train_scores, axis=1),
                         train_scores_mean + np.std(train_scores, axis=1), color='0.2', alpha=0.1)
        plt.fill_between(train_sizes, test_scores_mean - np.std(test_scores, axis=1),
                         test_scores_mean + np.std(test_scores, axis=1), color='0.6', alpha=0.1)
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve (Random Forest)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not compute learning curve:", e)

    # Save model and label encoder
    joblib.dump(best_model, save_random_forest_model)
    joblib.dump(le, save_label_encoder_rf)


def regress_fish_with_random_forest():
    # Load training and validation data
    train_df = pd.read_csv(f"{data_output}{weight_train_data}")
    valid_df = pd.read_csv(f"{data_output}{weight_valid_data}")

    # Split features and labels
    X_train = train_df.drop(columns=["weight"])
    y_train = train_df["weight"].astype(float)

    X_valid = valid_df.drop(columns=["weight"])
    y_valid = valid_df["weight"].astype(float)

    # Train Random Forest Regressor (final model)
    final_n_estimators = 400
    rf_regressor = RandomForestRegressor(
        n_estimators=final_n_estimators, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=42, 
        max_features='sqrt', 
        bootstrap=True, 
        n_jobs=-1
    )
    rf_regressor.fit(X_train, y_train)

    # Save regressor feature importances
    feature_names = X_train.columns.tolist()
    rf_regressor_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_regressor.feature_importances_
    }).sort_values('importance', ascending=False)
    rf_regressor_importance_df.to_csv(f"{data_output}rf_regressor_feature_importances.csv", index=False)
    print(f"Regressor feature importances saved to: {data_output}rf_regressor_feature_importances.csv")

    # Print top feature importances
    importance_df = rf_regressor_importance_df
    print("\nFeature Importance (Top 10):")
    print(importance_df.head(10))

    # --- Left: Learning curve (train/valid RMSE vs training size) ---
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            RandomForestRegressor(**{k: v for k, v in rf_regressor.get_params().items() if k != 'n_estimators'}),
            X_train, y_train,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            shuffle=True,
            random_state=42
        )
        # convert neg MSE -> RMSE
        train_rmse_lc = np.sqrt(-np.mean(train_scores, axis=1))
        valid_rmse_lc = np.sqrt(-np.mean(test_scores, axis=1))
    except Exception as e:
        print("Could not compute learning_curve for RF regressor:", e)
        train_sizes, train_rmse_lc, valid_rmse_lc = None, None, None

    # --- Right: RMSE vs n_estimators (incremental fit using warm_start) ---
    try:
        max_estimators = final_n_estimators
        step = max(20, max_estimators // 20)  # about 20 points
        estimators_range = list(range(step, max_estimators + 1, step))
        rf_inc = RandomForestRegressor(
            n_estimators=step,
            warm_start=True,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1
        )

        train_rmse_vs_estim = []
        valid_rmse_vs_estim = []
        for n in estimators_range:
            rf_inc.n_estimators = n
            rf_inc.fit(X_train, y_train)  # adds trees when warm_start=True
            tr_pred = rf_inc.predict(X_train)
            val_pred = rf_inc.predict(X_valid)
            train_rmse_vs_estim.append(np.sqrt(mean_squared_error(y_train, tr_pred)))
            valid_rmse_vs_estim.append(np.sqrt(mean_squared_error(y_valid, val_pred)))
    except Exception as e:
        print("Could not compute incremental n_estimators curve for RF regressor:", e)
        estimators_range, train_rmse_vs_estim, valid_rmse_vs_estim = [], [], []

    # --- Combined side-by-side plot ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left subplot: learning-curve RMSE
        if train_sizes is not None:
            axes[0].plot(train_sizes, train_rmse_lc, marker='o', color='0.2', linewidth=1.8, label='Train RMSE')
            axes[0].plot(train_sizes, valid_rmse_lc, marker='s', color='0.6', linewidth=1.8, label='Valid RMSE')
            axes[0].set_xlabel('Training examples')
            axes[0].set_ylabel('RMSE')
            axes[0].set_title('Learning Curve (RF Regressor) - RMSE')
            axes[0].legend(loc='best')
            axes[0].grid(alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Learning curve not available', ha='center', va='center')
            axes[0].axis('off')

        # Right subplot: RMSE vs n_estimators
        if estimators_range:
            axes[1].plot(estimators_range, train_rmse_vs_estim, marker='o', color='0.25', linewidth=1.8, label='Train RMSE')
            axes[1].plot(estimators_range, valid_rmse_vs_estim, marker='s', color='0.6', linewidth=1.8, label='Valid RMSE')
            axes[1].set_xlabel('n_estimators')
            axes[1].set_ylabel('RMSE')
            axes[1].set_title('RMSE vs n_estimators (RF Regressor)')
            axes[1].legend(loc='best')
            axes[1].grid(alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'n_estimators curve not available', ha='center', va='center')
            axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{data_output}rf_regressor_rmse_curves.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved RF regressor RMSE/estimators plot to: {data_output}rf_regressor_rmse_curves.png")
    except Exception as e:
        print("Could not create/save combined RF regressor plots:", e)

    # Predict on validation set and report final RMSE
    try:
        y_valid_pred = rf_regressor.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
        print(f"Validation RMSE: {rmse:.4f}")
    except Exception as e:
        print("Could not compute final validation RMSE:", e)

    # Save the regressor model
    joblib.dump(rf_regressor, regressor_random_forest_model)
    print(f"Regressor model saved to: {regressor_random_forest_model}")