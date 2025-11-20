import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, classification_report, root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import joblib
from utils.directories_utils import (
    data_output, size_train_data, size_valid_data,
    save_gradient_boosting_model, save_label_encoder,
    weight_train_data, weight_valid_data, regressor_gradient_boosting_model
)
import matplotlib.pyplot as plt

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

     # Save feature importances
    feature_names = X_train.columns.tolist()
    gb_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    gb_importance_df.to_csv(f"{data_output}gb_feature_importances.csv", index=False)
    
    print(f"\nModel saved to: {save_gradient_boosting_model}")
    print(f"Label encoder saved to: {save_label_encoder}")
    print(f"Feature importances saved to: {data_output}gb_feature_importances.csv")

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
    try:
        # use the uncalibrated best estimator for learning curve
        base_estimator = random_search.best_estimator_
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

        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='0.2', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 's-', color='0.6', label='Validation score')
        plt.fill_between(train_sizes, train_scores_mean - np.std(train_scores, axis=1),
                         train_scores_mean + np.std(train_scores, axis=1), color='0.2', alpha=0.1)
        plt.fill_between(train_sizes, test_scores_mean - np.std(test_scores, axis=1),
                         test_scores_mean + np.std(test_scores, axis=1), color='0.6', alpha=0.1)
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve (Gradient Boosting)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not compute learning curve:", e)

    # --- Training/Validation Loss Curve ---
    try:
        # Retrain best model and track loss at each iteration
        best_params = random_search.best_params_
        gb_loss_model = GradientBoostingClassifier(
            **best_params,
            validation_fraction=0.2,
            n_iter_no_change=None,
            warm_start=False
        )
        gb_loss_model.fit(X_train, y_train_encoded)
        
        # Get training loss per iteration
        train_loss = gb_loss_model.train_score_
        
        plt.figure(figsize=(7, 5))
        plt.plot(range(len(train_loss)), train_loss, 'o-', color='0.2', linewidth=2, label='Training Loss')
        plt.xlabel('Iteration (n_estimators)')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve (Gradient Boosting)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not compute training loss curve:", e)

    # --- Combined Learning Curve (accuracy) and Training Loss ---
    try:
        # Learning curve (training vs validation accuracy)
        base_estimator = random_search.best_estimator_
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

        # Training loss per iteration
        best_params = random_search.best_params_
        gb_loss_model = GradientBoostingClassifier(
            **best_params,
            validation_fraction=0.2,
            n_iter_no_change=None,
            warm_start=False
        )
        gb_loss_model.fit(X_train, y_train_encoded)
        train_loss = gb_loss_model.train_score_

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Learning Curve
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color='0.2', label='Training score', linewidth=2)
        axes[0].plot(train_sizes, test_scores_mean, 's-', color='0.6', label='Validation score', linewidth=2)
        axes[0].fill_between(train_sizes, train_scores_mean - np.std(train_scores, axis=1),
                             train_scores_mean + np.std(train_scores, axis=1), color='0.2', alpha=0.1)
        axes[0].fill_between(train_sizes, test_scores_mean - np.std(test_scores, axis=1),
                             test_scores_mean + np.std(test_scores, axis=1), color='0.6', alpha=0.1)
        axes[0].set_xlabel('Training examples')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Learning Curve (Gradient Boosting)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Training Loss Curve
        axes[1].plot(range(len(train_loss)), train_loss, 'o-', color='0.2', linewidth=2, label='Training Loss')
        axes[1].set_xlabel('Iteration (n_estimators)')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss Curve (Gradient Boosting)')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not compute learning curve or training loss curve:", e)

    # Save model and label encoder
    joblib.dump(best_model, save_gradient_boosting_model)
    joblib.dump(le, save_label_encoder)


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
    gb_regressor = GradientBoostingRegressor(n_estimators=400, learning_rate=0.001, max_depth=2, random_state=30)
    gb_regressor.fit(X_train, y_train)

    # Save regressor feature importances
    feature_names = X_train.columns.tolist()
    gb_regressor_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': gb_regressor.feature_importances_
    }).sort_values('importance', ascending=False)
    gb_regressor_importance_df.to_csv(f"{data_output}gb_regressor_feature_importances.csv", index=False)
    print(f"Regressor feature importances saved to: {data_output}gb_regressor_feature_importances.csv")

    # Print feature importance
    feature_importance = gb_regressor.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Top 10):")
    print(importance_df.head(10))

    # --- Learning curve (train/valid RMSE vs training size) and staged n_estimators curve ---
    try:
        # Left: Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            GradientBoostingRegressor(n_estimators=400, learning_rate=0.001, max_depth=2, random_state=30),
            X_train, y_train,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            shuffle=True,
            random_state=42
        )
        # Convert neg MSE to RMSE
        train_rmse_lc = np.sqrt(-np.mean(train_scores, axis=1))
        valid_rmse_lc = np.sqrt(-np.mean(test_scores, axis=1))

        # Right: RMSE vs n_estimators using staged_predict
        max_estimators = 400
        step = max(20, max_estimators // 20)
        estimators_range = list(range(step, max_estimators + 1, step))
        
        gb_inc = GradientBoostingRegressor(
            n_estimators=step,
            warm_start=True,
            learning_rate=0.001,
            max_depth=2,
            random_state=30
        )

        train_rmse_vs_estim = []
        valid_rmse_vs_estim = []
        for n in estimators_range:
            gb_inc.n_estimators = n
            gb_inc.fit(X_train, y_train)
            tr_pred = gb_inc.predict(X_train)
            val_pred = gb_inc.predict(X_valid)
            train_rmse_vs_estim.append(np.sqrt(mean_squared_error(y_train, tr_pred)))
            valid_rmse_vs_estim.append(np.sqrt(mean_squared_error(y_valid, val_pred)))

        # Create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Learning curve
        axes[0].plot(train_sizes, train_rmse_lc, marker='o', color='0.2', linewidth=1.8, label='Train RMSE')
        axes[0].plot(train_sizes, valid_rmse_lc, marker='s', color='0.6', linewidth=1.8, label='Valid RMSE')
        axes[0].set_xlabel('Training examples')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Learning Curve (GB Regressor)')
        axes[0].legend(loc='best')
        axes[0].grid(alpha=0.3)

        # Right: RMSE vs n_estimators
        axes[1].plot(estimators_range, train_rmse_vs_estim, marker='o', color='0.2', linewidth=1.8, label='Train RMSE')
        axes[1].plot(estimators_range, valid_rmse_vs_estim, marker='s', color='0.6', linewidth=1.8, label='Valid RMSE')
        axes[1].set_xlabel('n_estimators')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE vs n_estimators (GB Regressor)')
        axes[1].legend(loc='best')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{data_output}gb_regressor_rmse_curves.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved GB regressor RMSE curves to: {data_output}gb_regressor_rmse_curves.png")
    except Exception as e:
        print("Could not compute learning curve or n_estimators curves:", e)

    # Predict on validation set
    y_valid_pred = gb_regressor.predict(X_valid)

    # Evaluate on validation set
    try:
        rmse = root_mean_squared_error(y_valid, y_valid_pred)
        mae = mean_absolute_error(y_valid, y_valid_pred)
        r2 = r2_score(y_valid, y_valid_pred)
        print(f"\nValidation RMSE: {rmse:.4f}")
        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation R²: {r2:.4f}")
    except Exception as e:
        print("Could not compute validation metrics:", e)

    # Save the regressor model
    joblib.dump(gb_regressor, regressor_gradient_boosting_model)

    print(f"Regressor model saved to: {regressor_gradient_boosting_model}")
