from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
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
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Ensemble Test RMSE: {rmse:.4f}")
    print(f"Ensemble Test MAE: {mae:.4f}")
    print(f"Ensemble Test R²: {r2:.4f}")

    # --- Residual scatter plot (Actual vs Predicted) ---
    try:
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, color='0.4', edgecolor='black', linewidth=0.5)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        plt.xlabel('Predicted Weight')
        plt.ylabel('Residuals')
        plt.title('Residual Plot - Weighted Average Ensemble (Regression)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{data_output}ensemble_regressor_residuals.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved residual plot to: {data_output}ensemble_regressor_residuals.png")
    except Exception as e:
        print("Could not create residual plot:", e)

    # --- Extract and plot feature importances from base models ---
    try:
        # Load feature importances from training
        gb_importance_df = pd.read_csv(f"{data_output}gb_regressor_feature_importances.csv")
        rf_importance_df = pd.read_csv(f"{data_output}rf_regressor_feature_importances.csv")
        
        # Get top 10 features from each
        top_k = min(10, len(gb_importance_df))
        gb_top = gb_importance_df.head(top_k).iloc[::-1]
        rf_top = rf_importance_df.head(top_k).iloc[::-1]
        
        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, top_k * 0.5)))
        
        # Gradient Boosting
        bars_gb = axes[0].barh(gb_top['feature'], gb_top['importance'], color='0.3', edgecolor='black')
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Top Features - Gradient Boosting (Regressor)')
        for bar in bars_gb:
            axes[0].text(bar.get_width() + max(1e-6, gb_top['importance'].max()*0.01),
                         bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}",
                         va='center', fontsize=8)
        
        # Random Forest
        bars_rf = axes[1].barh(rf_top['feature'], rf_top['importance'], color='0.5', edgecolor='black')
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top Features - Random Forest (Regressor)')
        for bar in bars_rf:
            axes[1].text(bar.get_width() + max(1e-6, rf_top['importance'].max()*0.01),
                         bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}",
                         va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{data_output}ensemble_regressor_feature_importances.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved feature importance plot to: {data_output}ensemble_regressor_feature_importances.png")
        
    except Exception as e:
        print("Could not plot feature importances:", e)