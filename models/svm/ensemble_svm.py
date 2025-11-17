import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib
from utils.directories_utils import (
    size_valid_data, size_test_data, data_output, save_gradient_boosting_model, 
    save_label_encoder, save_random_forest_model, classify_svm_meta, saved_class_ensemble_scaler
)
import matplotlib.pyplot as plt

def ensemble_with_svm():
    # Load your trained models
    gb_model = joblib.load(save_gradient_boosting_model)
    rf_model = joblib.load(save_random_forest_model)

    # Load test data
    valid_df = pd.read_csv(f"{data_output}{size_valid_data}")

    X_valid = valid_df.drop(columns=["mapped_class"])
    y_valid = valid_df['mapped_class'] 

    le = joblib.load(save_label_encoder)
    y_valid_encoded = le.transform(y_valid)

    # Get predictions from Base Models
    gb_preds = gb_model.predict(X_valid)
    rf_preds = rf_model.predict(X_valid)

    # Stack predictions as 2D features: shape (n_samples, 2)
    meta_X_val = np.column_stack([rf_preds, gb_preds])

    # Normalize the meta-features using StandardScaler
    scaler = StandardScaler()
    meta_X_val = scaler.fit_transform(meta_X_val)
    joblib.dump(scaler, saved_class_ensemble_scaler)  # Save the scaler for future use

    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid.fit(meta_X_val, y_valid_encoded)
    print("Best params:", grid.best_params_)

    # Use the best estimator from GridSearchCV
    svm_meta = grid.best_estimator_

    # Load test data
    test_df = pd.read_csv(f"{data_output}{size_test_data}")
    X_test = test_df.drop(columns=["mapped_class"])
    y_test = test_df['mapped_class']

    y_test_encoded = le.transform(y_test)

    # Predict on test set
    gb_test_preds = gb_model.predict(X_test)
    rf_test_preds = rf_model.predict(X_test)

    # Stack predictions as 2D features: shape (n_samples, 2)
    meta_X_test = np.column_stack([rf_test_preds, gb_test_preds])

    # Normalize the meta-features using StandardScaler
    meta_X_test = scaler.transform(meta_X_test)

    # Predict with the SVM meta-model
    svm_meta_preds = svm_meta.predict(meta_X_test)
    svm_meta_proba = svm_meta.predict_proba(meta_X_test)

    # Save the SVM model
    joblib.dump(svm_meta, classify_svm_meta)

    accuracy = accuracy_score(y_test_encoded, svm_meta_preds)
    report = classification_report(y_test_encoded, svm_meta_preds, target_names=le.classes_)

    print(f"SVM Ensemble Accuracy: {accuracy:.4f}")
    print("Classification Report (SVM):")
    print(report)

    # Confusion matrix with annotations
    cm = confusion_matrix(y_test_encoded, svm_meta_preds)
    print("\nConfusion Matrix (counts):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Greys', xticks_rotation='vertical')
    ax = plt.gca()

    plt.title("SVM Ensemble - Test Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # --- Compute ROC-AUC (macro) for all three models ---
    try:
        n_classes = len(le.classes_)
        y_test_binarized = label_binarize(y_test_encoded, classes=list(range(n_classes)))
        
        # Get probabilities from all models
        gb_proba = gb_model.predict_proba(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        
        # Compute ROC-AUC (macro) for each model
        gb_roc_auc = roc_auc_score(y_test_binarized, gb_proba, average='macro', multi_class='ovr')
        rf_roc_auc = roc_auc_score(y_test_binarized, rf_proba, average='macro', multi_class='ovr')
        svm_roc_auc = roc_auc_score(y_test_binarized, svm_meta_proba, average='macro', multi_class='ovr')
        
        print(f"\nROC-AUC (macro) - Gradient Boosting: {gb_roc_auc:.4f}")
        print(f"ROC-AUC (macro) - Random Forest: {rf_roc_auc:.4f}")
        print(f"ROC-AUC (macro) - SVM Ensemble: {svm_roc_auc:.4f}")
        
        # Plot comparative bar chart
        models = ['Gradient\nBoosting', 'Random\nForest', 'SVM\nEnsemble']
        roc_auc_scores = [gb_roc_auc, rf_roc_auc, svm_roc_auc]
        colors = ['0.3', '0.5', '0.7']
        
        plt.figure(figsize=(10, 7))
        bars = plt.bar(models, roc_auc_scores, color=colors, edgecolor='black', linewidth=2, width=0.6)
        plt.ylabel('ROC-AUC (macro)', fontsize=12, fontweight='bold')
        plt.title('Comparative ROC-AUC (macro) on Test Set', fontsize=13, fontweight='bold')
        plt.ylim([0, 1.2])
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Annotate bars with values
        for bar, score in zip(bars, roc_auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print("Could not compute or plot ROC-AUC comparison:", e)

    # --- Extract and plot feature importances from base models ---
    try:
        # Load feature importances from training
        gb_importance_df = pd.read_csv(f"{data_output}gb_feature_importances.csv")
        rf_importance_df = pd.read_csv(f"{data_output}rf_feature_importances.csv")
        
        # Get top 10 features from each
        top_k = min(10, len(gb_importance_df))
        gb_top = gb_importance_df.head(top_k).iloc[::-1]
        rf_top = rf_importance_df.head(top_k).iloc[::-1]
        
        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, top_k * 0.6)))
        
        # Gradient Boosting
        bars_gb = axes[0].barh(gb_top['feature'], gb_top['importance'], color='0.3', edgecolor='black')
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Top Features - Gradient Boosting')
        for bar in bars_gb:
            axes[0].text(bar.get_width() + max(1e-6, gb_top['importance'].max()*0.01),
                         bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}",
                         va='center', fontsize=8)
        
        # Random Forest
        bars_rf = axes[1].barh(rf_top['feature'], rf_top['importance'], color='0.5', edgecolor='black')
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top Features - Random Forest')
        for bar in bars_rf:
            axes[1].text(bar.get_width() + max(1e-6, rf_top['importance'].max()*0.01),
                         bar.get_y() + bar.get_height()/2, f"{bar.get_width():.4f}",
                         va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print("Could not plot feature importances:", e)