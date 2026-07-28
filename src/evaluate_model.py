import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import joblib

from data_preprocessing import load_data, clean_data

def evaluate_model(data_path, model_path='models/churn_model.pkl', outputs_dir='outputs'):
    """
    Load model pipeline, evaluate on test split of raw data, and save evaluation plots.
    """
    print("Loading model and data for evaluation...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    pipeline = joblib.load(model_path)
    
    df = load_data(data_path)
    df = clean_data(df)
    
    from sklearn.model_selection import train_test_split
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n================ Best Model Evaluation ================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("========================================================")
    
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print("Saved confusion_matrix.png")
    
    # 2. ROC Curve
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    print("Saved roc_curve.png")
    
    # 3. Feature Importance / Coefficients
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + encoded_cat_features
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        title_str = 'Top 15 Feature Importances'
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
        title_str = 'Top 15 Absolute Feature Coefficients'
    else:
        importances = np.zeros(len(feature_names))
        title_str = 'Feature Impact'
        
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis', hue=feat_imp.index, legend=False)
    plt.title(title_str, fontsize=14, pad=15)
    plt.xlabel('Importance / Weight Magnitude', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    print("Saved feature_importance.png")
    
    with open(os.path.join(outputs_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n")

if __name__ == '__main__':
    data_path = os.path.join('Customer-Churn-Prediction', 'data', 'Telco_Customer_Churn_Dataset.csv')
    model_path = os.path.join('Customer-Churn-Prediction', 'models', 'churn_model.pkl')
    outputs_dir = os.path.join('Customer-Churn-Prediction', 'outputs')
    if os.path.exists(data_path) and os.path.exists(model_path):
        evaluate_model(data_path, model_path, outputs_dir)
    else:
        print("Required data or model files not found.")
