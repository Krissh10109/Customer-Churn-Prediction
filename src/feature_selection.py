import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def select_features_mutual_info(X, y, feature_names, top_n=15):
    """
    Calculate and return Mutual Information scores for all features.
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    print("\n--- Top Mutual Information Scores ---")
    print(mi_series.head(top_n))
    return mi_series

def select_features_rf(X, y, feature_names, top_n=15):
    """
    Calculate and return Random Forest feature importances.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n--- Top Random Forest Feature Importances ---")
    print(importances.head(top_n))
    return importances

if __name__ == '__main__':
    from data_preprocessing import preprocess_data
    data_path = os.path.join('Customer-Churn-Prediction', 'data', 'Telco_Customer_Churn_Dataset.csv')
    if os.path.exists(data_path):
        X_train, X_test, y_train, y_test, features = preprocess_data(data_path)
        select_features_mutual_info(X_train, y_train, features)
        select_features_rf(X_train, y_train, features)
    else:
        print(f"Please run preprocessing from the root directory or ensure {data_path} exists.")
