import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(filepath):
    """
    Load the Telco customer churn dataset from the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Perform basic cleaning operations on the dataset.
    """
    df = df.copy()
    
    # Convert TotalCharges to numeric, replace blank spaces with NaN, and impute with median or 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    # Impute missing TotalCharges (which are usually for tenure=0) with 0.0
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Drop customerID as it's not a predictor
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    return df

def preprocess_data(data_path, models_dir='models'):
    """
    Load, clean, build preprocessing pipeline, and split data into train/test sets.
    Saves the fitted preprocessing pipeline to models_dir.
    """
    df = load_data(data_path)
    df = clean_data(df)
    
    # Map target Churn variable to binary
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        X = df.drop(columns=['Churn'])
        y = df['Churn']
    else:
        X = df
        y = None
        
    # Define feature types
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Split the data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit preprocessor on training data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        # Get feature names after one-hot encoding
        cat_encoder = preprocessor.named_transformers_['cat']
        encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
        feature_names = numeric_features + encoded_cat_features
        
        # Convert to dataframes
        X_train_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)
        
        # Save preprocessor
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
        print("Preprocessor saved successfully.")
        
        return X_train_df, X_test_df, y_train, y_test, feature_names
    else:
        return preprocessor.transform(X)

if __name__ == '__main__':
    # Test script run
    data_path = os.path.join('Customer-Churn-Prediction', 'data', 'Telco_Customer_Churn_Dataset.csv')
    if os.path.exists(data_path):
        X_train, X_test, y_train, y_test, features = preprocess_data(data_path)
        print(f"Data preprocessed. X_train shape: {X_train.shape}, features count: {len(features)}")
    else:
        print(f"Please place the Telco_Customer_Churn_Dataset.csv in {data_path} first.")
