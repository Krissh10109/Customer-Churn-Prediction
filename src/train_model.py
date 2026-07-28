import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Import preprocessing steps
from data_preprocessing import load_data, clean_data, preprocess_data

def train_model(data_path, models_dir='models'):
    """
    Train a Random Forest classifier using GridSearchCV and save the full pipeline.
    """
    print("Loading and cleaning data...")
    df = load_data(data_path)
    df = clean_data(df)
    
    # Map target Churn variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Reload preprocessor fitted during preprocessing (or define it fresh)
    # We will build a pipeline containing a fresh preprocessor to keep it fully end-to-end.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Split raw data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Setting up training pipeline...")
    # Define classifier
    classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Create final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Define grid search parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 15, None],
        'classifier__min_samples_split': [2, 5]
    }
    
    print("Running GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'churn_model.pkl')
    
    # Save the complete pipeline
    joblib.dump(best_pipeline, model_path)
    print(f"Complete pipeline model saved to {model_path}")
    
    return best_pipeline, X_test, y_test

if __name__ == '__main__':
    data_path = os.path.join('Customer-Churn-Prediction', 'data', 'Telco_Customer_Churn_Dataset.csv')
    if os.path.exists(data_path):
        models_dir = os.path.join('Customer-Churn-Prediction', 'models')
        train_model(data_path, models_dir=models_dir)
    else:
        print(f"Dataset not found at {data_path}")
