import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from data_preprocessing import load_data, clean_data

def train_and_benchmark_models(data_path, models_dir='models', outputs_dir='outputs'):
    """
    Train and benchmark Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
    Selects the best performing model based on ROC-AUC and saves it.
    """
    print("Loading and preprocessing dataset...")
    df = load_data(data_path)
    df = clean_data(df)
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define candidate models
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    print("\n================ Model Benchmarking ================")
    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        trained_pipelines[name] = pipeline
        
        print(f"[{name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
    print("====================================================")
    
    results_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
    
    # Ensure outputs and models directory exist
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save comparison plot
    plt.figure(figsize=(10, 6))
    melted_df = pd.melt(results_df, id_vars=['Model'], value_vars=['Accuracy', 'F1-Score', 'ROC-AUC'],
                        var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=melted_df, palette='viridis')
    plt.title('Algorithm Comparison (Task 4: Model Selection)', fontsize=14, pad=15)
    plt.ylim(0.4, 1.0)
    plt.ylabel('Performance Score', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved algorithm comparison plot to {plot_path}")
    
    # Save comparison metrics CSV
    results_df.to_csv(os.path.join(outputs_dir, 'model_benchmark.csv'), index=False)
    
    # Save the best model
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    best_model_path = os.path.join(models_dir, 'churn_model.pkl')
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n[Best Model Selected] '{best_model_name}' (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
    print(f"Saved best model pipeline to {best_model_path}")
    
    return best_pipeline, results_df

if __name__ == '__main__':
    data_path = os.path.join('Customer-Churn-Prediction', 'data', 'Telco_Customer_Churn_Dataset.csv')
    models_dir = os.path.join('Customer-Churn-Prediction', 'models')
    outputs_dir = os.path.join('Customer-Churn-Prediction', 'outputs')
    
    if os.path.exists(data_path):
        train_and_benchmark_models(data_path, models_dir=models_dir, outputs_dir=outputs_dir)
    else:
        print(f"Dataset not found at {data_path}")
