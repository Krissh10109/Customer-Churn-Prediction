# Task 5: Model Training

## Objective
Train the selected machine learning model using the training dataset. Utilize chosen features as input and "Churn" column as the target variable.

## Implementation Details
1. **Pipeline Execution**:
   - The winning model from the `GridSearchCV` selection process was fully fitted on the training split (`X_train`, `y_train`).
   - The target variable `Churn` was explicitly mapped to `1` (Yes) and `0` (No).
2. **End-to-End Pipeline**:
   - The model is not trained in isolation. It is packaged within a scikit-learn/imblearn `Pipeline` that sequentially executes:
     - Scaling & One-Hot Encoding (`ColumnTransformer`)
     - Minority Oversampling (`SMOTE`)
     - Classification (e.g., `RandomForestClassifier`)
3. **Persistence**:
   - The fully trained `Pipeline` was persisted to disk as `models/churn_model.pkl` using `joblib`.
   - This ensures the exact scaling states and model weights are identically replicated during Streamlit app inference.
