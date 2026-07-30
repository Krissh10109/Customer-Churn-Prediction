# Task 1: Data Preparation

## Objective
Load and preprocess the dataset, addressing missing values, and encoding categorical variables for machine learning readiness.

## Implementation Details
1. **Data Loading**: We load the raw `Telco_Customer_Churn_Dataset.csv`.
2. **Data Cleaning**:
   - Dropped the `customerID` column as it is not a predictor.
   - Handled missing values in `TotalCharges` (originally blank strings) by coercing to numeric and imputing with 0.0, assuming missing values mean tenure = 0.
3. **Encoding & Scaling**:
   - Mapped the target variable `Churn` from `Yes/No` to `1/0`.
   - Used `StandardScaler` to scale continuous numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`).
   - Used `OneHotEncoder` to encode all categorical features, ensuring no ordinal relationships are falsely inferred by the model.

## Output
A fitted `preprocessor.pkl` is saved to process future inferences consistently.
