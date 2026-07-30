# Task 3: Feature Selection

## Objective
Identify and select relevant features (attributes) influencing churn prediction, such as contract type, monthly charges, and tenure.

## Implementation Details
1. **Mutual Information Classification**:
   - We utilized `mutual_info_classif` from `sklearn.feature_selection` to score non-linear dependencies between categorical/continuous features and the target variable (Churn).
   - Variables like `Contract`, `tenure`, and `OnlineSecurity` typically yield the highest mutual information.
2. **Random Forest Feature Importance**:
   - We trained an auxiliary `RandomForestClassifier` purely to analyze Gini impurity reductions.
   - The Random Forest consistently identified `TotalCharges`, `MonthlyCharges`, and `tenure` as top driving attributes for predicting customer behavior.
3. **Outcome**: The results of this analysis directly informed the features we passed to the final pipeline and the insights built into the Streamlit Executive Analytics Dashboard.
