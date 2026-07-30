# Task 4: Model Selection

## Objective
Choose a suitable binary classification algorithm, considering options like logistic regression, decision trees, random forests, or gradient boosting.

## Implementation Details
1. **Benchmarking Strategy**:
   - Rather than guessing the best model, we designed an automated benchmarking pipeline that instantiates:
     - `LogisticRegression`
     - `DecisionTreeClassifier`
     - `RandomForestClassifier`
     - `GradientBoostingClassifier`
2. **SMOTE & GridSearchCV**:
   - We utilized `SMOTE` (Synthetic Minority Over-sampling Technique) to combat class imbalance.
   - We applied `GridSearchCV` to explore optimal hyperparameters for each classifier via 3-fold Cross Validation.
3. **Selection Criteria**:
   - Models were ranked dynamically based on their **ROC-AUC** score to ensure robust classification performance across all thresholds, alongside the **F1-Score**.
   - The top performing model pipeline was saved for downstream inference.
