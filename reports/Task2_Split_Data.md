# Task 2: Split Data for Training and Testing

## Objective
Divide the data into training (80%) and testing (20%) sets for model training and evaluation, ensuring a representative split.

## Implementation Details
1. **Train-Test Split**: We use `train_test_split` from `sklearn.model_selection` with a `test_size=0.2` (yielding an 80/20 split).
2. **Stratification**: 
   - Because churn is often an imbalanced dataset (e.g., ~26% Yes / 74% No), we applied `stratify=y` to ensure that both the training and testing sets maintain the exact same proportion of churned vs. retained customers.
3. **Random State**: Set `random_state=42` to guarantee reproducibility across multiple pipeline runs.
