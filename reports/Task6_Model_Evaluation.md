# Task 6: Model Evaluation

## Objective
Assess the model's performance on the testing dataset using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

## Implementation Details
1. **Testing Environment**:
   - The best saved `Pipeline` is loaded and applied to the 20% hold-out test set (`X_test`, `y_test`).
2. **Metrics Computation**:
   - We utilized `sklearn.metrics` to compute:
     - **Accuracy**: Overall correctness of the model.
     - **Precision**: How many predicted churners actually churned (minimizing false alarms).
     - **Recall**: How many actual churners were successfully identified (our priority, boosted by SMOTE).
     - **F1-Score**: Harmonic mean of Precision and Recall.
     - **ROC-AUC**: Model's ability to distinguish between classes across varying thresholds.
3. **Visual Output Generation**:
   - `confusion_matrix.png`: Heatmap of True Positives, True Negatives, False Positives, and False Negatives.
   - `roc_curve.png`: Receiver Operating Characteristic curve.
   - `feature_importance.png`: Visual representation of top attributes driving the model's predictions.
   - `evaluation_metrics.txt`: Formal text log of all metrics.
