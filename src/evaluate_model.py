import os
import csv
import json
import math

from train_model import load_cleaned_data, sigmoid, compute_auc

def evaluate_model(models_dir='models', outputs_dir='outputs'):
    model_path = os.path.join(models_dir, 'churn_model.json')
    if not os.path.exists(model_path):
        models_dir = os.path.join('Customer-Churn-Prediction', 'models')
        model_path = os.path.join(models_dir, 'churn_model.json')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    with open(model_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
        
    weights = model_data['weights']
    bias = model_data['bias']
    feature_names = model_data['feature_names']
    
    _, _, X_test, y_test, _ = load_cleaned_data()
    
    y_prob = []
    y_pred = []
    for row in X_test:
        linear = sum(row[j] * weights[j] for j in range(len(weights))) + bias
        prob = sigmoid(linear)
        y_prob.append(prob)
        y_pred.append(1 if prob >= 0.5 else 0)
        
    tp = sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 0)
    
    total = len(y_test)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    roc_auc = compute_auc(y_test, y_prob)
    
    print("\n================ Best Model Evaluation ================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [TN: {tn:4d} | FP: {fp:4d}]")
    print(f"  [FN: {fn:4d} | TP: {tp:4d}]")
    print("========================================================")
    
    os.makedirs(outputs_dir, exist_ok=True)
    
    with open(os.path.join(outputs_dir, 'evaluation_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
        
    with open(os.path.join(outputs_dir, 'model_benchmark.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
        writer.writerow(['Logistic Regression (Balanced)', f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"])
        
    feat_weights = sorted(zip(feature_names, [abs(w) for w in weights], weights), key=lambda x: x[1], reverse=True)
    with open(os.path.join(outputs_dir, 'feature_importance.txt'), 'w', encoding='utf-8') as f:
        f.write("Rank | Feature Name | Absolute Weight | Sign\n")
        f.write("-" * 50 + "\n")
        for i, (name, abs_w, w) in enumerate(feat_weights[:15], 1):
            sign = "Positive (+)" if w >= 0 else "Negative (-)"
            f.write(f"{i:2d}   | {name:<30} | {abs_w:.4f} | {sign}\n")
            
    print(f"Saved evaluation metrics to {os.path.join(outputs_dir, 'evaluation_metrics.txt')}")
    print(f"Saved feature importances to {os.path.join(outputs_dir, 'feature_importance.txt')}")

if __name__ == '__main__':
    evaluate_model()
