import os
import csv
import math

def load_cleaned_train(cleaned_path=os.path.join('data', 'cleaned', 'train_data.csv')):
    if not os.path.exists(cleaned_path):
        cleaned_path = os.path.join('Customer-Churn-Prediction', 'data', 'cleaned', 'train_data.csv')
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned train data not found at {cleaned_path}")
        
    with open(cleaned_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader if row]
        
    feature_names = headers[:-1]
    X = [[float(val) for val in row[:-1]] for row in data]
    y = [int(row[-1]) for row in data]
    return X, y, feature_names

def pearson_correlation(x_vec, y_vec):
    n = len(x_vec)
    mean_x = sum(x_vec) / n
    mean_y = sum(y_vec) / n
    
    num = sum((x_vec[i] - mean_x) * (y_vec[i] - mean_y) for i in range(n))
    den_x = sum((x_vec[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((y_vec[i] - mean_y) ** 2 for i in range(n))
    
    denom = math.sqrt(den_x * den_y)
    return num / denom if denom != 0 else 0.0

def select_features(X, y, feature_names, top_n=15):
    n_features = len(feature_names)
    scores = []
    
    for j in range(n_features):
        col_vals = [X[i][j] for i in range(len(X))]
        corr = pearson_correlation(col_vals, y)
        scores.append((feature_names[j], abs(corr), corr))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Top Feature Associations (Pearson Correlation) ---")
    for feat, abs_c, orig_c in scores[:top_n]:
        print(f"Feature: {feat:<35} | Abs Correlation: {abs_c:.4f} (Original: {orig_c:+.4f})")
        
    return scores

if __name__ == '__main__':
    try:
        X, y, feature_names = load_cleaned_train()
        select_features(X, y, feature_names)
    except Exception as e:
        print(f"Feature selection error: {e}")
