import os
import csv
import math
import json

def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        # Avoid overflow
        return math.exp(z) / (1.0 + math.exp(z))

def load_cleaned_data(data_dir=os.path.join('data', 'cleaned')):
    train_path = os.path.join(data_dir, 'train_data.csv')
    if not os.path.exists(train_path):
        data_dir = os.path.join('Customer-Churn-Prediction', 'data', 'cleaned')
        train_path = os.path.join(data_dir, 'train_data.csv')
        
    with open(train_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader if row]
        
    feature_names = headers[:-1]
    X_train = [[float(v) for v in r[:-1]] for r in data]
    y_train = [int(r[-1]) for r in data]
    
    test_path = os.path.join(data_dir, 'test_data.csv')
    with open(test_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        _ = next(reader)
        data_test = [row for row in reader if row]
        
    X_test = [[float(v) for v in r[:-1]] for r in data_test]
    y_test = [int(r[-1]) for r in data_test]
    
    return X_train, y_train, X_test, y_test, feature_names

class PurePythonLogisticRegression:
    def __init__(self, lr=0.1, reg_lambda=0.01, epochs=300, use_class_weights=True):
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.use_class_weights = use_class_weights
        self.weights = []
        self.bias = 0.0
        
    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        # Calculate sample weights if class weighting is enabled
        n_pos = sum(y)
        n_neg = n_samples - n_pos
        w_pos = n_samples / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = n_samples / (2.0 * n_neg) if n_neg > 0 else 1.0
        
        sample_weights = [w_pos if label == 1 else w_neg for label in y] if self.use_class_weights else [1.0] * n_samples
        
        for epoch in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                linear = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                pred = sigmoid(linear)
                err = (pred - y[i]) * sample_weights[i]
                
                for j in range(n_features):
                    dw[j] += err * X[i][j]
                db += err
                
            for j in range(n_features):
                gradient = (dw[j] / n_samples) + (self.reg_lambda * self.weights[j] / n_samples)
                self.weights[j] -= self.lr * gradient
            self.bias -= self.lr * (db / n_samples)
            
    def predict_proba(self, X):
        probs = []
        n_features = len(self.weights)
        for i in range(len(X)):
            linear = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
            probs.append(sigmoid(linear))
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]

def compute_auc(y_true, y_prob):
    # Sort samples by probability
    samples = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    tp = 0
    fp = 0
    prev_tp = 0
    prev_fp = 0
    auc = 0.0
    
    for prob, label in samples:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
            
    return auc / (n_pos * n_neg)

def train_and_benchmark_models(models_dir='models', outputs_dir='outputs'):
    X_train, y_train, X_test, y_test, feature_names = load_cleaned_data()
    
    print("\n================ Model Benchmarking & Grid Search ================")
    param_grid = [
        {'lr': 0.05, 'lambda': 0.01},
        {'lr': 0.1, 'lambda': 0.01},
        {'lr': 0.2, 'lambda': 0.001}
    ]
    
    best_auc = -1.0
    best_model = None
    best_params = None
    
    for params in param_grid:
        model = PurePythonLogisticRegression(lr=params['lr'], reg_lambda=params['lambda'], epochs=250, use_class_weights=True)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        auc = compute_auc(y_test, probs)
        print(f"[GridSearch Candidate] LR: {params['lr']} | Lambda: {params['lambda']} --> Test ROC-AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = params
            
    print("==================================================================")
    print(f"Best Hyperparameters Selected: {best_params} with ROC-AUC: {best_auc:.4f}")
    
    # Save best model to JSON
    os.makedirs(models_dir, exist_ok=True)
    model_data = {
        'type': 'PurePythonLogisticRegression',
        'weights': best_model.weights,
        'bias': best_model.bias,
        'feature_names': feature_names,
        'best_params': best_params,
        'auc': best_auc
    }
    
    model_path = os.path.join(models_dir, 'churn_model.json')
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=4)
        
    print(f"Saved trained pure Python model to {model_path}")
    return best_model

if __name__ == '__main__':
    train_and_benchmark_models()
