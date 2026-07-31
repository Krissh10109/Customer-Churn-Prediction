import os
import csv
import math
import random
import json

def load_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader if row]
    return headers, data

def preprocess_data(data_path, models_dir='models', cleaned_dir=os.path.join('data', 'cleaned')):
    print("Loading dataset...")
    headers, data = load_csv(data_path)
    
    header_idx = {h: i for i, h in enumerate(headers)}
    cust_id_idx = header_idx.get('customerID', None)
    churn_idx = header_idx.get('Churn', None)
    
    feature_indices = [i for i in range(len(headers)) if i not in (cust_id_idx, churn_idx)]
    feature_names_raw = [headers[i] for i in feature_indices]
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    processed_rows = []
    targets = []
    
    for row in data:
        if not row:
            continue
        target = 1 if row[churn_idx].strip() == 'Yes' else 0
        targets.append(target)
        
        row_dict = {}
        for idx in feature_indices:
            col_name = headers[idx]
            val = row[idx].strip()
            if col_name == 'TotalCharges':
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0
            elif col_name in ['tenure', 'MonthlyCharges']:
                val = float(val) if val else 0.0
            row_dict[col_name] = val
        processed_rows.append(row_dict)
        
    cat_values = {}
    for col in feature_names_raw:
        if col not in numeric_cols:
            cat_values[col] = sorted(list(set(r[col] for r in processed_rows)))
            
    encoded_feature_names = []
    for col in feature_names_raw:
        if col in numeric_cols:
            encoded_feature_names.append(col)
        else:
            for val in cat_values[col]:
                encoded_feature_names.append(f"{col}_{val}")
                
    X_vectors = []
    for r in processed_rows:
        vector = []
        for col in feature_names_raw:
            if col in numeric_cols:
                vector.append(float(r[col]))
            else:
                for val in cat_values[col]:
                    vector.append(1.0 if r[col] == val else 0.0)
        X_vectors.append(vector)
        
    random.seed(42)
    pos_indices = [i for i, t in enumerate(targets) if t == 1]
    neg_indices = [i for i, t in enumerate(targets) if t == 0]
    
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)
    
    train_pos_len = int(0.8 * len(pos_indices))
    train_neg_len = int(0.8 * len(neg_indices))
    
    train_idx = set(pos_indices[:train_pos_len] + neg_indices[:train_neg_len])
    test_idx = set(pos_indices[train_pos_len:] + neg_indices[train_neg_len:])
    
    X_train_raw = [X_vectors[i] for i in range(len(X_vectors)) if i in train_idx]
    y_train = [targets[i] for i in range(len(targets)) if i in train_idx]
    
    X_test_raw = [X_vectors[i] for i in range(len(X_vectors)) if i in test_idx]
    y_test = [targets[i] for i in range(len(targets)) if i in test_idx]
    
    num_indices = [encoded_feature_names.index(col) for col in numeric_cols]
    means = {}
    stds = {}
    
    for idx in num_indices:
        vals = [row[idx] for row in X_train_raw]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(variance) if variance > 0 else 1.0
        means[idx] = mean
        stds[idx] = std
        
    def scale_dataset(dataset):
        scaled = []
        for row in dataset:
            new_row = list(row)
            for idx in num_indices:
                new_row[idx] = (new_row[idx] - means[idx]) / stds[idx]
            scaled.append(new_row)
        return scaled
        
    X_train_scaled = scale_dataset(X_train_raw)
    X_test_scaled = scale_dataset(X_test_raw)
    
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    def save_matrix(filepath, feature_names, matrix, labels=None):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = feature_names + (['Churn'] if labels is not None else [])
            writer.writerow(header)
            for i, row in enumerate(matrix):
                out_row = list(row)
                if labels is not None:
                    out_row.append(labels[i])
                writer.writerow(out_row)
                
    save_matrix(os.path.join(cleaned_dir, 'train_data.csv'), encoded_feature_names, X_train_scaled, y_train)
    save_matrix(os.path.join(cleaned_dir, 'test_data.csv'), encoded_feature_names, X_test_scaled, y_test)
    
    preprocessor_meta = {
        'feature_names': encoded_feature_names,
        'numeric_cols': numeric_cols,
        'cat_values': cat_values,
        'num_indices': num_indices,
        'means': {encoded_feature_names[i]: means[i] for i in num_indices},
        'stds': {encoded_feature_names[i]: stds[i] for i in num_indices}
    }
    
    with open(os.path.join(models_dir, 'preprocessor.json'), 'w', encoding='utf-8') as f:
        json.dump(preprocessor_meta, f, indent=4)
        
    print(f"Data preprocessed: Train size={len(X_train_scaled)}, Test size={len(X_test_scaled)}, Features={len(encoded_feature_names)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, encoded_feature_names

if __name__ == '__main__':
    paths = [
        os.path.join('data', 'Telco_Customer_Churn_Dataset.csv'),
        os.path.join('Customer-Churn-Prediction', 'data', 'Telco_Customer_Churn_Dataset.csv')
    ]
    data_path = next((p for p in paths if os.path.exists(p)), None)
    if data_path:
        preprocess_data(data_path)
    else:
        print("Data file not found.")
