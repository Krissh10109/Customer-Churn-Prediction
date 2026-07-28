import os
import joblib

def ensure_dir(file_path):
    """
    Ensure that the directory for the given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_model(model, filepath):
    """
    Save model using joblib.
    """
    ensure_dir(filepath)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load model using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")
    return joblib.load(filepath)
