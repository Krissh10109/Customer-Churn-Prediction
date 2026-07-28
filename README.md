# Customer Churn Prediction Project

This repository contains an end-to-end Machine Learning pipeline to analyze customer demographics and billing records, predict the risk of churn, and assist retention strategy.

## Project Structure
```
Customer-Churn-Prediction/
│
├── data/
│   └── Telco_Customer_Churn_Dataset.csv   # Telco Churn Dataset
│
├── notebooks/
│   └── EDA.ipynb                          # Exploratory Data Analysis
│
├── src/
│   ├── data_preprocessing.py               # Preprocessing pipelines & data split
│   ├── feature_selection.py               # Feature relevance analysis
│   ├── train_model.py                     # Hyperparameter-tuned model training
│   ├── evaluate_model.py                  # Evaluation metrics & visual plots
│   └── utils.py                           # Helper & IO functions
│
├── models/
│   └── churn_model.pkl                    # Pickled trained ML pipeline
│
├── outputs/
│   ├── confusion_matrix.png               # Evaluation: Confusion Matrix
│   ├── roc_curve.png                      # Evaluation: ROC-AUC Curve
│   ├── feature_importance.png             # Evaluation: Feature Importance scores
│   └── evaluation_metrics.txt             # Text log of computed metrics
│
├── requirements.txt                       # Core dependencies
├── README.md                              # Main documentation
└── app.py                                 # Interactive Streamlit Web Application
```

## How to Set Up and Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   Run the training script to perform hyperparameter tuning (GridSearchCV) and save the trained pipeline to the `models/` folder:
   ```bash
   python src/train_model.py
   ```

3. **Evaluate the Model**:
   Generate performance figures and classification metrics output:
   ```bash
   python src/evaluate_model.py
   ```
   Check the `outputs/` directory for results.

4. **Run the Streamlit Dashboard**:
   Run the interactive web application to predict churn for individual customers:
   ```bash
   streamlit run app.py
   ```
