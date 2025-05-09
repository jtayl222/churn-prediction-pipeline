
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load test data
test_data = pd.read_csv('/opt/ml/processing/test/test.csv')
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Load model
model = xgb.XGBClassifier()
model.load_model('/opt/ml/processing/model/xgboost-model')

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1': float(f1_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
}

# Save metrics
with open('/opt/ml/processing/evaluation/evaluation.json', 'w') as f:
    json.dump(metrics, f)
