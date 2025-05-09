
import xgboost as xgb
import pandas as pd
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import json

if __name__ == '__main__':
    train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')
    X_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    
    valid_data = pd.read_csv('/opt/ml/input/data/validation/test.csv')
    X_valid = valid_data.iloc[:, 1:]
    y_valid = valid_data.iloc[:, 0]
    
    model = xgb.XGBClassifier(
        max_depth=int(float(os.environ.get('SM_HP_MAX_DEPTH', 5))),
        eta=float(os.environ.get('SM_HP_ETA', 0.2)),
        min_child_weight=float(os.environ.get('SM_HP_MIN_CHILD_WEIGHT', 1)),
        subsample=float(os.environ.get('SM_HP_SUBSAMPLE', 0.8)),
        objective='binary:logistic',
        num_round=100
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    
    # Save model
    model.save_model('/opt/ml/model/xgboost-model')
    
    # Output metrics
    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    accuracy = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_pred_proba)
    
    # Print metrics in the format expected by SageMaker HPO
    print(f"validation:accuracy: {accuracy}")
    print(f"validation:auc: {auc}")
    
    # Also save metrics to file
    metrics = {'accuracy': accuracy, 'auc': auc}
    with open('/opt/ml/output/metrics.json', 'w') as f:
        json.dump(metrics, f)
