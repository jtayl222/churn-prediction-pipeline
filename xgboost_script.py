
import xgboost as xgb
import pandas as pd
import os

if __name__ == '__main__':
    train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')
    X_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    
    model = xgb.XGBClassifier(
        max_depth=5,
        eta=0.2,
        min_child_weight=1,
        subsample=0.8,
        objective='binary:logistic',
        num_round=100
    )
    model.fit(X_train, y_train)
    
    model.save_model('/opt/ml/model/xgboost-model')
