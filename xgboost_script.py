
import xgboost as xgb
import pandas as pd
import os

def handler(event, context):
    train_data = pd.read_csv(os.path.join(event['data_dir'], 'train.csv'))
    X_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    model.save_model(os.path.join(event['model_dir'], 'xgboost-model'))
