import xgboost as xgb
import pandas as pd
import os
import logging
from sklearn.metrics import accuracy_score, roc_auc_score
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting XGBoost training script.")

    # Load training data
    logger.info("Loading training data from /opt/ml/input/data/train/train.csv")
    train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')
    X_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    logger.info(f"Training data loaded. Shape: {train_data.shape}")

    # Load validation data
    logger.info("Loading validation data from /opt/ml/input/data/validation/test.csv")
    valid_data = pd.read_csv('/opt/ml/input/data/validation/test.csv')
    X_valid = valid_data.iloc[:, 1:]
    y_valid = valid_data.iloc[:, 0]
    logger.info(f"Validation data loaded. Shape: {valid_data.shape}")

    # Initialize XGBoost model
    logger.info("Initializing XGBoost model with hyperparameters.")
    model = xgb.XGBClassifier(
        max_depth=int(float(os.environ.get('SM_HP_MAX_DEPTH', 5))),
        eta=float(os.environ.get('SM_HP_ETA', 0.2)),
        min_child_weight=float(os.environ.get('SM_HP_MIN_CHILD_WEIGHT', 1)),
        subsample=float(os.environ.get('SM_HP_SUBSAMPLE', 0.8)),
        objective='binary:logistic',
        num_round=100
    )
    logger.info("XGBoost model initialized.")

    # Train the model
    logger.info("Starting model training.")
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    logger.info("Model training completed.")

    # Save the model
    logger.info("Saving the trained model to /opt/ml/model/xgboost-model")
    model.save_model('/opt/ml/model/xgboost-model')
    logger.info("Model saved successfully.")

    # Evaluate the model
    logger.info("Evaluating the model on validation data.")
    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    accuracy = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_pred_proba)
    logger.info(f"Model evaluation completed. Accuracy: {accuracy}, AUC: {auc}")

    # Print metrics in the format expected by SageMaker HPO
    print(f"validation:accuracy: {accuracy}")
    print(f"validation:auc: {auc}")

    # Save metrics to file
    logger.info("Saving evaluation metrics to /opt/ml/output/metrics.json")
    metrics = {'accuracy': accuracy, 'auc': auc}
    with open('/opt/ml/output/metrics.json', 'w') as f:
        json.dump(metrics, f)
    logger.info("Metrics saved successfully.")
