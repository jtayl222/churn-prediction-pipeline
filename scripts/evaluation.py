import pandas as pd
import numpy as np
import xgboost as xgb
import json
import logging
import os
import tarfile
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting evaluation script.")

    # List directories for debugging
    logger.info("Listing directories for debugging:")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files in /opt/ml/processing: {os.listdir('/opt/ml/processing') if os.path.exists('/opt/ml/processing') else 'Directory not found'}")
    logger.info(f"Files in /opt/ml/processing/test: {os.listdir('/opt/ml/processing/test') if os.path.exists('/opt/ml/processing/test') else 'Directory not found'}")
    logger.info(f"Files in /opt/ml/processing/model: {os.listdir('/opt/ml/processing/model') if os.path.exists('/opt/ml/processing/model') else 'Directory not found'}")

    # Load test data
    try:
        logger.info("Loading test data from /opt/ml/processing/test/test.csv")
        test_data = pd.read_csv('/opt/ml/processing/test/test.csv')
        X_test = test_data.iloc[:, 1:]
        y_test = test_data.iloc[:, 0]
        logger.info(f"Test data loaded successfully. Shape: {test_data.shape}")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise

    # Load model
    try:
        t = tarfile.open('/opt/ml/processing/model/model.tar.gz', 'r:gz')
        t.extractall()
        logger.info(f"Files in CWD: {os.listdir('.')}")
        logger.info("Loading model from xgboost-model")
        model = xgb.XGBClassifier()
        model.load_model('xgboost-model')
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Make predictions
    try:
        logger.info("Making predictions on test data.")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        logger.info("Predictions completed successfully.")
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        raise

    # Calculate metrics
    try:
        logger.info("Calculating evaluation metrics.")
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        logger.info(f"Evaluation metrics calculated: {metrics}")
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        raise

    # Save metrics to /opt/ml/processing/evaluation/evaluation.json
    try:
        output_path = '/opt/ml/processing/evaluation/evaluation.json'
        logger.info(f"Saving evaluation metrics to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(metrics, f)
        logger.info("Evaluation metrics saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save evaluation metrics: {e}")
        raise
