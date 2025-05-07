import sagemaker
from sagemaker.xgboost import XGBoostModel
from sagemaker.serializers import CSVSerializer
import boto3
import pandas as pd
import os

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::913524944914:role/ChurnPredictionEC2Role"
bucket = "churn-prediction-pipeline"
prefix = "telco-customer-churn"

# Step 1: Create a SageMaker model from training artifacts
model_data = "s3://sagemaker-us-east-1-913524944914/churn-training-ec2-user/output/model.tar.gz"
xgb_model = XGBoostModel(
    model_data=model_data,
    role=role,
    entry_point="xgboost_script.py",
    framework_version="1.5-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

# Inference script for SageMaker endpoint
xgb_script = """
import json
import os
import xgboost as xgb
import numpy as np
from io import StringIO
import pandas as pd

def model_fn(model_dir):
    \"\"\"Load the XGBoost model from the model_dir.\"\"\"
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'xgboost-model'))
    return model

def input_fn(request_body, request_content_type):
    \"\"\"Parse input data.\"\"\"
    if request_content_type == 'text/csv':
        # Convert CSV string to DataFrame
        df = pd.read_csv(StringIO(request_body), header=None)
        return df.values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    \"\"\"Make predictions using the loaded model.\"\"\"
    dmatrix = xgb.DMatrix(input_data)
    return model.predict(dmatrix)

def output_fn(prediction, content_type):
    \"\"\"Format prediction output.\"\"\"
    if content_type == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    elif content_type == 'text/csv':
        return ','.join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def ping():
    \"\"\"Health check endpoint.\"\"\"
    return 'Model is healthy'
"""

with open("xgboost_script.py", "w") as f:
    f.write(xgb_script)

# Step 2: Deploy the model to an endpoint
predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    serializer=CSVSerializer(),
    endpoint_name="churn-prediction-endpoint"
)

# Step 3: Test the endpoint with sample data
# Load a sample from the test set
test_data_s3 = f"s3://{bucket}/{prefix}/output/test/test.csv"
test_df = pd.read_csv(sagemaker_session.boto_session.client("s3").get_object(
    Bucket=bucket,
    Key=f"{prefix}/output/test/test.csv"
)["Body"])
sample = test_df.iloc[0, 1:].values  # First row, excluding churn label

# Predict
prediction = predictor.predict(sample.tolist())
print(f"Sample prediction (0 = No Churn, 1 = Churn): {prediction}")

# Note: Keep the endpoint running for further testing, or delete it to save costs
# To delete: predictor.delete_endpoint()