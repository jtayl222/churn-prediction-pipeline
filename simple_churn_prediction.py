import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoost
from sagemaker.inputs import TrainingInput
import boto3
import os
import json
import requests

def get_instance_role():
    # Get token for IMDSv2
    token_url = "http://169.254.169.254/latest/api/token"
    token_headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
    token = requests.put(token_url, headers=token_headers).text
    
    # Get IAM role info with token
    role_info_url = "http://169.254.169.254/latest/meta-data/iam/info"
    role_headers = {"X-aws-ec2-metadata-token": token}
    role_info = json.loads(requests.get(role_info_url, headers=role_headers).text)
    
    # Extract the instance profile ARN and convert to role ARN
    instance_profile_arn = role_info['InstanceProfileArn']
    # Instance profile ARN format: arn:aws:iam::ACCOUNT_ID:instance-profile/ROLE_NAME
    # We need to convert this to: arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME
    role_arn = instance_profile_arn.replace(':instance-profile/', ':role/')
    
    return role_arn

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()

role = get_instance_role()
print(f"Using role: {role}")
# role = sagemaker.get_execution_role()
# role = "arn:aws:iam::<account-id>:role/ChurnPredictionEC2Role"  # Replace <account-id> with your AWS account ID
bucket = "churn-prediction-pipeline"
prefix = "telco-customer-churn"

# Step 1: Preprocess data with SageMaker Processing
processor = Processor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", sagemaker_session.boto_region_name, "0.23-1"),
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    sagemaker_session=sagemaker_session
)

preprocessing_script = """
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('/opt/ml/processing/input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df.drop(['customerID'], axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train = pd.concat([y_train, X_train], axis=1)
test = pd.concat([y_test, X_test], axis=1)

train.to_csv('/opt/ml/processing/output/train/train.csv', index=False)
test.to_csv('/opt/ml/processing/output/test/test.csv', index=False)
"""

with open("preprocessing.py", "w") as f:
    f.write(preprocessing_script)

# Run preprocessing job
processor.run(
    code="preprocessing.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/{prefix}/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/{prefix}/output/train"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{bucket}/{prefix}/output/test"
        )
    ],
    job_name=f"churn-preprocessing-{os.path.basename(os.path.dirname(os.getcwd()))}"
)

# Step 2: Train XGBoost model
xgb_estimator = XGBoost(
    entry_point="xgboost_script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.5-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "max_depth": 5,
        "eta": 0.2,
        "min_child_weight": 1,
        "subsample": 0.8,
        "objective": "binary:logistic",
        "num_round": 100
    }
)

xgb_script = """
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
"""

with open("xgboost_script.py", "w") as f:
    f.write(xgb_script)

# Run training job
xgb_estimator.fit({
    "train": TrainingInput(
        s3_data=f"s3://{bucket}/{prefix}/output/train/train.csv",
        content_type="text/csv"
    ),
    "validation": TrainingInput(
        s3_data=f"s3://{bucket}/{prefix}/output/test/test.csv",
        content_type="text/csv"
    )
}, job_name=f"churn-training-{os.path.basename(os.path.dirname(os.getcwd()))}")

# Print model artifacts location
print(f"Model artifacts saved to: {xgb_estimator.model_data}")