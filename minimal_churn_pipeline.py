import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
import boto3
import os

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::913524944914:role/ChurnPredictionEC2Role"
bucket = "churn-prediction-pipeline"
prefix = "telco-customer-churn"

# Step 1: Preprocessing
processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", sagemaker_session.boto_region_name, "0.23-1"),
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
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

step_process = ProcessingStep(
    name="PreprocessData",
    processor=processor,
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
    code="preprocessing.py"
)

# Step 2: Training
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

step_train = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri.to_string(),
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri.to_string(),
            content_type="text/csv"
        )
    }
)

# Step 3: Register Model directly (skip model creation step)
model_package_group_name = "ChurnPredictionModelGroup"

# Create ModelPackageGroup if it doesn't exist
try:
    sagemaker_client = boto3.client("sagemaker")
    sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription="Churn prediction models"
    )
    print(f"Created ModelPackageGroup: {model_package_group_name}")
except Exception as e:
    if "ResourceInUse" in str(e):
        print(f"ModelPackageGroup {model_package_group_name} already exists")
    else:
        print(f"Error creating ModelPackageGroup: {e}")

# Use RegisterModel step collection - this handles pipeline properties correctly
step_register = RegisterModel(
    name="RegisterModel",
    estimator=xgb_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status="PendingManualApproval"
)

# Pipeline definition - simplified
pipeline_name = "ChurnPredictionPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    steps=[step_process, step_train, step_register],
    sagemaker_session=sagemaker_session
)

# Upsert and execute pipeline
pipeline.upsert(role_arn=role)
execution = pipeline.start()

# Wait for pipeline completion (optional, for debugging)
execution.wait()

# Print pipeline execution ARN
print(f"Pipeline execution ARN: {execution.arn}")