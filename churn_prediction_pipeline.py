import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner
from sagemaker.xgboost import XGBoost
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import Processor
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.model_step import ModelStep
import boto3
import os
from uuid import uuid4

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::913524944914:role/ChurnPredictionEC2Role"
bucket = "churn-prediction-pipeline"
prefix = "telco-customer-churn"

# Step 1: Preprocess data with SageMaker Processing
processor = Processor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", sagemaker_session.boto_region_name, "0.23-1"),
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium"
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

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    inputs=[sagemaker.processing.ProcessingInput(
        source=f"s3://{bucket}/{prefix}/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        destination="/opt/ml/processing/input"
    )],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train", source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/{prefix}/output/train"
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="test", source="/opt/ml/processing/output/test",
            destination=f"s3://{bucket}/{prefix}/output/test"
        )
    ],
    code="preprocessing.py"
)

# Step 2: Train XGBoost model with hyperparameter tuning
xgb_estimator = XGBoost(
    entry_point="xgboost_script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.5-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

xgb_script = """
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
"""

with open("xgboost_script.py", "w") as f:
    f.write(xgb_script)

hyperparameter_ranges = {
    "max_depth": sagemaker.tuner.IntegerParameter(3, 10),
    "eta": sagemaker.tuner.ContinuousParameter(0.1, 0.5),
    "min_child_weight": sagemaker.tuner.IntegerParameter(1, 10),
    "subsample": sagemaker.tuner.ContinuousParameter(0.5, 1.0)
}

tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[{"Name": "validation:auc", "Regex": "validation:auc ([0-9.]+)"}],
    max_jobs=4,
    max_parallel_jobs=2
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Step 3: Register model
model = sagemaker.model.Model(
    image_uri=xgb_estimator.training_image_uri(),
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=sagemaker_session
)

register_step = ModelStep(
    name="RegisterModel",
    model=model
)

# Step 4: Lambda for deployment
lambda_client = boto3.client("lambda")
lambda_function_name = f"deploy-churn-model-{uuid4()}"

lambda_code = """
import json
import boto3

def lambda_handler(event, context):
    sagemaker_client = boto3.client('sagemaker')
    endpoint_name = 'churn-prediction-endpoint'

    try:
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=event['endpoint_config_name']
        )
        return {'statusCode': 200, 'body': json.dumps('Endpoint created')}
    except sagemaker_client.exceptions.ClientError as e:
        if 'ResourceInUse' in str(e):
            sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=event['endpoint_config_name']
            )
            return {'statusCode': 200, 'body': json.dumps('Endpoint updated')}
        raise e
"""

with open("deploy_lambda.py", "w") as f:
    f.write(lambda_code)

lambda_client.create_function(
    FunctionName=lambda_function_name,
    Runtime="python3.8",
    Role=role,
    Handler="deploy_lambda.lambda_handler",
    Code={"ZipFile": open("deploy_lambda.py", "rb").read()},
    Timeout=30
)

deploy_step = LambdaStep(
    name="DeployModel",
    lambda_func=sagemaker.lambda_helper.Lambda(
        function_arn=f"arn:aws:lambda:{sagemaker_session.boto_region_name}:{sagemaker_session.account_id}:function:{lambda_function_name}"
    ),
    inputs={"endpoint_config_name": register_step.properties.ModelName + "-config"},
    outputs={}
)

# Step 5: Define pipeline
pipeline = Pipeline(
    name="ChurnPredictionPipeline",
    steps=[processing_step, training_step, register_step, deploy_step],
    sagemaker_session=sagemaker_session
)

# Execute pipeline
pipeline.upsert(role_arn=role)
pipeline.start()

# Step 6: Monitor with CloudWatch
cloudwatch_client = boto3.client("cloudwatch")

cloudwatch_client.put_metric_alarm(
    AlarmName="ChurnModelAccuracyAlarm",
    MetricName="validation:auc",
    Namespace="AWS/SageMaker",
    Statistic="Average",
    Threshold=0.7,
    ComparisonOperator="LessThanThreshold",
    Period=3600,
    EvaluationPeriods=1,
    AlarmActions=[]  # Add SNS topic ARN if needed
)