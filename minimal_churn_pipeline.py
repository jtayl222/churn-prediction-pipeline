import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TuningStep, CreateModelStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
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

# Step 2: Hyperparameter Tuning
xgb_estimator = XGBoost(
    entry_point="xgboost_script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.5-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100
    }
)

xgb_script = """
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
"""

with open("xgboost_script.py", "w") as f:
    f.write(xgb_script)

tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    hyperparameter_ranges={
        "max_depth": sagemaker.tuner.IntegerParameter(3, 10),
        "eta": sagemaker.tuner.ContinuousParameter(0.1, 0.5),
        "min_child_weight": sagemaker.tuner.ContinuousParameter(1, 10),
        "subsample": sagemaker.tuner.ContinuousParameter(0.5, 1.0)
    },
    metric_definitions=[
        {"Name": "validation:auc", "Regex": "validation:auc: ([0-9\\.]+)"},
        {"Name": "validation:accuracy", "Regex": "validation:accuracy: ([0-9\\.]+)"}
    ],
    max_jobs=3,
    max_parallel_jobs=1
)

step_tune = TuningStep(
    name="TuneModel",
    tuner=tuner,
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

# Step 3: Create and register model
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=sagemaker_session.boto_region_name,
    version="1.5-1",
    instance_type="ml.m5.xlarge"
)
model_data=step_tune.get_top_model_s3_uri(top_k=0, s3_bucket=bucket).to_string()

# Use RegisterModel from sagemaker.model instead
step_register_model = RegisterModel(
    name="RegisterModel",
    estimator=xgb_estimator,
    model_data=model_data,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="ChurnPredictionModels",
    approval_status="PendingManualApproval"
)

# Step 4: Evaluation
evaluation_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", sagemaker_session.boto_region_name, "0.23-1"),
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)

evaluation_code = """
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
"""

with open("evaluation.py", "w") as f:
    f.write(evaluation_code)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

step_evaluate = ProcessingStep( 
    name="EvaluateModel",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=step_tune.get_top_model_s3_uri(top_k=0, s3_bucket=bucket),
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri.to_string(),
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{bucket}/{prefix}/evaluation"
        )
    ],
    code="evaluation.py",
    property_files=[evaluation_report]  # Include property_files here
)

# Condition step for model accuracy - Use JsonGet to access property
cond_auc = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate.name,
        property_file=evaluation_report,
        json_path="roc_auc"
    ),
    right=0.8
)

# Deploy endpoint if condition is met
step_deploy = ConditionStep(
    name="DeployModelIfAucAboveThreshold",
    conditions=[cond_auc],
    if_steps=[],  # Add deployment steps here
    else_steps=[]
)

# Step 5: Define and Execute Pipeline
pipeline_name = "ChurnPredictionPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    steps=[step_process, step_tune, step_register_model, step_evaluate, step_deploy],
    sagemaker_session=sagemaker_session
)

# Upsert and execute pipeline
try:
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait()
    print(f"Pipeline execution ARN: {execution.arn}")
except Exception as e:
    print(f"Pipeline execution failed: {str(e)}")
