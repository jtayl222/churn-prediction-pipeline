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
from stepfunctions.steps.sagemaker import EndpointConfigStep, EndpointStep
from sagemaker.workflow.fail_step import FailStep    # add this import
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
    code="scripts/preprocessing.py"
)

# Step 2: Hyperparameter Tuning
xgb_estimator = XGBoost(
    entry_point="scripts/xgboost_script.py",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.5-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100
    },
    output_path=f"s3://{bucket}/{prefix}/hpo"    # <-- move output_path here
)

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
model_data = step_tune.get_top_model_s3_uri(
    top_k=0,
    s3_bucket=bucket,
    prefix="telco-customer-churn/hpo"
).to_string()

# Change the name to distinguish from any RegisterModel subcomponent
step_register_model = RegisterModel(
    name="UniqueModelPackageRegistrationStep", # <--- Further revised for uniqueness
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
    image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=sagemaker_session.boto_region_name,
            version="1.5-1"
        ),
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)

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
            source=step_tune.get_top_model_s3_uri(
                top_k=0,
                s3_bucket=bucket,
                prefix="telco-customer-churn/hpo"
            ).to_string(),
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
    code="scripts/evaluation.py",
    property_files=[evaluation_report]
)

# Create model step with a unique name that doesn't conflict
step_create_model = CreateModelStep(
    name="UniqueConditionalModelCreationStep", # <--- Further revised for uniqueness
    model=Model(
        image_uri=sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=sagemaker_session.boto_region_name,
            version="1.5-1"
        ),
        model_data=step_tune.get_top_model_s3_uri(
            top_k=0, 
            s3_bucket=bucket,
            prefix="telco-customer-churn/hpo"
        ).to_string(),
        role=role,
        sagemaker_session=sagemaker_session
    )
)

# Endpoint configuration step
step_endpoint_config = EndpointConfigStep(
    state_id="CreateEndpointConfig", # This is a state_id, not a pipeline step 'name'
    endpoint_config_name="ChurnPredictionEndpointConfig",
    model_name=step_create_model.properties.ModelName, # This will refer to "UniqueConditionalModelCreationStep"
    initial_instance_count=1,
    instance_type="ml.t2.medium"
)

# Endpoint step
step_endpoint = EndpointStep(
    state_id="CreateEndpoint", # This is a state_id
    endpoint_name="ChurnPredictionEndpoint",
    endpoint_config_name="ChurnPredictionEndpointConfig"
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

# Define a failure step for the else branch
step_fail = FailStep(
    name="FailDeployment",
    error_message="Model AUC below 0.8 – skipping endpoint deployment.",
)

# Deploy endpoint if condition is met
step_deploy = ConditionStep(
    name="DeployModelIfAucAboveThreshold",
    conditions=[cond_auc],
    if_steps=[step_create_model, step_endpoint_config, step_endpoint],
    else_steps=[step_fail]
)

# Step 5: Define and Execute Pipeline
pipeline_name = "ChurnPredictionPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    steps=[
        step_process,
        step_tune,
        step_register_model, # Name is now "UniqueModelPackageRegistrationStep"
        step_evaluate,
        step_deploy
    ],
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
