# Telco Customer Churn Prediction with SageMaker

This project uses Amazon SageMaker to build a machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. The initial implementation focuses on preprocessing the dataset and training an XGBoost model.

## SET UP

```bash
# EC2 Public IPv4 address
REMOTE_IP=34.203.200.200

# REMOTE SHELL
$ ssh -i ~/.ssh/ec2-churn-prediction-pipeline.pem ec2-user@@REMOTE_IP

# REMOTE COPY EXAMPLE
$ scp -i ~/.ssh/ec2-churn-prediction-pipeline.pem churn_prediction_pipeline.py ec2-user@REMOTE_IP:~/churn_prediction_pipeline.py

# FOR VS-CODE REMOTE SSH
$ cat ~/.ssh/config 
Host EC2 Churn Prediction
  HostName 34.203.200.200
  User ec2-user
  IdentityFile ~/.ssh/ec2-churn-prediction-pipeline.pem
  Port 22

# GIT CLONE in REMOTE SHELL
$ ssh-keygen -t rsa
$ cat ~/.ssh/id_rsa.pub # copy this key to https://github.com/settings/keys
$ sudo yum install git
$ git clone git@github.com:jtayl222/churn-prediction-pipeline.git
```


## Prerequisites

- **EC2 Instance**: Running in `us-east-1` with `ChurnPredictionEC2Role` attached.
- **IAM Role**: `ChurnPredictionEC2Role` with:
  - Policies: `AmazonS3FullAccess`, `AmazonSageMakerFullAccess`, `AWSLambda_FullAccess`, `CloudWatchFullAccess`, and a custom `iam:PassRole` policy.
  - Trust policy:
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": [
                        "ec2.amazonaws.com",
                        "sagemaker.amazonaws.com"
                    ]
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    ```
- **S3 Bucket**: `churn-prediction-pipeline` containing the dataset at `s3://churn-prediction-pipeline/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Python Environment**: Virtual environment with SageMaker SDK (`sagemaker>=2.100.0`).

## Setup

1. **Set Up CloudWatch Monitoring**:
   * Create SNS Topic
      * https://us-east-1.console.aws.amazon.com/sns/v3/home?region=us-east-1#/homepage 
      * PipelineFailureTopic

   ```bash
   # arn:aws:sns:<region>:<account-id>:<topic-name>
   export SNS_TOPIC_ARN=arn:aws:sns:us-east-1:913524944914:PipelineFailureTopic
   
   aws cloudwatch put-metric-alarm --alarm-name PipelineFailure --metric-name PipelineExecutionStatus --namespace AWS/SageMaker --threshold 1 --comparison-operator GreaterThanThreshold --evaluation-periods 1 --period 300 --statistic Maximum --alarm-actions $SNS_TOPIC_ARN --region us-east-1
   ```

1. **Launch EC2 Instance**:
   - Use `us-east-1`, attach `ChurnPredictionEC2Role`, and enable IMDSv2.
   - SSH into the instance:
     ```bash
     ssh -i <key.pem> ec2-user@<instance-public-ip>
     ```

2. **Set Up Environment**:
   ```bash
   python3 -m venv sagemaker_env
   source sagemaker_env/bin/activate
   pip install --upgrade pip
   pip install sagemaker boto3
   ```

3. **Verify Credentials**:
   ```bash
   aws sts get-caller-identity
   ```
   Expected: ARN includes `ChurnPredictionEC2Role`.

4. **Remove User Credentials** (if present):
   ```bash
   rm ~/.aws/credentials ~/.aws/config
   unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
   ```

## Running the Script

1. **Upload Script**:
   ```bash
   scp -i <key.pem> simple_churn_prediction.py ec2-user@<instance-public-ip>:~/sagemaker_env/
   ```

2. **Run**:
   ```bash
   cd ~/sagemaker_env
   source sagemaker_env/bin/activate
   python3 simple_churn_prediction.py
   ```

3. **Outputs**:
   - Preprocessed data: `s3://churn-prediction-pipeline/telco-customer-churn/output/train/` and `test/`.
   - Model artifacts: `s3://sagemaker-us-east-1-913524944914/churn-training-ec2-user/output/model.tar.gz`.
   - Check SageMaker Console → Processing Jobs and Training Jobs.

## Current Functionality

- **Preprocessing**: Uses `ScriptProcessor` to clean the Telco Churn dataset, encode categorical variables, and split into train/test sets.
- **Training**: Trains an XGBoost model with fixed hyperparameters using SageMaker’s managed training.

## Next Steps

- Deploy the trained model to a SageMaker endpoint for inference.
- Add hyperparameter tuning with `HyperparameterTuner`.
- Implement a SageMaker Pipeline with `ProcessingStep`, `TrainingStep`, and `ModelStep`.
- Add monitoring with CloudWatch alarms.

## Cost Management

### Monitor AWS Billing for EC2, SageMaker Processing, and Training costs.

1. Environment:
   ```bash
   # Adjust for your environment
   export INSTANCE_ID=i-0bafe0043c71493cd
   ```

1. Start EC2 (if stopped):
   ```bash
   aws ec2 start-instances --instance-ids $INSTANCE_ID --region us-east-1
   ```

1. Stop EC2 When Done:
   ```bash
   aws ec2 stop-instances --instance-ids $INSTANCE_ID --region us-east-1
   ```

1. Check Endpoints:
   ```bash
   aws sagemaker list-endpoints --region us-east-1
   ```

1. Delete churn-prediction-endpoint if present:
   ```bash
   aws sagemaker delete-endpoint --endpoint-name churn-prediction-endpoint --region us-east-1
   ```

1. Monitor Jobs:
   ```bash
   aws sagemaker list-processing-jobs --region us-east-1 --status-equals InProgress
   aws sagemaker list-training-jobs --region us-east-1 --status-equals InProgress
   aws sagemaker list-hyper-parameter-tuning-jobs --region us-east-1 --status-equals InProgress
   ```
    
1. Cost Estimate:

* Preprocessing: ml.t3.medium (~$0.05 for ~5 minutes).
* Tuning: 3 jobs on ml.m5.xlarge (~$0.20 each, ~$0.60 total).
* Evaluation: ml.t3.medium (~$0.05).
* EC2: t3.medium (~$0.04/hour).
* Total per run: ~$0.70, plus EC2.

## Troubleshooting

1. Check Logs:
   * SageMaker Console → Pipelines → ChurnPredictionPipeline → Executions.
   * Tuning job: SageMaker Console → Training → Hyperparameter tuning jobs → <TuneModel-job-name>.

1. Verify Inputs:
   ```bash
   aws s3 ls s3://churn-prediction-pipeline/telco-customer-churn/output/
   ```

1. IAM:
   ```bash
   # Ensure ChurnPredictionEC2Role has AmazonSageMakerFullAccess, AmazonS3FullAccess.
   aws sts get-caller-identity
   ```

1. Validation File:

   * Check if /opt/ml/input/data/validation/test.csv is accessed correctly in xgboost_script.py.

1.  **IAM Issues**: Verify `ChurnPredictionEC2Role` permissions and trust policy.

1.  **S3 Access**: Ensure `AmazonS3FullAccess` is attached.

1.  **Logs**: Check SageMaker Console → Processing/Training Jobs → View logs.

1.  **SDK Version**:
   ```bash
   pip show sagemaker
   pip install --upgrade sagemaker
   ```
