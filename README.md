# Telco Customer Churn Prediction with SageMaker

This project uses Amazon SageMaker to build a machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. The initial implementation focuses on preprocessing the dataset and training an XGBoost model.

```bash
$ ssh -i ~/.ssh/ec2-churn-prediction-pipeline.pem ec2-user@35.172.191.151

$ scp -i ~/.ssh/ec2-churn-prediction-pipeline.pem churn_prediction_pipeline.py ec2-user@35.172.191.151:~/churn_prediction_pipeline.py
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

- Monitor AWS Billing for EC2, SageMaker Processing, and Training costs.
- Stop the EC2 instance when done:
  ```bash
  aws ec2 stop-instances --instance-ids <instance-id>
  ```

## Troubleshooting

- **IAM Issues**: Verify `ChurnPredictionEC2Role` permissions and trust policy.
- **S3 Access**: Ensure `AmazonS3FullAccess` is attached.
- **Logs**: Check SageMaker Console → Processing/Training Jobs → View logs.
- **SDK Version**:
  ```bash
  pip show sagemaker
  pip install --upgrade sagemaker
  ```
