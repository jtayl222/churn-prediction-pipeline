import kagglehub
import boto3
import os

# Download dataset once
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
dataset_file = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")  # Verify filename

# Upload to S3
s3 = boto3.client("s3")
bucket = "churn-prediction-pipeline"  # Replace with your bucket name
s3_key = "telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"

s3.upload_file(dataset_file, bucket, s3_key, ExtraArgs={"StorageClass": "STANDARD_IA"})

print(f"Dataset uploaded to s3://{bucket}/{s3_key}")