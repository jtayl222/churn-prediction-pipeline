# AWS CLI Examples
```bash

   aws cloudwatch put-metric-alarm --alarm-name PipelineFailure --metric-name PipelineExecutionStatus --namespace AWS/SageMaker --threshold 1 --comparison-operator GreaterThanThreshold --evaluation-periods 1 --period 300 --statistic Maximum --alarm-actions $SNS_TOPIC_ARN --region us-east-1
   export SNS_TOPIC_ARN=arn:aws:sns:us-east-1:913524944914:PipelineFailureTopic


/usr/bin/python3 /home/ec2-user/.vscode-server/extensions/ms-python.python-2025.6.0-linux-x64/python_files/printEnvVariablesToFile.py /home/ec2-user/.vscode-server/extensions/ms-python.python-2025.6.0-linux-x64/python_files/deactivate/bash/envVars.txt
aws

aws iam get-account-authorization-details > output.json
aws iam get-policy --policy-arn arn:aws:iam::913524944914:policy/AmazonSageMakerManageAccessPolicy-603ju42csr2fvr --region us-east-1
aws iam get-policy --policy-arn arn:aws:iam::913524944914:policy/service-role/AmazonSageMakerManageAccessPolicy-603ju42csr2fvr --region us-east-1
aws iam get-policy-version --policy-arn arn:aws:iam::913524944914:policy/service-role/AmazonSageMakerManageAccessPolicy-603ju42csr2fvr --version-id v1 --region us-east-1
aws iam get-role --role-name ChurnPredictionEC2Role --region us-east-1
aws iam list-entities-for-policy --policy-arn arn:aws:iam::913524944914:policy/service-role/AmazonSageMakerManageAccessPolicy-603ju42csr2fvr --region us-east-1
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/Pipelines --region us-east-1
aws logs filter-log-events --log-group-name /aws/sagemaker/Pipelines --log-stream-name-prefix "arn:aws:sagemaker:us-east-1:913524944914:pipeline/ChurnPredictionPipeline/execution/p5wc6y4i34jq" --region us-east-1
aws logs filter-log-events --log-group-name /aws/sagemaker/Pipelines --log-stream-name-prefix <pipeline-execution-arn> --region us-east-1
aws logs filter-log-events --log-group-name /aws/sagemaker/ProcessingJobs --log-stream-name-prefix pipeline-abc123-PreprocessData --region us-east-1
aws logs filter-log-events --log-group-name /aws/sagemaker/ProcessingJobs --log-stream-name-prefix pipeline-p5wc6y4i34jq-EvaluateModel --region us-east-1
aws logs filter-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix <tuning-job-name> --region us-east-1
aws s3 ls
aws s3 ls --recursive
aws s3 ls churn-prediction-pipeline --recursive
aws s3 ls s3://churn-prediction-pipeline/telco-customer-churn --recursive
aws s3 ls s3://churn-prediction-pipeline/telco-customer-churn/
aws s3 ls s3://churn-prediction-pipeline/telco-customer-churn/evaluation/
aws s3 ls s3://churn-prediction-pipeline/telco-customer-churn/evaluation/ --recursive
aws s3 ls s3://churn-prediction-pipeline/telco-customer-churn/output/
aws s3 ls s3://churn-prediction-pipeline/telco-customer-churn/output/test/
aws s3 ls s3://sagemaker-us-east-1-913524944914/

########

aws sagemaker list-hyper-parameter-tuning-jobs --region us-east-1 --sort-by CreationTime --sort-order Descending --max-items 1
aws sagemaker list-model-packages --model-package-group-name ChurnPredictionModelGroup --region us-east-1
aws sagemaker list-processing-jobs --region us-east-1 --sort-by CreationTime --sort-order Descending --max-items 5
aws sagemaker list-processing-jobs --region us-east-1 --sort-by CreationTime --sort-order Descending --max-items 5 --query 'ProcessingJobSummaries[?CreationTime>=`2025-05-09T21:52:00Z`]'
aws sagemaker list-processing-jobs --region us-east-1 --sort-by CreationTime --sort-order Descending --max-items 5 | grep ProcessingJobStatus
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name p5wc6y4i34jq-TuneMode-VCGaM6Q1ad --region us-east-1
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name p5wc6y4i34jq-TuneMode-VCGaM6Q1ad --region us-east-1 --query 'TrainingJobSummaries[*].TrainingJobName'
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name p5wc6y4i34jq-TuneMode-VCGaM6Q1ad --region us-east-1 | grep ObjectiveStatus
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name pipeline-abc123-TuneModel-001 --region us-east-1
aws service-quotas list-service-quotas --service-code sagemaker --region us-east-1

########

aws cloudformation deploy --template-file iac/churn_prediction_role.yaml --stack-name ChurnPredictionRoleStack --region us-east-1 --capabilities CAPABILITY_NAMED_IAM

#########

aws sns list-subscriptions-by-topic --topic-arn arn:aws:sns:us-east-1:913524944914:PipelineFailureTopic --region us-east-1
aws sns publish --topic-arn arn:aws:sns:us-east-1:913524944914:PipelineFailureTopic --message "Test notification" --region us-east-1

##########

aws sagemaker list-pipeline-executions --pipeline-name ChurnPredictionPipeline --region us-east-1 --sort-by CreationTime --sort-order Descending --max-items 1
export PipelineExecutionArn="arn:aws:sagemaker:us-east-1:913524944914:pipeline/ChurnPredictionPipeline/execution/k0ov40edd2ds"
aws sagemaker describe-pipeline-execution --pipeline-execution-arn $PipelineExecutionArn --region us-east-1
aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn $PipelineExecutionArn

#########

aws iam list-attached-user-policies --user jtayl22med@gmail.com
aws iam get-policy --policy-arn arn:aws:iam::913524944914:policy/ManageOwnAccessKeys
aws iam get-policy-version --version-id  v21 --policy-arn arn:aws:iam::913524944914:policy/ManageOwnAccessKeys

#######

aws iam list-attached-role-policies --role-name ChurnPredictionEC2Role --region us-east-1
aws iam list-role-policies --role-name ChurnPredictionEC2Role --region us-east-1
aws iam  get-role-policy --role-name ChurnPredictionEC2Role --region us-east-1 --policy-name SageMakerModelRegistryPolicy

#######
aws logs describe-log-streams --log-group-name /aws/sagemaker/ProcessingJobs  --order-by LastEventTime --descending  --max-items 1
aws logs get-log-events --log-group-name /aws/sagemaker/ProcessingJobs --log-stream-name pipelines-hd1ew8n7jr17-EvaluateModel-OxEiQ6lREC/algo-1-1746921595
```