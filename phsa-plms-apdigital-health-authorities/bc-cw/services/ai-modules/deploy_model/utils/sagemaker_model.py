import boto3
from botocore.exceptions import ClientError

def get_sagemaker_client(profile, region):
    """
    Create and return a boto3 SageMaker client for the given AWS profile and region.
    """
    if profile:
        boto_session = boto3.Session(profile_name=profile, region_name=region)
    else:
        boto_session = boto3.Session(region_name=region)
    return boto_session.client("sagemaker")

def create_model(sagemaker_client, model_name, image_uri, role_arn):
    """
    Create a SageMaker model with the specified parameters.
    """
    try:
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={'Image': image_uri},
            ExecutionRoleArn=role_arn
        )
        print(f"Created model: {model_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in str(e):
            print(f"Model {model_name} already exists.")
        else:
            raise
