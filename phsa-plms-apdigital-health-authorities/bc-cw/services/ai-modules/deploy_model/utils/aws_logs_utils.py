"""
Utility functions for managing AWS CloudWatch log groups using boto3.
Provides helpers to create log groups and set retention policies for SageMaker and other AWS services.
"""

import boto3
from botocore.exceptions import ClientError

def get_logs_client(profile, region):
    """
    Create and return a boto3 CloudWatch Logs client for the given AWS profile and region.
    Args:
        profile (str): AWS CLI profile name. If None, use default credentials.
        region (str): AWS region name (e.g., 'ca-central-1').
    Returns:
        boto3.client: CloudWatch Logs client.
    """
    if profile:
        boto_session = boto3.Session(profile_name=profile, region_name=region)
    else:
        boto_session = boto3.Session(region_name=region)
    return boto_session.client("logs")

def create_log_group(logs_client, log_group_name):
    """
    Create a CloudWatch log group with 1 day retention policy.
    If the log group already exists, the error is ignored.
    Args:
        logs_client (boto3.client): CloudWatch Logs client.
        log_group_name (str): Name of the log group to create.
    """
    try:
        logs_client.create_log_group(logGroupName=log_group_name)
    except ClientError as e:
        if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
            raise
    logs_client.put_retention_policy(logGroupName=log_group_name, retentionInDays=1)
