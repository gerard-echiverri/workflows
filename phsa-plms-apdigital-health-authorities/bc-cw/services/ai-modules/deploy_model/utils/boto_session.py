"""
AWS Boto3 Session Utilities

Provides helper functions for creating boto3 sessions with support for
multiple authentication methods (AWS Profile, access keys, session tokens).
"""

import os
import boto3
from typing import Optional


def get_boto_session(region: Optional[str] = None, role_arn=None) -> boto3.Session:
    """
    Create boto3 session with automatic credential detection.
    
    Authentication priority order:
    1. AWS_PROFILE environment variable
    2. AWS_SESSION_TOKEN (with region)
    3. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (with region)
    4. Default credential chain (with region)
    
    Args:
        region: AWS region name. If None, uses AWS_REGION env var or default
    
    Returns:
        boto3.Session: Configured boto3 session
    
    Environment Variables:
        AWS_PROFILE: AWS CLI profile name (highest priority)
        AWS_SESSION_TOKEN: Temporary session token
        AWS_ACCESS_KEY_ID: AWS access key ID
        AWS_SECRET_ACCESS_KEY: AWS secret access key
        AWS_REGION: Default AWS region
    
    Example:
        >>> # Using AWS profile
        >>> os.environ['AWS_PROFILE'] = 'dev'
        >>> session = get_boto_session()
        >>> s3 = session.client('s3')
        
        >>> # Using explicit credentials
        >>> os.environ['AWS_ACCESS_KEY_ID'] = '...'
        >>> os.environ['AWS_SECRET_ACCESS_KEY'] = '...'
        >>> session = get_boto_session(region='us-west-2')
        >>> sagemaker = session.client('sagemaker')
    """
    # Get region from parameter or environment
    if region is None:
        region = os.environ.get("AWS_REGION")
    
    # Priority 1: AWS Profile
    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        return boto3.Session(profile_name=aws_profile, region_name=region)
    
    # Priority 2: Session Token
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    if aws_session_token:
        return boto3.Session(
            region_name=region,
            aws_session_token=aws_session_token
        )
    
    # Priority 3: If role_arn is provided, uses STS to assume the role.
    if role_arn:
        sts = boto3.client("sts", region_name=region)
        assumed = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="sagemaker-cicd-session"
        )
        creds = assumed["Credentials"]
        return boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=region
        )
    
    # Priority 4: Access Key + Secret Key
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if aws_access_key_id and aws_secret_access_key:
        return boto3.Session(
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    
    # Priority 5: Default credential chain
    return boto3.Session(region_name=region)


def create_clients(session: boto3.Session, *service_names: str):
    """
    Create multiple boto3 clients from a session.
    
    Args:
        session: Boto3 session
        *service_names: Service names (e.g., 's3', 'sagemaker', 'logs')
    
    Returns:
        tuple: Boto3 clients in the order requested
    
    Example:
        >>> session = get_boto_session(region='us-west-2')
        >>> s3, sagemaker, logs = create_clients(session, 's3', 'sagemaker', 'logs')
        >>> response = sagemaker.list_endpoints()
    """
    return tuple(session.client(service) for service in service_names)
