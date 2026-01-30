#!/usr/bin/env python3
"""
Register and approve SageMaker model packages.
"""

import sys


def create_model_package_group(sm_client, model_package_group_name, description="Sagemaker model group"):
    """
    Create a SageMaker model package group.
    
    Args:
        sm_client: Boto3 SageMaker client
        model_package_group_name: Name for the model package group
        description: Description for the model package group
    
    Returns:
        str: The model package group name
    """
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription=description
        )
        print(f"✓ Created model package group: {model_package_group_name}")
    except sm_client.exceptions.ResourceInUse:
        print(f"✓ Model package group already exists: {model_package_group_name}")
    except Exception as e:
        print(f"ERROR: Failed to create model package group: {e}", file=sys.stderr)
        sys.exit(1)
    
    return model_package_group_name


def register_model_package(sm_client, model_package_group_name, container_image, model_s3_uri, 
                          instance_type, inference_script="inference.py", environment_vars=None):
    """
    Register a model package in SageMaker.
    
    Args:
        sm_client: Boto3 SageMaker client
        model_package_group_name: Name of the model package group
        container_image: ECR container image URI
        model_s3_uri: S3 URI of the model artifacts
        instance_type: Instance type for inference
        inference_script: Name of the inference script (default: inference.py)
        environment_vars: Dict of custom environment variables for the container (optional)
    
    Returns:
        str: The model package ARN
    """
    try:
        print(f"Registering model with image: {container_image}")
        
        # Build environment variables with required SageMaker vars + custom vars
        container_env = {
            "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri
        }
        if environment_vars:
            container_env.update(environment_vars)
            print(f"Custom environment variables: {list(environment_vars.keys())}")
        
        response = sm_client.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription="Sagemaker model version",
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": container_image,
                        "ModelDataUrl": model_s3_uri,
                        "Environment": container_env
                    }
                ],
                "SupportedRealtimeInferenceInstanceTypes": [instance_type],
                "SupportedTransformInstanceTypes": [instance_type]
            },
            ModelApprovalStatus="PendingManualApproval",
            Domain="MACHINE_LEARNING",
            Task="REGRESSION"
        )
        
        model_package_arn = response["ModelPackageArn"]
        print(f"✓ Registered model package: {model_package_arn}")
        return model_package_arn
    
    except Exception as e:
        print(f"ERROR: Failed to register model package: {e}", file=sys.stderr)
        sys.exit(1)


def approve_model_package(sm_client, model_package_arn):
    """
    Approve a model package in SageMaker.
    
    Args:
        sm_client: Boto3 SageMaker client
        model_package_arn: ARN of the model package to approve
    
    Returns:
        str: The approved model package ARN
    """
    try:
        sm_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved"
        )
        print(f"✓ Approved model package: {model_package_arn}")
        return model_package_arn
    
    except Exception as e:
        print(f"ERROR: Failed to approve model package: {e}", file=sys.stderr)
        sys.exit(1)


def register_and_approve_model(sm_client, model_package_group_name, container_image, 
                               model_s3_uri, instance_type,
                               inference_script='inference.py',
                               environment_vars=None):
    """
    Complete workflow: create group, register model package, and approve it.
    
    Args:
        sm_client: Boto3 SageMaker client
        model_package_group_name: Name for the model package group
        container_image: ECR container image URI
        model_s3_uri: S3 URI of the model artifacts
        instance_type: Instance type for inference
        inference_script: Name of the inference script (default: inference.py)
        environment_vars: Dict of custom environment variables for the container (optional)
    
    Returns:
        str: The approved model package ARN
    """
    # Step 1: Create model package group
    create_model_package_group(sm_client, model_package_group_name)
    
    # Step 2: Register model package
    model_package_arn = register_model_package(
        sm_client, 
        model_package_group_name, 
        container_image, 
        model_s3_uri, 
        instance_type, 
        inference_script,
        environment_vars
    )
    
    # Step 3: Approve model package
    approve_model_package(sm_client, model_package_arn)
    
    return model_package_arn


if __name__ == "__main__":
    import boto3
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Register and approve SageMaker model package.")
    parser.add_argument("-g", "--model-package-group", required=True, help="Model package group name")
    parser.add_argument("-i", "--image", required=True, help="Container image URI")
    parser.add_argument("-m", "--model-uri", required=True, help="S3 URI of model artifacts")
    parser.add_argument("-t", "--instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("-f", "--inference-script", default="inference.py", help="Inference script name")
    parser.add_argument("-r", "--region", default="us-east-1", help="AWS region")
    args = parser.parse_args()
    
    # Create SageMaker client
    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        boto_session = boto3.Session(profile_name=aws_profile, region_name=args.region)
    else:
        boto_session = boto3.Session(region_name=args.region)
    
    sm_client = boto_session.client("sagemaker")
    
    # Register and approve model
    register_and_approve_model(
        sm_client, 
        args.model_package_group, 
        args.image, 
        args.model_uri, 
        args.instance_type
    )
