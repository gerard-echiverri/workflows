#!/usr/bin/env python3
"""
Upload model artifacts to S3 for SageMaker deployment.
"""

import os
import sys


def upload_model_to_s3(s3_client, bucket, tar_file="model.tar.gz"):
    """
    Upload model tar.gz file to S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        inference_file_name: Name of the inference file (used for S3 key structure)
        tar_file: Local tar.gz file to upload (default: model.tar.gz)
    
    Returns:
        str: S3 URI of the uploaded model (e.g., s3://bucket/models/inference.py/model.tar.gz)
    """
    try:
        # Validate tar file exists
        if not os.path.exists(tar_file):
            print(f"ERROR: Model tar file not found: {tar_file}", file=sys.stderr)
            sys.exit(1)
        
        # Construct S3 key
        s3_key = f"models/{tar_file}"
        
        # Upload to S3
        s3_client.upload_file(tar_file, bucket, s3_key)
        
        # Construct S3 URI
        model_s3_uri = f"s3://{bucket}/{s3_key}"
        
        print(f"✓ Uploaded to: {model_s3_uri}")
        return model_s3_uri
    
    except Exception as e:
        print(f"ERROR: Failed to upload model to S3: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import boto3
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload model artifacts to S3.")
    parser.add_argument("-b", "--bucket", required=True, help="S3 bucket name")
    parser.add_argument("-f", "--inference-file", default="inference.py", 
                        help="Inference file name (default: inference.py)")
    parser.add_argument("-t", "--tar-file", default="model.tar.gz", 
                        help="Model tar.gz file (default: model.tar.gz)")
    parser.add_argument("-r", "--region", default="us-east-1", 
                        help="AWS region (default: us-east-1)")
    args = parser.parse_args()
    
    # Create S3 client
    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        boto_session = boto3.Session(profile_name=aws_profile, region_name=args.region)
    else:
        boto_session = boto3.Session(region_name=args.region)
    
    s3_client = boto_session.client("s3")
    
    # Upload model
    upload_model_to_s3(s3_client, args.bucket, args.inference_file, args.tar_file)
