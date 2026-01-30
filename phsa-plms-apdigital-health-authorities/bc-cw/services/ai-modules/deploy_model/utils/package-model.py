#!/usr/bin/env python3
"""
Package a model for SageMaker deployment.

This script builds, uploads, registers, and approves a model package in SageMaker,
making it ready for deployment to a real-time inference endpoint.

Acceptance Criteria:
    ✓ Supports linear algorithm models (sklearn/scikit-learn)
    ✓ Parameterized inputs via command-line arguments
    ✓ Prepares models for real-time endpoint deployment

Model Requirements:
    - Must be a linear algorithm in scikit-learn format (model.joblib)
    - Must include inference.py with SageMaker handler functions
    - Must include requirements.txt with runtime dependencies

Workflow:
    1. Bundle model artifacts into tar.gz
    2. Upload to S3
    3. Create/reuse model package group
    4. Register model package with container environment variables
    5. Auto-approve model package for deployment

The script outputs the model package ARN which can be used with deploy-model.py.
"""

import os
import sys
import argparse
from time import strftime
from modules.bundle_artifacts import bundle_model_artifacts
from modules.upload_artifacts import upload_model_to_s3
from modules.bundle_package import register_and_approve_model
from utils.boto_session import get_boto_session
from utils.env_validation import validate_env_vars
from utils.output import print_header, print_kv_pairs, print_success, print_info

# ================================
# CONFIG
# ================================
date_tag = strftime("%Y%m%d%H%M")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Package a model for SageMaker deployment",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Package model with defaults
  python package-model.py
  
  # Package with custom stack and group name
  python package-model.py --stack plmscw --model-package-group-name MyModelGroup
  
  # Specify container repo details
  python package-model.py --repo-name prod --import-branch-name inference-v2
  
  # With specific instance type for validation
  python package-model.py --instance-type ml.c5.xlarge
    """
)

parser.add_argument("-s", "--stack", default="plmscw", 
                    help="Main stack name (default: plmscw)")
parser.add_argument("-t", "--instance-type", default="ml.c5.2xlarge", 
                    help="SageMaker instance type for inference (default: ml.c5.2xlarge)")
parser.add_argument("-g", "--model-package-group-name", default="PLMSCWInferenceModelGroup", 
                    help="Model package group name (default: PLMSCWInferenceModelGroup)")
parser.add_argument("--repo-name", default="dev", 
                    help="Repository/environment name (default: dev)")
parser.add_argument("--import-branch-name", default="inference-data", 
                    help="Import branch name (default: inference-data)")
parser.add_argument("--output-file", "-o",
                    help="Save model package ARN to file")
args = parser.parse_args()

# --- ENVIRONMENT VARIABLE VALIDATION ---
required_env = [
    ("AWS_REGION", "AWS region for SageMaker deployment"),
    ("AWS_ACCOUNT", "AWS account ID"),
    ("DATA_BUCKET", "S3 bucket for model artifacts"),
    ("AI_INPUT_QUEUE_URL", "SQS queue URL for AI processing"),
    ("SAGEMAKER_EXECUTION_ROLE_ARN", "IAM role ARN for SageMaker execution")
]

env = validate_env_vars(required_env, "package-model.py")
AWS_REGION = env["AWS_REGION"]
AWS_ACCOUNT = env["AWS_ACCOUNT"]
DATA_BUCKET = env["DATA_BUCKET"]
AI_INPUT_QUEUE_URL = env["AI_INPUT_QUEUE_URL"]
SAGEMAKER_EXECUTION_ROLE_ARN = env["SAGEMAKER_EXECUTION_ROLE_ARN"]

# From parameters
main_stack_name = args.stack
api_repo_name = args.repo_name
instance_type = args.instance_type
import_branch_name = args.import_branch_name
container_image = f"{AWS_ACCOUNT}.dkr.ecr.{AWS_REGION}.amazonaws.com/{main_stack_name}-{import_branch_name}-{api_repo_name}"

print_header("MODEL PACKAGING")
package_config = {
    "Stack": main_stack_name,
    "Branch": import_branch_name,
    "Repo": api_repo_name,
    "Container": container_image,
    "Instance Type": instance_type
}
print_kv_pairs(package_config, key_width=15)
print()

# ================================
# CLIENTS
# ================================
boto_session = get_boto_session(region=AWS_REGION)
s3 = boto_session.client("s3")
sagemaker_client = boto_session.client("sagemaker")

# ================================
# STEP 1 — BUILD MODEL TAR
# ================================
print_info("STEP 1: Building model artifacts...")
bundle_model_artifacts()

# ================================
# STEP 2 — UPLOAD TO S3
# ================================
print_info("STEP 2: Uploading to S3...")
model_s3_uri = upload_model_to_s3(s3, DATA_BUCKET)

# ================================
# STEP 3-5 — REGISTER & APPROVE MODEL
# ================================
print_info("STEP 3-5: Registering and approving model package...")
model_package_group_name = f"{args.model_package_group_name}-{date_tag}"

# Prepare environment variables for the container
container_env_vars = {
    "DATA_BUCKET": DATA_BUCKET,
    "AI_INPUT_QUEUE_URL": AI_INPUT_QUEUE_URL,
    "AWS_REGION": AWS_REGION
}

model_package_arn = register_and_approve_model(
    sagemaker_client, 
    model_package_group_name, 
    container_image, 
    model_s3_uri, 
    instance_type, 
    container_env_vars
)

# ================================
# OUTPUT
# ================================
print_header("MODEL PACKAGE CREATED")
package_info = {
    "ARN": model_package_arn,
    "Group": model_package_group_name,
    "S3 URI": model_s3_uri
}
print_kv_pairs(package_info, key_width=10)
print()

# Save to file if requested
if args.output_file:
    with open(args.output_file, 'w') as f:
        f.write(model_package_arn)
    print_success(f"Model package ARN saved to: {args.output_file}")

print_success("Model packaging completed")
print(f"\nTo deploy this model package, use:")
print(f"  python deploy-model.py --model-package-arn {model_package_arn}")
