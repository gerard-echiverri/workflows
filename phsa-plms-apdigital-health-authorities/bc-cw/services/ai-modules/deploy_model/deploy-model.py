#!/usr/bin/env python3
"""
SageMaker Model Deployment Script - QC AI Model

Deploys a Quality Control (QC) AI model for medical image classification
to a SageMaker real-time inference endpoint with monitoring and rollback.

Use Case:
    Medical image quality control using linear algorithm (scikit-learn)
    for automated assessment of imaging quality (pass/fail/inconclusive)

Acceptance Criteria:
    ✓ Real-time endpoint deployment (not batch transform)
    ✓ Parameterized inputs via command-line arguments AND environment variables
    ✓ Linear algorithm model support (sklearn/scikit-learn)
    ✓ Manual workflow execution deploys endpoint to InService state
    ✓ Endpoint visible in SageMaker console after successful deployment
    ✓ Invalid artifacts/config fail with clear error messages
    ✓ Deployment failures do not leave endpoint in broken state (auto-rollback)

Prerequisites:
    - QC model package registered and approved (use package-model.py)
    - Model in scikit-learn format (model.joblib) with linear algorithm
    - Container image pushed to ECR with inference handler
    - Required environment variables set
    - AWS credentials configured (via AWS_PROFILE or default credentials)

Configuration Sources (Priority Order):
    1. Command-line arguments (highest priority)
    2. Environment variables
    3. Default values (lowest priority)

Required Environment Variables:
    AWS_REGION                      : AWS region for deployment
    SAGEMAKER_EXECUTION_ROLE_ARN    : IAM role ARN for SageMaker execution

Optional Environment Variables:
    AWS_PROFILE                     : AWS CLI profile name for authentication

Deployment Parameters (via CLI args or env vars):
    --model-package-arn, -a         : Model package ARN (required, CLI only)
    --instance-type, -t             : Instance type (default: ml.c5.2xlarge)
    --endpoint-name, -e             : Custom endpoint name (default: auto-generated)
    --no-monitor                    : Disable deployment monitoring
    --skip-rollback                 : Disable auto-rollback on failure

Features:
    - Real-time inference endpoint for synchronous predictions
    - Non-blocking deployment with real-time monitoring
    - Automatic CloudWatch Logs streaming during deployment
    - Configurable instance types optimized for linear models
    - Automatic rollback on deployment failure (enabled by default)
    - Deployment timeout protection (30 minutes)
    - QC-specific validation and error handling

Workflow:
    1. Validate environment variables and arguments
    2. Initialize AWS clients (SageMaker, CloudWatch Logs)
    3. Deploy model package to endpoint (non-blocking)
    4. Monitor deployment status and stream CloudWatch logs
    5. Handle success/failure with optional rollback

Exit Codes:
    0 : Successful deployment
    1 : Validation error, deployment failure, or rollback triggered

Usage Examples:
    # Deploy QC AI model with monitoring and auto-rollback (recommended)
    $ export AWS_REGION=us-west-2
    $ export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::<account-id>:role/SageMakerRole
    $ python deploy-model.py \
        --model-package-arn arn:aws:sagemaker:us-west-2:<account-id>:model-package/qc-model/1
    
    # Deploy to custom endpoint without monitoring
    $ python deploy-model.py \\
        --model-package-arn arn:aws:sagemaker:... \\
        --endpoint-name qc-production-endpoint \\
        --no-monitor
    
    # Deploy with different instance type for testing
    $ python deploy-model.py \\
        --model-package-arn arn:aws:sagemaker:... \\
        --instance-type ml.t2.medium
    
    # Deploy without auto-rollback (keep failed endpoints for debugging)
    $ python deploy-model.py \\
        --model-package-arn arn:aws:sagemaker:... \\
        --skip-rollback
    
    # Full example with all parameters
    $ export AWS_REGION=ca-central-1
    $ export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::<account-id>:role/SageMakerRole
    $ export AWS_PROFILE=production
    $ python deploy-model.py \
        --model-package-arn arn:aws:sagemaker:ca-central-1:<account-id>:model-package/QCModel/2 \
        --endpoint-name qc-ai-production \\
        --instance-type ml.c5.2xlarge

Author: Gerard Rey Echiverri
Last Modified: January 2026
"""

import os
import sys
import argparse
import time

from utils.list_model_packages import list_model_packages
from time import strftime
from typing import Optional
from sagemaker import ModelPackage, Session
from modules.monitor_endpoint import monitor_and_tail, delete_endpoint
from utils.boto_session import get_boto_session
from utils.env_validation import validate_env_vars
from utils.output import print_header, print_kv_pairs, print_success, print_error, print_warning, print_section


# ========================================
# VALIDATION FUNCTIONS
# ========================================

def validate_model_package_arn(arn: str) -> tuple[bool, str]:
    """
    Validate model package ARN format and extract components.
    
    Args:
        arn: Model package ARN to validate
    
    Returns:
        tuple: (is_valid, error_message)
        
    Example:
        >>> valid, error = validate_model_package_arn(
        ...     "arn:aws:sagemaker:ca-central-1:<account-id>:model-package/MyModel/1"
        ... )
        >>> print(valid)  # True
    """
    if not arn or not isinstance(arn, str):
        return False, "Model package ARN cannot be empty"
    
    if not arn.startswith("arn:aws:sagemaker:"):
        return False, f"Invalid ARN format. Must start with 'arn:aws:sagemaker:', got: {arn[:30]}..."
    
    if ":model-package/" not in arn:
        return False, f"Invalid ARN. Must contain ':model-package/', got: {arn}"
    
    # Expected format: arn:aws:sagemaker:region:account:model-package/group/version
    parts = arn.split(":")
    if len(parts) < 6:
        return False, f"Invalid ARN format. Expected 6+ parts separated by ':', got {len(parts)} parts"
    
    return True, ""


def validate_instance_type(instance_type: str) -> tuple[bool, str]:
    """
    Validate SageMaker instance type format and recommend suitable types.
    
    Args:
        instance_type: Instance type to validate
    
    Returns:
        tuple: (is_valid, warning_message)
        
    Note:
        Returns True with warning for uncommon types, False only for invalid format
    """
    if not instance_type or not isinstance(instance_type, str):
        return False, "Instance type cannot be empty"
    
    if not instance_type.startswith("ml."):
        return False, f"Invalid instance type format. Must start with 'ml.', got: {instance_type}"
    
    # Recommended types for linear models (CPU-optimized)
    recommended = ['ml.c5.xlarge', 'ml.c5.2xlarge', 'ml.c5.4xlarge', 'ml.t2.medium', 'ml.m5.xlarge']
    
    if instance_type not in recommended:
        warning = f"Using non-standard instance type '{instance_type}'. Recommended for QC linear models: {', '.join(recommended)}"
        return True, warning
    
    return True, ""


# ========================================
# HELPER FUNCTIONS
# ========================================

def generate_endpoint_name(custom_name: Optional[str] = None) -> str:
    """
    Generate or validate endpoint name.
    
    Args:
        custom_name: Optional custom endpoint name
    
    Returns:
        str: Endpoint name to use
    
    Example:
        >>> name = generate_endpoint_name()
        >>> print(name)  # qc-ai-detection-ep-2026-01-29-15-30-45
        >>> name = generate_endpoint_name("my-endpoint")
        >>> print(name)  # my-endpoint
    """
    if custom_name:
        return custom_name
    else:
        return f"qc-ai-detection-ep-{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}"


def handle_deployment_result(status: str, endpoint_name: str, 
                            rollback_on_failure: bool,
                            sagemaker_client) -> int:
    """
    Handle the deployment result and perform rollback if needed.
    
    Args:
        status: Final deployment status (InService, Failed, Timeout)
        endpoint_name: Name of the deployed endpoint
        rollback_on_failure: Whether to rollback on failure
        sagemaker_client: Boto3 SageMaker client
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    
    Example:
        >>> exit_code = handle_deployment_result(
        ...     'InService', 'my-endpoint', True, sagemaker_client
        ... )
        >>> sys.exit(exit_code)
    """
    print(f"\n{'='*60}")
    print(f"Deployment Status: {status}")
    print(f"Endpoint: {endpoint_name}")
    print(f"{'='*60}\n")
    
    if status == 'InService':
        print_success("DEPLOYMENT SUCCESSFUL")
        print(f"\n{'='*60}")
        print_success("Endpoint Status: InService")
        print_success("Ready for real-time inference")
        print_success("Visible in SageMaker console")
        print(f"{'='*60}")
        
        print(f"\nEndpoint Details:")
        print(f"  Name: {endpoint_name}")
        print(f"  Type: Real-time inference endpoint")
        print(f"  Algorithm: Linear (scikit-learn)")
        
        print(f"\nView in AWS Console:")
        region = sagemaker_client.meta.region_name
        console_url = f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{endpoint_name}"
        print(f"  {console_url}")
        
        print(f"\nTest the endpoint:")
        print(f"  python test_endpoint.py --endpoint-name {endpoint_name}")
        
        print(f"\nDescribe endpoint:")
        print(f"  aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")
        
        return 0
    
    elif status in ['Failed', 'Timeout']:
        print_error("DEPLOYMENT FAILED")
        
        # Provide detailed error context
        print(f"\n{'='*60}")
        print(f"Status: {status}")
        print(f"Endpoint: {endpoint_name}")
        print(f"{'='*60}")
        
        print(f"\nCommon Causes:")
        print(f"  1. Invalid model artifacts (missing model.joblib or inference.py)")
        print(f"  2. Incompatible container image or missing dependencies")
        print(f"  3. Insufficient IAM permissions on execution role")
        print(f"  4. Instance type unavailable in region")
        print(f"  5. Model package not approved or accessible")
        
        print(f"\nTroubleshooting Steps:")
        print(f"  1. Check CloudWatch Logs for detailed error messages:")
        region = sagemaker_client.meta.region_name
        print(f"     https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FEndpoints$252F{endpoint_name}")
        print(f"\n  2. Verify model artifacts contain linear algorithm (sklearn):")
        print(f"     - model.joblib (trained scikit-learn model)")
        print(f"     - inference.py (with model_fn, input_fn, predict_fn)")
        print(f"     - requirements.txt (with scikit-learn version)")
        print(f"\n  3. Validate IAM role permissions:")
        print(f"     aws iam get-role --role-name <role-name>")
        print(f"\n  4. Check model package approval status:")
        print(f"     aws sagemaker describe-model-package --model-package-name <arn>")
        
        if rollback_on_failure:
            print(f"\n{'─'*60}")
            print_warning("Initiating automatic rollback...")
            print(f"{'─'*60}")
            if delete_endpoint(sagemaker_client, endpoint_name, delete_config=True):
                print_success("Rollback completed")
                print("  ")
                print_success("Endpoint deleted")
                print_success("Configuration deleted")
                print_success("No resources left in broken state")
            else:
                print_error("Rollback failed")
                print(f"\nManual cleanup required:")
                print(f"  aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")
                print(f"  aws sagemaker delete-endpoint-config --endpoint-config-name {endpoint_name}")
        else:
            print(f"\n{'─'*60}")
            print_warning("Auto-rollback disabled")
            print(f"{'─'*60}")
            print(f"\nManual cleanup required:")
            print(f"  aws sagemaker delete-endpoint --endpoint-name {endpoint_name}")
            print(f"  aws sagemaker delete-endpoint-config --endpoint-config-name {endpoint_name}")
            print(f"\nTip: Enable auto-rollback by removing --skip-rollback flag")
        
        return 1
    
    return 1


# ========================================
# COMMAND LINE INTERFACE
# ========================================

date_tag = strftime("%Y%m%d%H%M")

parser = argparse.ArgumentParser(
    description="Deploy QC AI model package to a real-time SageMaker inference endpoint",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Quick Start:
  # Basic QC AI model deployment with monitoring (recommended)
  python deploy-model.py --model-package-arn arn:aws:sagemaker:region:account:model-package/group/version

Common Options:
  # Custom endpoint name for QC model
  python deploy-model.py -a arn:aws:sagemaker:... -e qc-production-endpoint
  
  # Different instance type (testing vs production)
  python deploy-model.py -a arn:aws:sagemaker:... -t ml.t2.medium  # testing
  python deploy-model.py -a arn:aws:sagemaker:... -t ml.c5.2xlarge  # production
  
  # Deploy without monitoring (faster, but no log visibility)
  python deploy-model.py -a arn:aws:sagemaker:... --no-monitor
  
  # Disable auto-rollback (keep failed endpoints for debugging)
  python deploy-model.py -a arn:aws:sagemaker:... --skip-rollback

Full Example (QC AI Production Deployment):
  python deploy-model.py \\
    --model-package-arn arn:aws:sagemaker:ca-central-1:<account-id>:model-package/PLMSCWQCModel/1 \
    --endpoint-name plmscw-qc-production \\
    --instance-type ml.c5.2xlarge

Environment Setup:
  export AWS_REGION=ca-central-1
  export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::<account-id>:role/SageMakerExecutionRole
  export AWS_PROFILE=production  # Optional
    """
)

parser.add_argument(
    "--model-package-group", "-g", 
    required=True,
    metavar="NAME",
    help="Name of the approved QC AI model package"
)
parser.add_argument(
    "-t", "--instance-type", 
    default="ml.c5.2xlarge",
    metavar="TYPE",
    help="SageMaker instance type for QC model inference (default: %(default)s). Recommended: ml.c5.2xlarge (production), ml.c5.xlarge, ml.t2.medium (testing)"
)
parser.add_argument(
    "--endpoint-name", "-e",
    metavar="NAME",
    help="Custom endpoint name for QC AI model (default: auto-generated with timestamp)"
)
parser.add_argument(
    "--no-monitor", 
    action="store_true",
    help="Skip deployment monitoring and log streaming (not recommended for production)"
)
parser.add_argument(
    "--skip-rollback", 
    action="store_false",
    dest="rollback_on_failure",
    help="Disable automatic cleanup on deployment failure (rollback is enabled by default)"
)

args = parser.parse_args()


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    # Step 1: Validate environment variables
    required_env = [
        ("AWS_REGION", "AWS region for SageMaker deployment"),
        ("SAGEMAKER_EXECUTION_ROLE_ARN", "IAM role ARN for SageMaker execution")
    ]
    env_vars = validate_env_vars(required_env, "deploy-model.py")
    AWS_REGION = env_vars["AWS_REGION"]  # From environment
    SAGEMAKER_EXECUTION_ROLE_ARN = env_vars["SAGEMAKER_EXECUTION_ROLE_ARN"]  # From environment
    
    # Step 2: Extract deployment parameters from command-line arguments
    instance_type = args.instance_type  # From CLI arg (default: ml.c5.2xlarge)
    model_package_group = args.model_package_group  # From CLI arg (required)
    custom_endpoint_name = args.endpoint_name  # From CLI arg (optional)
    monitor_enabled = not args.no_monitor  # From CLI arg (default: enabled)
    rollback_enabled = args.rollback_on_failure  # From CLI arg (default: enabled)
    
    # Step 2a: Validate deployment parameters before AWS calls
    print_header("VALIDATING DEPLOYMENT PARAMETERS")

    # Validate instance type
    type_valid, type_warning = validate_instance_type(instance_type)
    if not type_valid:
        print_error(f"Invalid instance type: {type_warning}")
        print(f"\nValid format: ml.<family>.<size>")
        print(f"\nRecommended for QC linear models:")
        print(f"  - ml.c5.2xlarge (production)")
        print(f"  - ml.c5.xlarge (development)")
        print(f"  - ml.t2.medium (testing)")
        sys.exit(1)
    elif type_warning:
        print_warning(type_warning)
    else:
        print_success(f"Instance type validated: {instance_type}")
    
    print_success(f"All deployment parameters valid\n")
    
    # Step 3: Display QC AI model deployment configuration
    print_header("QC AI MODEL DEPLOYMENT")
    config = {
        "Model Type": "Quality Control (Linear Algorithm)",
        "Model Package Group Name": model_package_group,
        "Instance Type": instance_type,
        "AWS Region": AWS_REGION,
        "Endpoint Type": "Real-time Inference",
        "Monitoring": "Enabled" if monitor_enabled else "Disabled",
        "Auto-Rollback": "Yes" if rollback_enabled else "No"
    }
    print_kv_pairs(config, key_width=20)
    print()
    
    # Step 4: Initialize AWS clients
    boto_session = get_boto_session(region=AWS_REGION)
    sagemaker_client = boto_session.client("sagemaker")
    logs_client = boto_session.client("logs")
    sagemaker_session = Session(boto_session=boto_session)

    # Step 5: Generate endpoint name (from arg or auto-generate)
    endpoint_name = generate_endpoint_name(custom_endpoint_name)
    print(f"Target Endpoint: {endpoint_name}")
    if custom_endpoint_name:
        print(f"  (Custom name from --endpoint-name argument)")
    else:
        print(f"  (Auto-generated with timestamp)")
    print()
    
    # Step 6: Get approved model package from group
    print("Preparing model package for deployment...")

    packages = list_model_packages(sagemaker_client, model_package_group)
    if not packages:
        print_error(f"No approved model packages found in group: {model_package_group}")
        print(f"\nTroubleshooting:")
        print(f"  1. Verify the model package group name is correct")
        print(f"  2. Check if packages exist:")
        print(f"     python utils/list_model_packages.py --filter {model_package_group}")
        print(f"  3. Ensure at least one package is approved")
        print(f"  4. Run package-model.py to create and approve a package:")
        print(f"     python utils/package-model.py -g {model_package_group}")
        sys.exit(1)
    
    model_package_arn = packages[0]['ModelPackageArn']
    print_success(f"Found approved package: {model_package_arn}")
    print()

    model = ModelPackage(
        model_package_arn=model_package_arn,
        role=SAGEMAKER_EXECUTION_ROLE_ARN,
        sagemaker_session=sagemaker_session
    )
    
    # Step 7: Deploy to endpoint (non-blocking)
    deploy_config = {
        "Target Endpoint": endpoint_name,
        "Instance Type": instance_type,
        "Instance Count": "1",
        "Deployment Mode": "Non-blocking"
    }
    print_kv_pairs(deploy_config, key_width=18)
    print()
    
    predictor = model.deploy(
        instance_type=instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        wait=False  # Non-blocking: allows real-time monitoring
    )
    
    print_success("Deployment request submitted")
    print(f"  Endpoint: {endpoint_name}")
    print(f"  Status: Creating...\n")

    # Step 8: Monitor deployment (optional)
    if not args.no_monitor:
        log_group_name = f"/aws/sagemaker/Endpoints/{endpoint_name}"
        
        monitor_config = {
            "Endpoint": endpoint_name,
            "Log Group": log_group_name,
            "Max Wait": "30 minutes"
        }
        print_section("MONITORING DEPLOYMENT", monitor_config, char="─")
        
        # Monitor with real-time log streaming
        status = monitor_and_tail(
            sagemaker_client, 
            logs_client, 
            endpoint_name, 
            log_group_name,
            max_wait=1800  # 30 minutes timeout
        )
        
        # Handle deployment result
        exit_code = handle_deployment_result(
            status, 
            endpoint_name, 
            args.rollback_on_failure,
            sagemaker_client
        )
        sys.exit(exit_code)
    
    else:
        # Monitoring disabled - deployment continues in background
        print(f"{'─'*60}")
        print_warning("MONITORING DISABLED")
        print(f"{'─'*60}")
        print(f"Endpoint deployment initiated: {endpoint_name}")
        print(f"\nDeployment is running in the background.")
        print(f"\nCheck status with:")
        print(f"  aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")
        print(f"\nOr re-run with monitoring:")
        print(f"  python deploy-model.py -a {model_package_arn} -e {endpoint_name}")
        print(f"{'─'*60}\n")
        sys.exit(0)