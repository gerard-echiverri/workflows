# SageMaker Model Deployment

Deployment pipeline for packaging and deploying ML models to AWS SageMaker endpoints with comprehensive utilities and monitoring.

## Acceptance Criteria

This deployment system meets the following requirements:

✅ **Real-time Endpoint Deployment** - Deploys models to SageMaker real-time inference endpoints (not batch transform jobs)

✅ **Linear Algorithm Support** - Designed for linear algorithm models in scikit-learn format (model.joblib), with inference based on linear algorithms

✅ **Parameterized Inputs** - All scripts accept command-line arguments AND environment variables for flexible configuration without code changes

✅ **Successful Manual Execution** - Running the workflow manually with valid inputs successfully executes deploy-model.py and deploys (or updates) a SageMaker endpoint that reaches "InService" state and is visible in the SageMaker console

✅ **Robust Error Handling** - Invalid model artifacts or configuration fail with clear, actionable error messages and do not leave the endpoint in a broken state (automatic rollback enabled by default)

✅ **Pre-deployment Validation** - Validates model package ARN format and instance type before making AWS API calls to catch errors early

## Overview

The deployment pipeline is organized into modular scripts with shared utility libraries:

### Main Scripts

1. **`utils/package-model.py`** - Packages a model for SageMaker (Steps 1-5)
2. **`deploy-model.py`** - Deploys a packaged model to an endpoint (Steps 6-7)
3. **`test_endpoint.py`** - Tests deployed endpoints with validation
4. **`utils/list_model_packages.py`** - Lists and manages model packages

### Utility Modules

The codebase uses shared utility modules for consistent behavior across all scripts:

- **`utils/boto_session.py`** - AWS session management with automatic credential detection
- **`utils/env_validation.py`** - Environment variable validation with consistent error handling
- **`utils/endpoint_operations.py`** - SageMaker endpoint operations (status, info, waiting)
- **`utils/output.py`** - Consistent output formatting (headers, tables, JSON, status messages)

This architecture provides:
- Package once, deploy multiple times
- Deploy different versions to different endpoints
- Consistent error handling and output formatting
- Reusable components across scripts
- Simplified CI/CD integration

## Scripts

### utils/package-model.py

Handles the model packaging process for linear algorithm models:
- Bundles model artifacts (model.joblib, inference.py) into tar.gz
- Uploads to S3
- Creates/reuses model package group
- Registers model package with environment variables
- Auto-approves the package
- Outputs the model package ARN (can save to file with `--output-file`)

**Model Requirements:**
- Linear algorithm in scikit-learn format (model.joblib)
- inference.py with SageMaker handler functions (model_fn, input_fn, predict_fn, output_fn)
- requirements.txt with runtime dependencies

**Required Environment Variables:**
- `AWS_REGION` - AWS region for deployment
- `AWS_ACCOUNT` - AWS account ID
- `DATA_BUCKET` - S3 bucket for model artifacts
- `AI_INPUT_QUEUE_URL` - SQS queue URL for AI processing
- `SAGEMAKER_EXECUTION_ROLE_ARN` - IAM role ARN for SageMaker

**Command Line Arguments:**
- `-s, --stack` - Main stack name (default: plmscw)
- `-t, --instance-type` - Instance type for validation (default: ml.c5.2xlarge)
- `-g, --model-package-group-name` - Package group name (default: PLMSCWInferenceModelGroup)
- `--repo-name` - Repository/environment name (default: dev)
- `--import-branch-name` - Branch name (default: inference-data)
- `-o, --output-file` - Save model package ARN to file

### deploy-model.py

Handles the real-time endpoint deployment with comprehensive validation and error handling:
- Takes a model package ARN as required input
- Deploys to a SageMaker **real-time inference endpoint** (auto-generated name with timestamp)
- **Validates parameters before deployment** (ARN format, instance type)
- **Monitors deployment with CloudWatch logs** (enabled by default)
- **Auto-rollback on failure** (enabled by default, use `--skip-rollback` to disable)
- **Provides clear console visibility** with AWS Console URLs on success
- **Actionable error messages** with troubleshooting steps on failure
- Supports custom endpoint naming and instance types

**Endpoint Type:**
- Creates real-time inference endpoints for synchronous predictions using linear algorithms
- Not for batch transform jobs or asynchronous inference

**Validation Features:**
- Pre-deployment ARN format validation
- Instance type validation with recommendations for linear models
- Early error detection before AWS API calls
- No broken endpoints left behind (automatic cleanup on failure)

**Success Criteria:**
- Endpoint reaches "InService" state
- Visible in SageMaker console with direct link provided
- Ready for real-time inference requests

**Required Environment Variables:**
- `AWS_REGION` - AWS region for deployment
- `SAGEMAKER_EXECUTION_ROLE_ARN` - IAM role ARN for SageMaker

**Command Line Arguments:**
- `-a, --model-package-arn` - ARN of approved model package (required)
- `-t, --instance-type` - Instance type (default: ml.c5.2xlarge, recommended for linear models)
- `-e, --endpoint-name` - Custom endpoint name (default: auto-generated with timestamp)
- `--no-monitor` - Skip deployment monitoring (not recommended for production)
- `--skip-rollback` - Disable automatic rollback on failure (rollback enabled by default)

### test_endpoint.py

Tests deployed SageMaker endpoints:
- Sends inference requests with JSON data
- Validates response format and content
- Supports file or string input
- Displays formatted results

**Command Line Arguments:**
- `-e, --endpoint-name` - Name of endpoint to test (required)
- `-d, --input-data` - Path to JSON file or JSON string (required)
- `--region` - AWS region (default: ca-central-1 or AWS_REGION env var)
- `-v, --verbose` - Print detailed response

**Note:** Uses `AWS_PROFILE` environment variable for authentication if set.

### utils/list_model_packages.py

Lists and filters model packages:
- Shows model package groups
- Lists packages by approval status
- JSON or formatted output

**Command Line Arguments:**
- `--filter` - Filter groups by name substring
- `--all-statuses` - Show all packages regardless of status
- `--json` - Output as JSON

## Workflow

### Basic Workflow

```bash
# Step 1: Package the model
python utils/package-model.py --output-file model_arn.txt

# Step 2: Deploy the packaged model (with monitoring and auto-rollback)
python deploy-model.py --model-package-arn $(cat model_arn.txt)

# Step 3: Test the endpoint
python test_endpoint.py \
  --endpoint-name <endpoint-name> \
  --input-data sample-data.json
```

### Advanced Examples

**Verify successful deployment:**
```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name <endpoint-name>
# Should show: EndpointStatus: "InService"

# View in AWS Console (URL provided by deploy-model.py on success)
# Navigate to: SageMaker → Endpoints → <endpoint-name>

# Test with sample data
python test_endpoint.py \
  --endpoint-name <endpoint-name> \
  --input-data sample-data.json
```

**Package with custom settings:**
```bash
python utils/package-model.py \
  --stack plmscw \
  --model-package-group-name MyModelGroup \
  --repo-name prod \
  --import-branch-name inference-v2 \
  --instance-type ml.c5.xlarge \
  --output-file model_arn.txt
```

**Deploy with custom endpoint name:**
```bash
python deploy-model.py \
  --model-package-arn arn:aws:sagemaker:... \
  --endpoint-name production-endpoint \
  --instance-type ml.m5.large
```

**Deploy without monitoring (faster, but no log visibility):**
```bash
python deploy-model.py \
  --model-package-arn arn:aws:sagemaker:... \
  --no-monitor
```

**Deploy without auto-rollback (keep failed endpoints for debugging):**
```bash
python deploy-model.py \
  --model-package-arn arn:aws:sagemaker:... \
  --skip-rollback
```

**Test endpoint with JSON string:**
```bash
python test_endpoint.py \
  --endpoint-name my-endpoint \
  --input-data '{"data": [[1.0, 2.0, 3.0]]}'
```

## Utilities and Modules

### Core Utilities (utils/)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **boto_session.py** | AWS session management | `get_boto_session()` - Auto-detect credentials (profile/keys/token) |
| | | `create_clients()` - Create multiple boto3 clients |
| **env_validation.py** | Environment validation | `validate_env_vars()` - Validate required env vars |
| | | `get_env_var()` - Get single env var with default |
| | | `print_env_summary()` - Display config with masking |
| **endpoint_operations.py** | Endpoint utilities | `check_endpoint_status()` - Get endpoint status |
| | | `get_endpoint_info()` - Detailed endpoint information |
| | | `wait_for_endpoint()` - Poll until status reached |
| | | `list_endpoints()` - List with filtering |
| | | `endpoint_exists()` - Boolean check |
| **output.py** | Output formatting | `print_header/section/table()` - Formatted output |
| | | `print_success/error/warning/info()` - Status messages |
| | | `print_json_result()` - JSON with header |
| | | `format_duration/bytes()` - Human-readable formatting |

### Main Scripts

| Script | Purpose | Key Functions |
|--------|---------|---------------|
| **package-model.py** | Model packaging orchestration | Bundles → Uploads → Registers → Approves |
| **deploy-model.py** | Endpoint deployment orchestration | Validates → Deploys → Monitors → Rollback |
| **test_endpoint.py** | Endpoint testing | Invokes → Validates → Reports |
| **list_model_packages.py** | Package management | Lists groups and packages with filtering |

### Processing Modules (modules/)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **bundle_artifacts.py** | Artifact bundling | `bundle_model_artifacts()` - Creates tar.gz from model files |
| **upload_artifacts.py** | S3 upload | `upload_model_to_s3()` - Uploads to S3, returns URI |
| **bundle_package.py** | Model registration | `create_model_package_group()` - Creates/reuses group |
| | | `register_model_package()` - Registers with env vars |
| | | `approve_model_package()` - Auto-approves |
| | | `register_and_approve_model()` - Complete workflow |
| **monitor_endpoint.py** | Deployment monitoring | `monitor_and_tail()` - Status + logs in parallel |
| | | `delete_endpoint()` - Cleanup for rollback |
| | | `wait_for_log_group()` - Waits for CloudWatch logs |
| | | `tail_logs()` - Continuous log streaming |

## Testing

After deployment, test the endpoint with sample data:

```bash
# Test with JSON file
python test_endpoint.py \
  --endpoint-name your-endpoint-name \
  --input-data sample-data.json

# Test with JSON string
python test_endpoint.py \
  --endpoint-name your-endpoint-name \
  --input-data '{"data": [[1.0, 2.0, 3.0]]}'

# Test with verbose output
python test_endpoint.py \
  --endpoint-name your-endpoint-name \
  --input-data sample-data.json \
  --verbose

# Specify region
python test_endpoint.py \
  --endpoint-name your-endpoint-name \
  --input-data sample-data.json \
  --region us-west-2
```

**Input Format:**
```json
{
  "data": [
    [feature1, feature2, feature3, ...],
    [feature1, feature2, feature3, ...]
  ]
}
```

**Expected Output:**
```json
{
  "predictions": ["qc_pass", "qc_fail"],
  "probabilities": [[0.95, 0.05], [0.30, 0.70]],
  "class_indices": [0, 1],
  "confidence_scores": [0.95, 0.70]
}
```

## Listing Model Packages

View all model package groups and their approved packages:

```bash
# List all approved packages
python utils/list_model_packages.py

# Filter by group name
python utils/list_model_packages.py --filter "PLMSCWInference"

# Show all packages regardless of status
python utils/list_model_packages.py --all-statuses

# Output as JSON
python utils/list_model_packages.py --json

# Combine filters
python utils/list_model_packages.py --filter "PLMS" --all-statuses --json
```

## Directory Structure

```
deploy_model/
├── deploy-model.py              # Main: Endpoint deployment script
├── test_endpoint.py             # Main: Endpoint testing script
│
├── modules/                     # Processing modules
│   ├── bundle_artifacts.py      # Model artifact bundling
│   ├── upload_artifacts.py      # S3 upload functionality
│   ├── bundle_package.py        # Model package registration
│   └── monitor_endpoint.py      # Deployment monitoring
│
├── utils/                       # Utility modules and scripts
│   ├── boto_session.py          # AWS session management
│   ├── env_validation.py        # Environment variable validation
│   ├── endpoint_operations.py   # Endpoint utilities
│   ├── output.py                # Output formatting
│   ├── package-model.py         # Main: Model packaging script
│   └── list_model_packages.py   # Main: Package listing utility
│
├── artifacts/                   # Container image artifacts
│   ├── Dockerfile               # SageMaker container definition
│   ├── inference.py             # Inference handler code
│   ├── model.joblib             # Model artifact
│   ├── requirements.txt         # Python dependencies
│   ├── ecr-image-setup.sh       # ECR image build script
│   └── README.md                # Container documentation
│
└── README.md                    # This file
```

## Environment Setup

### Required Environment Variables

Set these before running any scripts:

```bash
# AWS Configuration
export AWS_REGION=ca-central-1
export AWS_ACCOUNT=358682933887

# SageMaker
export SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::358682933887:role/SageMakerExecutionRole

# S3 and SQS (for packaging only)
export DATA_BUCKET=plmscw-sagemaker-data-dev
export AI_INPUT_QUEUE_URL=https://sqs.ca-central-1.amazonaws.com/358682933887/ai-queue

# Authentication (optional - will use default credentials if not set)
export AWS_PROFILE=Administrator-358682933887
```

### Script-Specific Requirements

**For `utils/package-model.py`:**
- All environment variables above are required

**For `deploy-model.py`:**
- `AWS_REGION` (required)
- `SAGEMAKER_EXECUTION_ROLE_ARN` (required)
- `AWS_PROFILE` (optional)

**For `test_endpoint.py`:**
- `AWS_REGION` (optional, default: ca-central-1)
- `AWS_PROFILE` (optional)

### Authentication Methods

The utilities support multiple authentication methods (in priority order):

1. **AWS Profile** - `AWS_PROFILE` environment variable
2. **Session Token** - `AWS_SESSION_TOKEN` environment variable
3. **Access Keys** - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
4. **Default Credentials** - IAM role, instance profile, or default profile

## Monitoring and Rollback

### Default Behavior

Deployment monitoring and automatic rollback are **enabled by default**:

- **Monitoring**: Real-time CloudWatch log streaming and endpoint status tracking
- **Auto-Rollback**: Failed deployments are automatically cleaned up

### Monitoring Features

- Real-time CloudWatch log tailing (starts when log group becomes available)
- Parallel endpoint status monitoring (polls every 30 seconds)
- Automatic log retention (1 day)
- Timeout protection (30 minutes max)
- Log deduplication (no duplicate messages)

### Rollback Behavior

**Default (Auto-Rollback Enabled):**
```bash
python deploy-model.py --model-package-arn arn:aws:sagemaker:...
```
- Failed deployments automatically delete endpoint and configuration
- Useful for production to avoid orphaned resources

**Disable Rollback (Keep Failed Endpoints):**
```bash
python deploy-model.py \
  --model-package-arn arn:aws:sagemaker:... \
  --skip-rollback
```
- Failed deployments remain for debugging
- Manual cleanup required

**Skip Monitoring (Faster, No Log Visibility):**
```bash
python deploy-model.py \
  --model-package-arn arn:aws:sagemaker:... \
  --no-monitor
```
- Deployment continues in background
- No log streaming or status updates
- Check status manually with AWS CLI

### Manual Cleanup

If deployment fails without auto-rollback:

```bash
# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name <config-name>
```

## Troubleshooting

### Common Issues

**Issue: Invalid model package ARN format**
```
ERROR: Invalid model package ARN: Invalid ARN format. Must start with 'arn:aws:sagemaker:'
```
**Solution:** 
- Use correct ARN format: `arn:aws:sagemaker:<region>:<account>:model-package/<group>/<version>`
- Get valid ARN from: `python utils/package-model.py --output-file model_arn.txt`
- Or list packages: `python utils/list_model_packages.py`

**Issue: Invalid instance type**
```
ERROR: Invalid instance type: Invalid instance type format. Must start with 'ml.'
```
**Solution:** 
- Use valid SageMaker instance type format: `ml.<family>.<size>`
- Recommended for QC linear models:
  - Production: `ml.c5.2xlarge`, `ml.c5.xlarge`
  - Testing: `ml.t2.medium`
- Avoid GPU instances for linear models (cost-ineffective)

**Issue: Environment variables not set**
```
ERROR: Missing Required Environment Variables
  - AWS_REGION: AWS region for SageMaker deployment
```
**Solution:** Export all required environment variables for the script you're running. See [Environment Setup](#environment-setup).

**Issue: Deployment failed with invalid artifacts**
```
DEPLOYMENT FAILED
Status: Failed
Common Causes:
  1. Invalid model artifacts (missing model.joblib or inference.py)
```
**Solution:**
- Verify model artifacts structure in tar.gz:
  ```bash
  tar -tzf model.tar.gz
  # Should contain: model.joblib, inference.py, requirements.txt, code/, model/
  ```
- Check model format: `model.joblib` must be scikit-learn linear model
- Validate inference.py has required functions: `model_fn`, `input_fn`, `predict_fn`
- Test locally with Docker before deploying (see `artifacts/README.md`)
- **Automatic cleanup:** Failed deployments are rolled back automatically (no broken endpoints left)

**Issue: Model package not found or not approved**
```
ERROR: Could not find model package: arn:aws:sagemaker:...
```
**Solution:** 
- Verify the ARN is correct
- Check package status: `python utils/list_model_packages.py`
- Ensure package is approved (package-model.py auto-approves)
- Model package must exist in same region as deployment

**Issue: Endpoint already exists**
```
ResourceInUse: Endpoint 'my-endpoint' already exists
```
**Solution:** 
- Use a different endpoint name with `--endpoint-name`
- Delete the existing endpoint if no longer needed
- Let auto-generation create a unique timestamped name

**Issue: Deployment stuck or timeout**
```
⚠ Timeout: Endpoint did not reach 'InService' status
```
**Solution:** 
- Check CloudWatch logs in AWS Console
- Verify instance type availability in your region
- Check SageMaker quotas and limits
- Review endpoint status: `aws sagemaker describe-endpoint --endpoint-name <name>`

**Issue: Container image not found**
```
Could not find image: 123456789.dkr.ecr.us-west-2.amazonaws.com/...
```
**Solution:** 
- Build and push container image first (see `artifacts/README.md`)
- Verify ECR repository exists
- Check image tag matches (default: `latest`)

**Issue: Insufficient IAM permissions**
```
AccessDeniedException: User is not authorized to perform: sagemaker:CreateEndpoint
```
**Solution:**
- Ensure your AWS credentials have required SageMaker permissions
- Verify `SAGEMAKER_EXECUTION_ROLE_ARN` has proper trust relationship
- Check IAM policies for SageMaker, S3, and CloudWatch access

**Issue: Log group not appearing**
```
Waiting for CloudWatch log group to be available...
```
**Solution:** 
- Wait up to 2 minutes for SageMaker to create the log group
- This is normal behavior on first deployment
- Logs will start streaming once container initializes

### Debug Mode

For detailed troubleshooting:

```bash
# Enable verbose output in test_endpoint.py
python test_endpoint.py --endpoint-name <name> --input-data data.json --verbose

# Check endpoint details
python -c "
from utils.endpoint_operations import get_endpoint_info
from utils.boto_session import get_boto_session
import os

session = get_boto_session(region=os.environ['AWS_REGION'])
client = session.client('sagemaker')
info = get_endpoint_info(client, '<endpoint-name>')
print(info)
"

# Monitor CloudWatch logs manually
aws logs tail /aws/sagemaker/Endpoints/<endpoint-name> --follow
```

### Getting Help

1. Check CloudWatch logs for detailed error messages
2. Review SageMaker endpoint events in AWS Console
3. Verify all environment variables are correctly set
4. Ensure container image is built and pushed to ECR
5. Check AWS service quotas and limits

## Architecture Notes

### Modular Design

The codebase follows a modular architecture:

- **Separation of Concerns**: Main scripts, processing modules, and utilities are clearly separated
- **Reusable Components**: Utility modules (`boto_session`, `env_validation`, `output`) are shared across all scripts
- **Consistent Patterns**: All scripts use the same utilities for session management, validation, and output
- **Single Source of Truth**: Common functionality exists in one place, reducing duplication

### Benefits

1. **Maintainability**: Changes to common functionality only need to be made once
2. **Consistency**: All scripts have consistent error handling and output formatting
3. **Testability**: Utility functions can be tested independently
4. **Extensibility**: Easy to add new scripts that leverage existing utilities
5. **Readability**: Main scripts focus on workflow orchestration, not implementation details

### Future Enhancements

Potential improvements:
- Add unit tests for utility modules
- Create integration tests for end-to-end workflows
- Add retry logic for transient failures
- Support for multi-region deployments
- Batch endpoint deployment support
- Model versioning and A/B testing utilities

---

**Last Updated:** January 2026  
**Author:** PHSA AI Team
