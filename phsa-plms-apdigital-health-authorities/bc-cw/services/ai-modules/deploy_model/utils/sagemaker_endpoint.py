"""
Helpers for creating or updating SageMaker endpoints and waiting for endpoint status.
Splits logic from model and endpoint config creation for modularity.
"""

import time
from botocore.exceptions import ClientError

def wait_for_endpoint(sagemaker_client, endpoint_name):
    """
    Wait for a SageMaker endpoint to reach 'InService' status.
    Raises an error if the endpoint enters a failed state.
    Args:
        sagemaker_client (boto3.client): SageMaker client.
        endpoint_name (str): Name of the endpoint to monitor.
    """
    desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = desc['EndpointStatus']
    while status in ("Creating", "Updating", "RollingBack"):
        print(f"Waiting for endpoint {endpoint_name} to become available (current status: {status})...")
        time.sleep(5)
        status = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']
    if status != "InService":
        raise RuntimeError(f"Endpoint did not reach InService state. Current state: {status}")

def create_or_update_endpoint(sagemaker_client, endpoint_name, config_name):
    """
    Create a new SageMaker endpoint or update an existing one with a new config.
    Waits for the endpoint to be available before updating.
    Args:
        sagemaker_client (boto3.client): SageMaker client.
        endpoint_name (str): Name of the endpoint.
        config_name (str): Name of the endpoint config to use.
    """
    try:
        desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = desc['EndpointStatus']
        print(f"Endpoint {endpoint_name} exists with status: {status}")
        wait_for_endpoint(sagemaker_client, endpoint_name)
        print(f"Updating endpoint: {endpoint_name}")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"Endpoint updated: {endpoint_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'Could not find endpoint' in str(e):
            print(f"Endpoint does not exist. Creating endpoint: {endpoint_name}")
            sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            print(f"Endpoint created: {endpoint_name}")
        else:
            raise
