"""
SageMaker Endpoint Operations Utilities

Provides helper functions for common SageMaker endpoint operations
including status checking, monitoring, and information retrieval.
"""

import time
from typing import Dict, Optional, Any


def check_endpoint_status(sagemaker_client, endpoint_name: str) -> str:
    """
    Check if a SageMaker endpoint exists and return its status.
    
    Args:
        sagemaker_client: Boto3 SageMaker client
        endpoint_name: Name of the endpoint
    
    Returns:
        str: Endpoint status (InService, Creating, Failed, etc.)
    
    Raises:
        Exception: If endpoint doesn't exist
    
    Example:
        >>> import boto3
        >>> client = boto3.client('sagemaker')
        >>> status = check_endpoint_status(client, 'my-endpoint')
        >>> print(f"Status: {status}")
    """
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except sagemaker_client.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            raise Exception(
                f"Endpoint '{endpoint_name}' not found. "
                "Please deploy the model first."
            )
        raise


def get_endpoint_info(sagemaker_client, endpoint_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a SageMaker endpoint.
    
    Args:
        sagemaker_client: Boto3 SageMaker client
        endpoint_name: Name of the endpoint
    
    Returns:
        Dict containing endpoint details (status, instance type, creation time, etc.)
    
    Raises:
        Exception: If endpoint doesn't exist
    
    Example:
        >>> info = get_endpoint_info(client, 'my-endpoint')
        >>> print(f"Created: {info['CreationTime']}")
        >>> print(f"Instance: {info['InstanceType']}")
    """
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        
        # Get endpoint config for instance details
        config_name = response['EndpointConfigName']
        config_response = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=config_name
        )
        
        # Extract production variant info
        variant = config_response['ProductionVariants'][0] if config_response['ProductionVariants'] else {}
        
        return {
            'endpoint_name': response['EndpointName'],
            'status': response['EndpointStatus'],
            'creation_time': response['CreationTime'],
            'last_modified': response['LastModifiedTime'],
            'endpoint_arn': response['EndpointArn'],
            'endpoint_config_name': config_name,
            'instance_type': variant.get('InstanceType', 'Unknown'),
            'instance_count': variant.get('InitialInstanceCount', 0),
            'variant_name': variant.get('VariantName', 'Unknown'),
            'failure_reason': response.get('FailureReason', None)
        }
    except sagemaker_client.exceptions.ClientError as e:
        if 'Could not find endpoint' in str(e):
            raise Exception(f"Endpoint '{endpoint_name}' not found.")
        raise


def wait_for_endpoint(
    sagemaker_client, 
    endpoint_name: str, 
    target_status: str = 'InService',
    max_wait: int = 1800,
    poll_interval: int = 30
) -> str:
    """
    Wait for an endpoint to reach a target status.
    
    Args:
        sagemaker_client: Boto3 SageMaker client
        endpoint_name: Name of the endpoint
        target_status: Status to wait for (default: 'InService')
        max_wait: Maximum wait time in seconds (default: 1800 = 30 min)
        poll_interval: Seconds between status checks (default: 30)
    
    Returns:
        str: Final endpoint status
    
    Example:
        >>> status = wait_for_endpoint(client, 'my-endpoint', max_wait=600)
        >>> if status == 'InService':
        ...     print("Endpoint ready!")
    """
    start_time = time.time()
    elapsed = 0
    
    while elapsed < max_wait:
        try:
            status = check_endpoint_status(sagemaker_client, endpoint_name)
            
            if status == target_status:
                return status
            
            if status == 'Failed':
                # Get failure reason
                info = get_endpoint_info(sagemaker_client, endpoint_name)
                failure_reason = info.get('failure_reason', 'Unknown')
                raise Exception(
                    f"Endpoint deployment failed. Reason: {failure_reason}"
                )
            
            # Still in progress
            print(f"  Status: {status} (waiting... {int(elapsed)}s / {max_wait}s)")
            time.sleep(poll_interval)
            elapsed = time.time() - start_time
            
        except Exception as e:
            if 'not found' in str(e):
                print(f"  Endpoint not found (waiting... {int(elapsed)}s / {max_wait}s)")
                time.sleep(poll_interval)
                elapsed = time.time() - start_time
            else:
                raise
    
    # Timeout
    current_status = check_endpoint_status(sagemaker_client, endpoint_name)
    print(f"\n⚠ Timeout: Endpoint did not reach '{target_status}' status")
    print(f"   Current status: {current_status}")
    return current_status


def list_endpoints(
    sagemaker_client,
    name_contains: Optional[str] = None,
    status_filter: Optional[str] = None,
    max_results: int = 100
) -> list:
    """
    List SageMaker endpoints with optional filtering.
    
    Args:
        sagemaker_client: Boto3 SageMaker client
        name_contains: Filter by endpoint name substring
        status_filter: Filter by status (InService, Creating, Failed, etc.)
        max_results: Maximum number of results to return
    
    Returns:
        List of endpoint dictionaries
    
    Example:
        >>> endpoints = list_endpoints(client, name_contains='production')
        >>> for ep in endpoints:
        ...     print(f"{ep['EndpointName']}: {ep['EndpointStatus']}")
    """
    params = {'MaxResults': max_results}
    if name_contains:
        params['NameContains'] = name_contains
    if status_filter:
        params['StatusEquals'] = status_filter
    
    response = sagemaker_client.list_endpoints(**params)
    return response.get('Endpoints', [])


def endpoint_exists(sagemaker_client, endpoint_name: str) -> bool:
    """
    Check if an endpoint exists.
    
    Args:
        sagemaker_client: Boto3 SageMaker client
        endpoint_name: Name of the endpoint
    
    Returns:
        bool: True if endpoint exists, False otherwise
    
    Example:
        >>> if endpoint_exists(client, 'my-endpoint'):
        ...     print("Endpoint exists!")
    """
    try:
        check_endpoint_status(sagemaker_client, endpoint_name)
        return True
    except Exception:
        return False
