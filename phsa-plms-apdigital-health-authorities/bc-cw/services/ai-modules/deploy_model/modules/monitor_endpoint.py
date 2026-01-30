#!/usr/bin/env python3
"""
Monitor SageMaker endpoint deployment and CloudWatch logs.
"""

import sys
import time
from utils.output import print_error, print_warning, print_success
from datetime import datetime, timedelta


def wait_for_log_group(logs_client, log_group_name, max_wait_seconds=300, check_interval=5):
    """
    Wait for CloudWatch log group to be available.
    
    Args:
        logs_client: Boto3 CloudWatch Logs client
        log_group_name: Name of the log group
        max_wait_seconds: Maximum time to wait (default: 300 seconds)
        check_interval: How often to check (default: 5 seconds)
    
    Returns:
        bool: True if log group is available, False if timeout
    """
    print(f"Waiting for CloudWatch log group: {log_group_name}")
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait_seconds:
        try:
            response = logs_client.describe_log_groups(
                logGroupNamePrefix=log_group_name,
                limit=1
            )
            
            for log_group in response.get('logGroups', []):
                if log_group['logGroupName'] == log_group_name:
                    print_success(f"Log group available: {log_group_name}")
                    return True
            
            time.sleep(check_interval)
            print(".", end="", flush=True)
            
        except Exception as e:
            print_warning(f"\nWarning: Error checking log group: {e}")
            time.sleep(check_interval)
    
    print_warning(f"Log group not available after {max_wait_seconds}s")
    return False


def tail_logs(logs_client, log_group_name, stop_event, stream_filter=None):
    """
    Tail CloudWatch logs until stop event is set.
    
    Args:
        logs_client: Boto3 CloudWatch Logs client
        log_group_name: Name of the log group
        stop_event: Threading event to signal when to stop tailing
        stream_filter: Optional filter for log stream names
    """
    print(f"\n{'='*60}")
    print(f"Tailing logs from: {log_group_name}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    # Look back 10 minutes to catch any early logs
    last_timestamp = int((start_time - timedelta(minutes=10)).timestamp() * 1000)
    seen_events = set()
    no_logs_count = 0
    
    try:
        while not stop_event.is_set():
            kwargs = {
                'logGroupName': log_group_name,
                'startTime': last_timestamp,
                'limit': 100
            }
            
            if stream_filter:
                kwargs['logStreamNamePrefix'] = stream_filter
            
            try:
                response = logs_client.filter_log_events(**kwargs)
                events = response.get('events', [])
                
                if events:
                    no_logs_count = 0
                    for event in events:
                        event_id = f"{event['logStreamName']}_{event['timestamp']}_{event['eventId']}"
                        
                        # Skip duplicate events
                        if event_id in seen_events:
                            continue
                        seen_events.add(event_id)
                        
                        timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                        message = event['message'].rstrip()
                        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
                        last_timestamp = max(last_timestamp, event['timestamp'] + 1)
                else:
                    no_logs_count += 1
                    if no_logs_count % 10 == 0:
                        print(".", end="", flush=True)
                
                time.sleep(3)
                
            except logs_client.exceptions.ResourceNotFoundException:
                print("Log group or stream not found yet, waiting...")
                time.sleep(10)
            except Exception as e:
                print_error(f"Error reading logs: {e}")
                time.sleep(5)
                
    except KeyboardInterrupt:
        print_warning("Log tailing stopped by user")


def monitor_endpoint_status(sm_client, endpoint_name, check_interval=30, max_wait_seconds=1800):
    """
    Monitor SageMaker endpoint deployment status.
    
    Args:
        sm_client: Boto3 SageMaker client
        endpoint_name: Name of the endpoint
        check_interval: How often to check status (default: 30 seconds)
        max_wait_seconds: Maximum time to wait (default: 1800 seconds / 30 minutes)
    
    Returns:
        str: Final endpoint status ('InService', 'Failed', 'Timeout')
    """
    print(f"\n{'='*60}")
    print(f"Monitoring endpoint: {endpoint_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    last_status = None
    
    while (time.time() - start_time) < max_wait_seconds:
        try:
            response = sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if status != last_status:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Endpoint status: {status}")
                
                if 'FailureReason' in response:
                    print(f"  ⚠ Failure reason: {response['FailureReason']}")
                
                last_status = status
            
            if status == 'InService':
                print_success(f"Endpoint is InService and ready!")
                return 'InService'
            
            elif status == 'Failed':
                print_error(f"Endpoint deployment failed!")
                if 'FailureReason' in response:
                    print(f"Failure reason: {response['FailureReason']}")
                return 'Failed'
            
            elif status in ['Creating', 'Updating']:
                time.sleep(check_interval)
            
            else:
                print(f"⚠ Unexpected status: {status}")
                time.sleep(check_interval)
                
        except sm_client.exceptions.ClientError as e:
            if 'Could not find endpoint' in str(e):
                print(f"Endpoint not found yet, waiting...")
                time.sleep(check_interval)
            else:
                print_error(f"Error checking endpoint status: {e}")
                return 'Error'
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            return 'Error'
    
    print_error(f"Endpoint deployment timeout after {max_wait_seconds}s")
    return 'Timeout'


def delete_endpoint(sm_client, endpoint_name, delete_config=True):
    """
    Delete a SageMaker endpoint and optionally its configuration.
    
    Args:
        sm_client: Boto3 SageMaker client
        endpoint_name: Name of the endpoint to delete
        delete_config: Also delete endpoint configuration (default: True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Delete endpoint
        print(f"Deleting endpoint: {endpoint_name}")
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print_success(f"Endpoint deletion initiated: {endpoint_name}")
        
        if delete_config:
            # Delete endpoint configuration
            endpoint_config_name = endpoint_name  # Usually same name
            try:
                sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                print_success(f"Endpoint config deleted: {endpoint_config_name}")
            except sm_client.exceptions.ClientError as e:
                print_warning(f"Could not delete endpoint config: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error deleting endpoint: {e}", file=sys.stderr)
        return False


def monitor_and_tail(sm_client, logs_client, endpoint_name, log_group_name=None, max_wait=1800):
    """
    Complete monitoring workflow: monitor endpoint and tail logs simultaneously.
    
    Args:
        sm_client: Boto3 SageMaker client
        logs_client: Boto3 CloudWatch Logs client
        endpoint_name: Name of the endpoint
        log_group_name: CloudWatch log group name (default: /aws/sagemaker/Endpoints/{endpoint_name})
        max_wait: Maximum time to wait for endpoint (default: 1800 seconds)
    
    Returns:
        str: Final endpoint status
    """
    if not log_group_name:
        log_group_name = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    
    print(f"\n{'='*60}")
    print(f"Starting endpoint deployment monitoring")
    print(f"Endpoint: {endpoint_name}")
    print(f"Log Group: {log_group_name}")
    print(f"{'='*60}\n")
    
    # Set log retention to 1 day
    try:
        logs_client.put_retention_policy(
            logGroupName=log_group_name,
            retentionInDays=1
        )
        print(f"✓ Log retention set to 1 day")
    except logs_client.exceptions.ResourceNotFoundException:
        print(f"⚠ Log group not created yet, will set retention once available")
    except Exception as e:
        print(f"⚠ Could not set log retention: {e}")
    
    # Start monitoring in parallel
    import threading
    
    status_result = {'status': None}
    stop_event = threading.Event()
    
    def monitor_status():
        status = monitor_endpoint_status(sm_client, endpoint_name, 
                                        check_interval=30, max_wait_seconds=max_wait)
        status_result['status'] = status
        # Signal log tailing to stop
        stop_event.set()
    
    def tail_logs_thread():
        # Wait a bit for log group to be created
        time.sleep(10)
        tail_logs(logs_client, log_group_name, stop_event)
    
    # Start both threads
    status_thread = threading.Thread(target=monitor_status, daemon=False)
    logs_thread = threading.Thread(target=tail_logs_thread, daemon=False)
    
    status_thread.start()
    logs_thread.start()
    
    # Wait for both to complete
    status_thread.join()
    logs_thread.join(timeout=10)  # Give logs thread a bit to finish
    
    # Try to set retention one more time in case log group was just created
    try:
        logs_client.put_retention_policy(
            logGroupName=log_group_name,
            retentionInDays=1
        )
    except:
        pass
    
    return status_result.get('status', 'Unknown')


if __name__ == "__main__":
    import boto3
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor SageMaker endpoint deployment.")
    parser.add_argument("-e", "--endpoint-name", required=True, help="Endpoint name")
    parser.add_argument("-l", "--log-group", help="CloudWatch log group name")
    parser.add_argument("-d", "--duration", type=int, default=120, help="Log tail duration (seconds)")
    parser.add_argument("-r", "--region", default="us-east-1", help="AWS region")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint after monitoring")
    args = parser.parse_args()
    
    # Create clients
    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        boto_session = boto3.Session(profile_name=aws_profile, region_name=args.region)
    else:
        boto_session = boto3.Session(region_name=args.region)
    
    sm_client = boto_session.client("sagemaker")
    logs_client = boto_session.client("logs")
    
    # Monitor endpoint
    status = monitor_and_tail(
        sm_client, 
        logs_client, 
        args.endpoint_name, 
        args.log_group,
        tail_duration=args.duration
    )
    
    print(f"\n\nFinal status: {status}")
    
    # Delete if requested and failed
    if args.delete and status in ['Failed', 'Timeout']:
        print("\nDeleting failed endpoint...")
        delete_endpoint(sm_client, args.endpoint_name)
