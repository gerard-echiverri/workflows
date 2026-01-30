#!/usr/bin/env python3
"""
List SageMaker model package groups and their approved packages.

This script retrieves and displays model package groups and their associated
approved model packages, showing details like ARN, status, and creation time.
"""

import argparse
import boto3
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import json


def list_model_package_groups(sm_client, name_filter: str = None) -> List[Dict[str, Any]]:
    """
    List all model package groups.
    
    Args:
        sm_client: Boto3 SageMaker client
        name_filter: Optional filter for group names (substring match)
    
    Returns:
        list: Model package groups
    
    Usage:
        >>> sm_client = boto3.client('sagemaker')
        >>> groups = list_model_package_groups(sm_client)
        >>> filtered = list_model_package_groups(sm_client, name_filter="inference")
    """
    groups = []
    paginator = sm_client.get_paginator('list_model_package_groups')
    
    try:
        for page in paginator.paginate():
            for group in page.get('ModelPackageGroupSummaryList', []):
                if name_filter is None or name_filter.lower() in group['ModelPackageGroupName'].lower():
                    groups.append(group)
        return groups
    except Exception as e:
        print(f"Error listing model package groups: {e}", file=sys.stderr)
        return []


def list_model_packages(sm_client, group_name: str, approval_status: str = "Approved") -> List[Dict[str, Any]]:
    """
    List model packages in a group filtered by approval status.
    
    Args:
        sm_client: Boto3 SageMaker client
        group_name: Name of the model package group
        approval_status: Filter by approval status (default: Approved)
    
    Returns:
        list: Model packages matching the criteria
    
    Usage:
        >>> sm_client = boto3.client('sagemaker')
        >>> approved = list_model_packages(sm_client, "MyModelGroup")
        >>> pending = list_model_packages(sm_client, "MyModelGroup", "PendingManualApproval")
    """
    packages = []
    paginator = sm_client.get_paginator('list_model_packages')
    
    try:
        for page in paginator.paginate(
            ModelPackageGroupName=group_name,
            ModelApprovalStatus=approval_status,
            SortBy='CreationTime',
            SortOrder='Descending'
        ):
            packages.extend(page.get('ModelPackageSummaryList', []))
        return packages
    except Exception as e:
        print(f"\nNo model packages found in group: '{group_name}':\n\n {e}", file=sys.stderr)
        return []


def get_model_package_details(sm_client, package_arn: str) -> Dict[str, Any]:
    """
    Get detailed information about a model package.
    
    Args:
        sm_client: Boto3 SageMaker client
        package_arn: ARN of the model package
    
    Returns:
        dict: Model package details
    
    Usage:
        >>> sm_client = boto3.client('sagemaker')
        >>> arn = "arn:aws:sagemaker:region:account:model-package/group/version"
        >>> details = get_model_package_details(sm_client, arn)
        >>> print(details['ModelApprovalStatus'])
    """
    try:
        return sm_client.describe_model_package(ModelPackageName=package_arn)
    except Exception as e:
        print(f"Error getting details for package '{package_arn}': {e}", file=sys.stderr)
        return {}


def format_timestamp(timestamp) -> str:
    """Format timestamp for display.
    
    Usage:
        >>> from datetime import datetime
        >>> ts = datetime.now()
        >>> formatted = format_timestamp(ts)
        >>> print(formatted)  # '2026-01-29 14:30:45'
    """
    if isinstance(timestamp, datetime):
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return str(timestamp)


def print_summary(groups: List[Dict[str, Any]], packages_by_group: Dict[str, List[Dict[str, Any]]]):
    """
    Print summary of model package groups and packages.
    
    Args:
        groups: List of model package groups
        packages_by_group: Dictionary mapping group names to their packages
    
    Usage:
        >>> groups = list_model_package_groups(sm_client)
        >>> packages = {'GroupA': [pkg1, pkg2], 'GroupB': [pkg3]}
        >>> print_summary(groups, packages)
    """
    print(f"\n{'='*80}")
    print("MODEL PACKAGE GROUPS AND APPROVED PACKAGES")
    print(f"{'='*80}\n")
    
    if not groups:
        print("No model package groups found.")
        return
    
    total_packages = sum(len(pkgs) for pkgs in packages_by_group.values())
    print(f"Found {len(groups)} model package group(s) with {total_packages} approved package(s)\n")
    
    for group in groups:
        group_name = group['ModelPackageGroupName']
        packages = packages_by_group.get(group_name, [])
        
        print(f"{'─'*80}")
        print(f"Group: {group_name}")
        print(f"ARN:   {group['ModelPackageGroupArn']}")
        print(f"Created: {format_timestamp(group['CreationTime'])}")
        if group.get('ModelPackageGroupDescription'):
            print(f"Description: {group['ModelPackageGroupDescription']}")
        print(f"Approved Packages: {len(packages)}")
        
        if packages:
            print(f"\n  Approved Model Packages:")
            for i, pkg in enumerate(packages, 1):
                print(f"\n  {i}. Package ARN: {pkg['ModelPackageArn']}")
                print(f"     Status: {pkg['ModelApprovalStatus']}")
                print(f"     Version: {pkg.get('ModelPackageVersion', 'N/A')}")
                print(f"     Created: {format_timestamp(pkg['CreationTime'])}")
                if pkg.get('ModelPackageDescription'):
                    print(f"     Description: {pkg['ModelPackageDescription']}")
        else:
            print(f"\n  No approved packages found.")
        
        print()
    
    print(f"{'='*80}")


def print_json_output(groups: List[Dict[str, Any]], packages_by_group: Dict[str, List[Dict[str, Any]]]):
    """
    Print results as JSON.
    
    Args:
        groups: List of model package groups
        packages_by_group: Dictionary mapping group names to their packages
    
    Usage:
        >>> groups = list_model_package_groups(sm_client)
        >>> packages = {'GroupA': [pkg1, pkg2]}
        >>> print_json_output(groups, packages)
        # Outputs formatted JSON to stdout
    """
    output = []
    
    for group in groups:
        group_name = group['ModelPackageGroupName']
        packages = packages_by_group.get(group_name, [])
        
        # Convert datetime objects to strings for JSON serialization
        group_data = {
            'group_name': group_name,
            'group_arn': group['ModelPackageGroupArn'],
            'created': format_timestamp(group['CreationTime']),
            'description': group.get('ModelPackageGroupDescription', ''),
            'packages': []
        }
        
        for pkg in packages:
            group_data['packages'].append({
                'arn': pkg['ModelPackageArn'],
                'status': pkg['ModelApprovalStatus'],
                'version': pkg.get('ModelPackageVersion', 'N/A'),
                'created': format_timestamp(pkg['CreationTime']),
                'description': pkg.get('ModelPackageDescription', '')
            })
        
        output.append(group_data)
    
    print(json.dumps(output, indent=2))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="List SageMaker model package groups and approved packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all model package groups and their approved packages
  python list_model_packages.py
  
  # Filter by group name
  python list_model_packages.py --filter "inference"
  
  # Show all packages (not just approved)
  python list_model_packages.py --all-statuses
  
  # Output as JSON
  python list_model_packages.py --json
  
  # Specify AWS profile and region
  python list_model_packages.py --profile my-profile --region us-west-2
        """
    )
    
    parser.add_argument(
        "--filter", "-f",
        help="Filter model package groups by name (case-insensitive substring match)"
    )
    
    parser.add_argument(
        "--all-statuses", "-a",
        action="store_true",
        help="Show packages with all approval statuses (not just Approved)"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE", ""),
        help="AWS profile name"
    )
    
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "ca-central-1"),
        help="AWS region (default: ca-central-1)"
    )
    
    parser.add_argument(
        "--details", "-d",
        action="store_true",
        help="Show detailed information for each package"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize AWS client
        if args.profile:
            boto_session = boto3.Session(profile_name=args.profile, region_name=args.region)
        else:
            boto_session = boto3.Session(region_name=args.region)
        
        sm_client = boto_session.client('sagemaker')
        
        # List model package groups
        if not args.json:
            print(f"Fetching model package groups from {args.region}...")
        
        groups = list_model_package_groups(sm_client, args.filter)
        
        if not groups:
            if args.filter:
                print(f"No model package groups found matching filter: '{args.filter}'")
            else:
                print("No model package groups found.")
            return 0
        
        # Get packages for each group
        packages_by_group = {}
        approval_status = None if args.all_statuses else "Approved"
        
        for group in groups:
            group_name = group['ModelPackageGroupName']
            if not args.json:
                print(f"  Fetching packages for group: {group_name}...")
            
            if args.all_statuses:
                # Get all packages regardless of status
                packages = []
                for status in ["Approved", "PendingManualApproval", "Rejected"]:
                    packages.extend(list_model_packages(sm_client, group_name, status))
            else:
                packages = list_model_packages(sm_client, group_name, approval_status)
            
            packages_by_group[group_name] = packages
        
        # Print results
        if args.json:
            print_json_output(groups, packages_by_group)
        else:
            print_summary(groups, packages_by_group)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
