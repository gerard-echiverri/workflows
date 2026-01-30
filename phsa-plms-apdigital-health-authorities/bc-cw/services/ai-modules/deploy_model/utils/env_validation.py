"""
Environment Variable Validation Utilities

Provides helper functions for validating required environment variables
with consistent error messaging across all deployment scripts.
"""

import os
import sys
from typing import List, Tuple, Dict, Optional


def validate_env_vars(
    required_vars: List[Tuple[str, str]], 
    script_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Validate that required environment variables are set and non-empty.
    
    Args:
        required_vars: List of (variable_name, description) tuples
        script_name: Optional script name for error context
    
    Returns:
        Dict[str, str]: Dictionary of validated environment variables
    
    Raises:
        SystemExit: If any required variable is missing or empty
    
    Example:
        >>> required = [
        ...     ("AWS_REGION", "AWS region for deployment"),
        ...     ("AWS_ACCOUNT", "AWS account ID")
        ... ]
        >>> env_vars = validate_env_vars(required, "package-model.py")
        >>> print(env_vars["AWS_REGION"])
    """
    errors = []
    validated = {}
    
    for var_name, description in required_vars:
        value = os.environ.get(var_name)
        if not value or value.strip() == "":
            errors.append(f"  - {var_name}: {description}")
        else:
            validated[var_name] = value
    
    if errors:
        print("\n" + "="*60)
        print("ERROR: Missing Required Environment Variables")
        if script_name:
            print(f"Script: {script_name}")
        print("="*60)
        for error in errors:
            print(error)
        print("\nPlease set these environment variables before running.")
        print("\nExample:")
        for var_name, _ in required_vars[:2]:  # Show first 2 as examples
            print(f"  export {var_name}=<value>")
        print("="*60 + "\n")
        sys.exit(1)
    
    return validated


def get_env_var(
    var_name: str, 
    default: Optional[str] = None, 
    required: bool = True
) -> str:
    """
    Get a single environment variable with optional default.
    
    Args:
        var_name: Environment variable name
        default: Default value if not set
        required: Whether the variable is required (ignored if default provided)
    
    Returns:
        str: Environment variable value or default
    
    Raises:
        SystemExit: If required variable is not set and no default provided
    
    Example:
        >>> region = get_env_var("AWS_REGION", default="us-west-2", required=False)
        >>> account = get_env_var("AWS_ACCOUNT")  # Will exit if not set
    """
    value = os.environ.get(var_name)
    
    if value and value.strip():
        return value
    
    if default is not None:
        return default
    
    if required:
        print(f"\nERROR: Required environment variable '{var_name}' is not set.")
        print(f"Please set it before running this script:")
        print(f"  export {var_name}=<value>\n")
        sys.exit(1)
    
    return ""


def print_env_summary(env_vars: Dict[str, str], title: str = "Environment Configuration"):
    """
    Print a formatted summary of environment variables.
    
    Args:
        env_vars: Dictionary of environment variables
        title: Section title
    
    Example:
        >>> env = {"AWS_REGION": "us-west-2", "AWS_ACCOUNT": "123456789"}
        >>> print_env_summary(env, "AWS Configuration")
    """
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    for key, value in env_vars.items():
        # Mask sensitive values (credentials, secrets)
        if any(word in key.upper() for word in ["KEY", "SECRET", "TOKEN", "PASSWORD"]):
            masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"{key:30s}: {masked_value}")
        else:
            print(f"{key:30s}: {value}")
    print(f"{'='*60}\n")
