"""
Instance Type Validation Utilities

Provides validation functions for SageMaker instance types with
recommendations for specific use cases.
"""


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
