"""
Output Formatting Utilities

Provides consistent output formatting functions for headers, sections,
tables, and JSON data across deployment scripts.
"""

import json
from typing import Dict, Any, Optional, List


def print_header(title: str, width: int = 60, char: str = "=") -> None:
    """
    Print a formatted header.
    
    Args:
        title: Header title text
        width: Total width of the header (default: 60)
        char: Character to use for border (default: "=")
    
    Example:
        >>> print_header("MODEL DEPLOYMENT")
        ============================================================
        MODEL DEPLOYMENT
        ============================================================
    """
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def print_section(
    title: str, 
    items: Dict[str, Any], 
    width: int = 60, 
    char: str = "─"
) -> None:
    """
    Print a formatted section with key-value pairs.
    
    Args:
        title: Section title
        items: Dictionary of key-value pairs to display
        width: Total width of the section (default: 60)
        char: Character to use for border (default: "─")
    
    Example:
        >>> config = {"Region": "us-west-2", "Instance": "ml.c5.2xlarge"}
        >>> print_section("Configuration", config)
        ────────────────────────────────────────────────────────────
        Configuration
        ────────────────────────────────────────────────────────────
        Region:        us-west-2
        Instance:      ml.c5.2xlarge
        ────────────────────────────────────────────────────────────
    """
    print(f"{char * width}")
    print(title)
    print(f"{char * width}")
    
    # Find max key length for alignment
    max_key_len = max(len(str(k)) for k in items.keys()) if items else 0
    
    for key, value in items.items():
        print(f"{str(key):{max_key_len}s}: {value}")
    
    print(f"{char * width}\n")


def print_kv_pairs(
    items: Dict[str, Any],
    indent: int = 0,
    key_width: int = 20
) -> None:
    """
    Print key-value pairs with consistent formatting.
    
    Args:
        items: Dictionary of items to print
        indent: Number of spaces to indent (default: 0)
        key_width: Width allocated for keys (default: 20)
    
    Example:
        >>> info = {"Name": "my-endpoint", "Status": "InService"}
        >>> print_kv_pairs(info, indent=2, key_width=15)
          Name          : my-endpoint
          Status        : InService
    """
    indent_str = " " * indent
    for key, value in items.items():
        print(f"{indent_str}{str(key):{key_width}s}: {value}")


def print_json_result(
    data: Dict[str, Any], 
    title: str = "RESULTS",
    width: int = 60,
    indent: int = 2
) -> None:
    """
    Print JSON data with a formatted header.
    
    Args:
        data: Dictionary to format as JSON
        title: Section title (default: "RESULTS")
        width: Width of header (default: 60)
        indent: JSON indentation level (default: 2)
    
    Example:
        >>> result = {"predictions": ["qc_pass"], "confidence": [0.95]}
        >>> print_json_result(result, "PREDICTION RESULTS")
        ============================================================
        PREDICTION RESULTS
        ============================================================
        
        {
          "predictions": ["qc_pass"],
          "confidence": [0.95]
        }
        
        ============================================================
    """
    print(f"\n{'=' * width}")
    print(title)
    print(f"{'=' * width}\n")
    print(json.dumps(data, indent=indent))
    print(f"\n{'=' * width}\n")


def print_table(
    headers: List[str],
    rows: List[List[Any]],
    title: Optional[str] = None,
    width: int = 60
) -> None:
    """
    Print a simple table with headers and rows.
    
    Args:
        headers: List of column headers
        rows: List of rows (each row is a list of values)
        title: Optional table title
        width: Total width (for title border)
    
    Example:
        >>> headers = ["Name", "Status", "Instance"]
        >>> rows = [
        ...     ["endpoint-1", "InService", "ml.c5.2xlarge"],
        ...     ["endpoint-2", "Creating", "ml.m5.xlarge"]
        ... ]
        >>> print_table(headers, rows, "ENDPOINTS")
    """
    if title:
        print(f"\n{'=' * width}")
        print(title)
        print(f"{'=' * width}\n")
    
    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    # Print header
    header_str = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(row_str)
    
    print()


def print_success(message: str) -> None:
    """
    Print a success message with checkmark in green color.
    
    Args:
        message: Success message to display
    
    Example:
        >>> print_success("Deployment completed")
        ✓ Deployment completed
    """
    print(f"\033[92m✓ {message}\033[0m")


def print_error(message: str) -> None:
    """
    Print an error message with X mark in red color.
    
    Args:
        message: Error message to display
    
    Example:
        >>> print_error("Deployment failed")
        ✗ Deployment failed
    """
    print(f"\033[91m✗ {message}\033[0m")


def print_warning(message: str) -> None:
    """
    Print a warning message with warning symbol in amber color.
    
    Args:
        message: Warning message to display
    
    Example:
        >>> print_warning("Endpoint not ready")
        ⚠  Endpoint not ready
    """
    print(f"\033[93m⚠  {message}\033[0m")


def print_info(message: str, prefix: str = ">") -> None:
    """
    Print an info message with custom prefix.
    
    Args:
        message: Info message to display
        prefix: Prefix character (default: ">")
    
    Example:
        >>> print_info("Checking endpoint status")
        > Checking endpoint status
    """
    print(f"{prefix} {message}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        str: Formatted duration (e.g., "1m 30s", "45s", "2h 15m")
    
    Example:
        >>> print(format_duration(95))
        1m 35s
        >>> print(format_duration(3665))
        1h 1m 5s
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def format_bytes(bytes_count: int) -> str:
    """
    Format byte count to human-readable string.
    
    Args:
        bytes_count: Number of bytes
    
    Returns:
        str: Formatted size (e.g., "1.5 KB", "2.3 MB", "1.2 GB")
    
    Example:
        >>> print(format_bytes(1536))
        1.5 KB
        >>> print(format_bytes(2621440))
        2.5 MB
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"
