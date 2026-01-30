#!/usr/bin/env python3
"""
Bundle model artifacts into a tar.gz file for SageMaker deployment.
"""

import os
import sys
import tarfile
import argparse


def bundle_model_artifacts(model_artifact_path, inference_script_path, output_file="model.tar.gz", requirements_path=None):
    """
    Bundle model artifacts into a tar.gz file.
    
    Args:
        model_artifact_path: Path to the model artifact file (e.g., model.joblib)
        inference_script_path: Path to the inference script (e.g., inference.py)
        output_file: Name of the output tar.gz file (default: model.tar.gz)
        requirements_path: Optional path to requirements.txt file
    
    Returns:
        str: Path to the created tar.gz file
    """
    try:
        # Create tar.gz bundle
        with tarfile.open(output_file, "w:gz") as tar:
            # Add model artifact with base name only
            tar.add(model_artifact_path, arcname=os.path.basename(model_artifact_path))
            # Add inference script with base name only
            tar.add(inference_script_path, arcname=os.path.basename(inference_script_path))
            
            # Add requirements.txt if provided and exists
            if requirements_path and os.path.exists(requirements_path):
                tar.add(requirements_path, arcname=os.path.basename(requirements_path))
        
        print(f"✓ Built {output_file}")
        print(f"  - Added: {os.path.basename(model_artifact_path)}")
        print(f"  - Added: {os.path.basename(inference_script_path)}")
        if requirements_path and os.path.exists(requirements_path):
            print(f"  - Added: {os.path.basename(requirements_path)}")
        return output_file
    
    except Exception as e:
        print(f"ERROR: Failed to bundle model artifacts: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Bundle model artifacts for SageMaker deployment.")
    parser.add_argument("-m", "--model-artifact", default=None, 
                        help="Model artifact (default: model.joblib)")
    parser.add_argument("-f", "--inference-file", default=None, 
                        help="SageMaker inference script (default: inference.py)")
    parser.add_argument("-d", "--artifacts-dir", default=None, 
                        help="Directory containing model artifacts (model.joblib and inference.py)")
    parser.add_argument("-o", "--output", default="model.tar.gz", 
                        help="Output tar.gz file (default: model.tar.gz)")
    args = parser.parse_args()
    
    # Determine paths based on artifacts-dir or individual arguments
    if args.artifacts_dir:
        # Use artifacts directory
        if not os.path.isdir(args.artifacts_dir):
            print(f"ERROR: Artifacts directory not found: {args.artifacts_dir}", file=sys.stderr)
            sys.exit(1)
        
        model_artifact = os.path.join(args.artifacts_dir, args.model_artifact or "model.joblib")
        inference_file = os.path.join(args.artifacts_dir, args.inference_file or "inference.py")
        
        # Check for requirements.txt in the artifacts directory
        requirements_file = os.path.join(args.artifacts_dir, "requirements.txt")
        if not os.path.exists(requirements_file):
            requirements_file = None
    else:
        # Use individual file paths
        model_artifact = args.model_artifact or "model.joblib"
        inference_file = args.inference_file or "inference.py"
        requirements_file = None
    
    # Validate input files exist
    if not os.path.exists(model_artifact):
        print(f"ERROR: Model artifact not found: {model_artifact}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(inference_file):
        print(f"ERROR: Inference script not found: {inference_file}", file=sys.stderr)
        sys.exit(1)
    
    # Bundle artifacts
    bundle_model_artifacts(model_artifact, inference_file, args.output, requirements_file)


if __name__ == "__main__":
    main()
