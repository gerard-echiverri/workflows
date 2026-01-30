#!/usr/bin/env python3
"""
Bundle model artifacts into a tar.gz file for SageMaker deployment.
"""

import os
import sys
import tarfile
import argparse


def bundle_model_artifacts(output_file="model.tar.gz"):
    """
    Bundle model artifacts into a tar.gz file.
    
    Args:
        model_artifact_path: Path to the model artifact file (e.g., model.joblib)
        inference_script_path: Path to the inference script (e.g., inference.py)
        output_file: Name of the output tar.gz file (default: model.tar.gz)
    
    Returns:
        str: Path to the created tar.gz file
    """
    try:
        # Create tar.gz bundle
        with tarfile.open(output_file, "w:gz") as tar:
            tar.add("artifacts", arcname=".")
        
        print(f"✓ Built {output_file}")
        return output_file
    
    except Exception as e:
        print(f"ERROR: Failed to bundle model artifacts: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Bundle model artifacts for SageMaker deployment.")
    parser.add_argument("-m", "--model-artifact", default="model.joblib", 
                        help="Model artifact (default: model.joblib)")
    parser.add_argument("-f", "--inference-file", default="inference.py", 
                        help="SageMaker inference script (default: inference.py)")
    parser.add_argument("-o", "--output", default="model.tar.gz", 
                        help="Output tar.gz file (default: model.tar.gz)")
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.model_artifact):
        print(f"ERROR: Model artifact not found: {args.model_artifact}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.inference_file):
        print(f"ERROR: Inference script not found: {args.inference_file}", file=sys.stderr)
        sys.exit(1)
    
    # Bundle artifacts
    bundle_model_artifacts(args.model_artifact, args.inference_file, args.output)


if __name__ == "__main__":
    main()
