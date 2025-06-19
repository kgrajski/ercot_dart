#!/usr/bin/env python3
"""Test script to run synthetic data validation for ERCOT Exp1 models.

Run this from the project root directory to validate that our linear, ridge, 
and lasso regression implementations work correctly on synthetic data with 
known ground truth relationships.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.ercot.exp1.synthetic_validation import run_validation

if __name__ == "__main__":
    print("🚀 Starting Synthetic Data Validation for ERCOT Exp1 Models")
    print("This will test linear, ridge, and lasso regression implementations")
    print("with known ground truth relationships.\n")

    try:
        results = run_validation()

        print("\n🎉 Validation completed successfully!")
        print(
            "Check the 'synthetic_validation_results' directory for detailed results."
        )

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
