"""Script for creating the study dataset for experiment 2."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path for direct script execution
# This allows the script to work with both "Run Python File" and "python -m" approaches
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.ercot.exp2 import Exp2Dataset


def main():
    """Main function for preparing the experiment 2 study dataset."""

    script_name = "03-kag-exp2-ercot-study-dataset"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects"
    project_dir = os.path.join(root_dir, "ercot_dart")

    # Required input files (from processed data directory):
    # - rt_spp_transformed.csv: Real-Time Settlement Point Prices with hourly statistics
    #                          Used as target variables (price_mean, price_std)
    # - dam_spp_clean.csv: Day-Ahead Settlement Point Prices
    #                      Used as features (previous day's price expectations)
    # - dam_system_lambda_clean.csv: Day-Ahead System Lambda
    #                      Used as features (system-wide price expectations)
    # - load_forecast_clean.csv: Load Forecast by weather zone
    #                           Used as features (demand expectations)
    # - wind_power_gen_clean.csv: Wind Generation forecast by region
    #                             Used as features (supply expectations)
    # - solar_power_gen_clean.csv: Solar Generation forecast by region
    #                              Used as features (supply expectations)

    # Identify the input directory by date
    processed_data_date = "2025-06-04"
    input_dir = os.path.join(project_dir, "data/processed", processed_data_date)
    output_dir = os.path.join(project_dir, "data/studies/exp2", processed_data_date)
    os.makedirs(output_dir, exist_ok=True)

    print("\nInitializing Exp2Dataset handler...")
    exp2 = Exp2Dataset(input_dir=input_dir, output_dir=output_dir)

    print("\nLoading and validating input data...")
    exp2.load_data()

    print("\nGenerating dependent variables...")
    exp2.generate_dependent_vars()

    print("\nGenerating independent variables...")
    exp2.generate_independent_vars()

    print("\nFinalizing study dataset...")
    exp2.finalize_study_dataset()

    print("\nRunning exploratory data analysis...")
    exp2.run_eda()

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()
