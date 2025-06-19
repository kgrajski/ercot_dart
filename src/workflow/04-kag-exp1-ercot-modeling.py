"""Experiment 1 modeling workflow for ERCOT DART prediction.

This workflow performs machine learning modeling experiments on the finalized
exp1 study datasets. It processes each settlement point (location/location_type)
independently and trains models for DART price prediction.

Key features:
- Loads finalized datasets using DartSltExp1Dataset
- Supports multiple model types (starting with linear regression)
- Processes each settlement point separately
- Saves modeling results and artifacts
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split

from src.models.ercot.exp1 import DartSltExp1Dataset
from src.models.ercot.exp1.model_trainer import Exp1ModelTrainer


def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "04-kag-exp1-ercot-modeling"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")

    # Set the random seed for reproducibility
    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    # Experiment name
    exp_name = "exp1"
    model_types = ["linear_regression", "ridge_regression", "lasso_regression"]
    processed_data_date = "2025-06-04"
    print(
        f"\n** Experiment name: {exp_name} on processed dataset created: {processed_data_date}"
    )

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects/"
    project_dir = os.path.join(root_dir, "ercot_dart")
    data_dir = os.path.join(project_dir, "data/studies", exp_name, processed_data_date)

    # Generate the list of subdirectories in the input directory (in_data_dir)
    spp_loc_list = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    print(f"Subdirectories in {data_dir}: {spp_loc_list}")

    # Perform experiments within the scope of a single subdirectory (i.e., a settlement price point).
    # Everything in this loop will assume that the dataset is for a single location/location_type.
    for spp_loc in spp_loc_list:
        print(f"\n** Training model(s) for {spp_loc}")
        spp_dir = os.path.join(data_dir, spp_loc)
        modeling_dir = os.path.join(spp_dir, "modeling")

        # Create and initialize dataset (automatically finds final dataset; creates modeling directory)
        study_dataset = DartSltExp1Dataset(spp_dir=spp_dir, output_dir=modeling_dir)

        # Initialize model trainer
        print(f"** Initializing model trainer for {spp_loc}")
        trainer = Exp1ModelTrainer(
            dataset=study_dataset,
            modeling_dir=modeling_dir,
            settlement_point=spp_loc,
            random_state=torch_seed,  # Use same seed as workflow
        )

        # Run complete experiment
        # Complete experiment runs all model types for all hours for this dataset (location + location_type)
        all_results = trainer.run_experiment(
            model_types=model_types,
            use_synthetic_data=False,  # Changed to False to test real data performance
            use_dart_features=False,  # 🔥 EXPERIMENT: Exclude DART lag/rolling features
        )

        print(f"** Completed modeling for {spp_loc}")
        print("-" * 60)

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()
