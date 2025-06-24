"""Experiment 2 modeling workflow for ERCOT DART prediction.

This workflow performs machine learning modeling experiments on the finalized
exp2 study datasets. It processes each settlement point (location/location_type)
independently and trains models for DART price prediction.

Key features:
- Loads finalized datasets using DartSltExp2Dataset
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

from src.models.ercot.exp2.DartSLTExp2Dataset import DartSltExp2Dataset
from src.models.ercot.exp2.model_trainer import Exp2ModelTrainer


def main():
    """
    Main function to set up the experiment and run it.
    """
    script_name = "04-kag-exp2-ercot-modeling"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")

    # Set the random seed for reproducibility
    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    # Experiment name
    exp_name = "exp2"
    bootstrap_iterations = 10
    model_types = [
        "xgboost_classification",
        # TODO: Add when implemented
        # "rf_classification"
    ]
    processed_data_date = "2025-06-04"

    # **NEW: Progressive validation option**
    run_progressive_validation = True  # Set to True for progressive validation
    num_weeks = (
        None  # Number of weeks for progressive validation (testing with 2 weeks)
    )
    verbose_training = False  # Set to True for detailed training output per hour

    print(
        f"\n** Experiment name: {exp_name} on processed dataset created: {processed_data_date}"
    )
    if run_progressive_validation:
        print("** Validation mode: PROGRESSIVE (weekly rolling)")
    else:
        print("** Validation mode: YEARLY (2024 train, 2025 validate)")

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects/"
    project_dir = os.path.join(root_dir, "ercot_dart")
    data_dir = os.path.join(project_dir, "data/studies", exp_name, processed_data_date)

    # Generate the list of subdirectories in the input directory (in_data_dir)
    # Note that exp2 is a major modeling approach change, but still uses the same data as from exp1.
    # We will manually copy the data from exp1 to exp2 and relabel from exp1 to exp2 final_dataset.
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
        study_dataset = DartSltExp2Dataset(spp_dir=spp_dir, output_dir=modeling_dir)

        # Initialize model trainer
        print(f"** Initializing model trainer for {spp_loc}")
        trainer = Exp2ModelTrainer(
            dataset=study_dataset,
            modeling_dir=modeling_dir,
            settlement_point=spp_loc,
            random_state=torch_seed,  # Use same seed as workflow
        )

        # Run complete experiment
        # Complete experiment runs all model types for all hours for this dataset (location + location_type)
        if run_progressive_validation:
            # Run progressive validation
            all_results = trainer.run_experiment_progressive(
                model_types=model_types,
                bootstrap_iterations=bootstrap_iterations,  # Fast for testing
                num_weeks=num_weeks,  # Pass configured number of weeks
                classification_strategy="sign_only",  # Start with sign-only classification
                use_dart_features=False,  # 🔥 EXPERIMENT: Exclude DART lag/rolling features (NO DART FEATURES)
            )
        else:
            # Run standard yearly validation
            all_results = trainer.run_experiment(
                model_types=model_types,
                bootstrap_iterations=bootstrap_iterations,  # Fast for testing
                classification_strategy="sign_only",  # Start with sign-only classification
                use_dart_features=False,  # 🔥 EXPERIMENT: Exclude DART lag/rolling features (NO DART FEATURES)
            )

        print(f"** Completed modeling for {spp_loc}")
        print("-" * 60)

    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"*** {script_name} - END ***")


if __name__ == "__main__":
    main()
