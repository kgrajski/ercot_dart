"""Script for visualizing raw ERCOT data obtained from public API endpoints."""

import os
import time
from ercot.clients.load import LoadForecastViz
from ercot.clients.solar import SolarGenerationViz

def main():
    """Main function for visualizing raw ERCOT data."""

    script_name = "01-kag-visualize-ercot-raw-data"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects"
    project_dir = os.path.join(root_dir, "ercot_dart")

    # Identify the input directory by date
    raw_data_date = "2025-05-27"
    data_dir = os.path.join(project_dir, "data/raw", raw_data_date)

    # Set up output directory
    output_dir = os.path.join(project_dir, "data/interim", raw_data_date)
    os.makedirs(output_dir, exist_ok=True)

    # Create load forecast visualizer and generate plots
    print("\nGenerating load forecast visualizations...")
    load_viz = LoadForecastViz(data_dir=data_dir, output_dir=output_dir)
    load_viz.generate_plots()

    # Create solar forecast visualizer and generate plots
    print("\nGenerating solar forecast visualizations...")
    solar_viz = SolarGenerationViz(data_dir=data_dir, output_dir=output_dir)
    solar_viz.generate_plots()

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()