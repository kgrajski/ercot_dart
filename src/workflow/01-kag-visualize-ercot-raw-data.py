"""Script for visualizing raw ERCOT data obtained from public API endpoints."""

import os
import time
from datetime import datetime
from visualization.ercot.clients.load import LoadForecastViz
from visualization.ercot.clients.solar import SolarGenerationViz
from visualization.ercot.clients.wind import WindGenerationViz
from visualization.ercot.clients.dam_lambda import DAMSystemLambdaViz
from visualization.ercot.clients.dam_spp import DAMSettlementPointPricesViz
from visualization.ercot.clients.rt_spp import RTSettlementPointPricesViz

def main():
    """Main function for visualizing raw ERCOT data."""

    script_name = "01-kag-visualize-ercot-raw-data"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects"
    project_dir = os.path.join(root_dir, "ercot_dart")

    # Identify the input directory by date
    raw_data_date = datetime.now().strftime('%Y-%m-%d')
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

    # Create wind forecast visualizer and generate plots
    print("\nGenerating wind forecast visualizations...")
    wind_viz = WindGenerationViz(data_dir=data_dir, output_dir=output_dir)
    wind_viz.generate_plots()

    # Create DAM System Lambda visualizer and generate plots
    print("\nGenerating DAM System Lambda visualizations...")
    dam_lambda_viz = DAMSystemLambdaViz(data_dir=data_dir, output_dir=output_dir)
    dam_lambda_viz.generate_plots()

    # Create DAM Settlement Point Prices visualizer and generate plots
    print("\nGenerating DAM Settlement Point Prices visualizations...")
    dam_spp_viz = DAMSettlementPointPricesViz(data_dir=data_dir, output_dir=output_dir)
    dam_spp_viz.generate_plots()
    
    # Create RT Settlement Point Prices visualizer and generate plots
    print("\nGenerating RT Settlement Point Prices visualizations...")
    rt_spp_viz = RTSettlementPointPricesViz(data_dir=data_dir, output_dir=output_dir)
    rt_spp_viz.generate_plots()

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()