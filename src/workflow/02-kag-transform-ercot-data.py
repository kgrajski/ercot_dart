"""Script for transforming raw ERCOT data into clean format."""

import os
import time
from datetime import datetime
from etl.ercot import (
    DAMSettlementPointPricesETL,
    DAMSystemLambdaETL,
    LoadForecastETL,
    WindGenerationETL,
    SolarGenerationETL,
    RTSettlementPointPricesETL,
)


def main():
    """Main function for transforming ERCOT data."""
    
    script_name = "02-kag-transform-ercot-data"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects"
    project_dir = os.path.join(root_dir, "ercot_dart")

    # Identify the input directory by date
    raw_data_date = datetime.now().strftime('%Y-%m-%d')
    raw_data_date = "2025-05-31"
    data_dir = os.path.join(project_dir, "data/raw", raw_data_date)

    # Set up output directory
    output_dir = os.path.join(project_dir, "data/processed", raw_data_date)
    os.makedirs(output_dir, exist_ok=True)

    # Set up database path
    db_path = os.path.join(output_dir, "ercot_data.db")

    # Transform Load Forecast
    print("\nTransforming Load Forecast...")
    load_etl = LoadForecastETL(data_dir=data_dir, output_dir=output_dir, db_path=db_path)
    load_etl.transform()

    # Transform Wind Generation
    print("\nTransforming Wind Generation...")
    wind_etl = WindGenerationETL(data_dir=data_dir, output_dir=output_dir, db_path=db_path)
    wind_etl.transform()

    # Transform Solar Generation
    print("\nTransforming Solar Generation...")
    solar_etl = SolarGenerationETL(data_dir=data_dir, output_dir=output_dir, db_path=db_path)
    solar_etl.transform()

    # Transform DAM Settlement Point Prices
    print("\nTransforming DAM Settlement Point Prices...")
    dam_spp_etl = DAMSettlementPointPricesETL(data_dir=data_dir, output_dir=output_dir, db_path=db_path)
    dam_spp_etl.transform()

    # Transform DAM System Lambda
    print("\nTransforming DAM System Lambda...")
    dam_lambda_etl = DAMSystemLambdaETL(data_dir=data_dir, output_dir=output_dir, db_path=db_path)
    dam_lambda_etl.transform()
    
    # Transform RT Settlement Point Prices
    print("\nTransforming RT Settlement Point Prices...")
    rt_spp_etl = RTSettlementPointPricesETL(data_dir=data_dir, output_dir=output_dir, db_path=db_path)
    rt_spp_etl.transform()

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main() 