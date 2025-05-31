"""Script for fetching raw ERCOT data from public API endpoints."""

from datetime import datetime, timedelta
from data.ercot.clients.load import LoadForecastClient
from data.ercot.clients.solar import SolarGenerationClient
from data.ercot.clients.wind import WindGenerationClient
from data.ercot.clients.dam_spp import DAMSettlementPointPricesClient
from data.ercot.clients.dam_lambda import DAMSystemLambdaClient
from data.ercot.clients.rt_spp import RTSettlementPointPricesClient

import os
import time


def main():
    """Main function for fetching ERCOT data."""

    script_name = "00-kag-get-ercot-raw-data"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")

    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    # Set a starting date as n days preceding today
    n_days = 880
    start_date = (datetime.now() - timedelta(days=n_days)).strftime('%Y-%m-%d')

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects"
    project_dir = os.path.join(root_dir, "ercot_dart")
    data_dir = os.path.join(project_dir, "data/raw")
    output_dir = os.path.join(data_dir, today)
    os.makedirs(output_dir, exist_ok=True)

    # Set up database path
    db_path = os.path.join(output_dir, "ercot_data.db")

    # Create clients with both CSV and database output
    load_client = LoadForecastClient(output_dir=output_dir, db_path=db_path)
    solar_client = SolarGenerationClient(output_dir=output_dir, db_path=db_path)
    wind_client = WindGenerationClient(output_dir=output_dir, db_path=db_path)
    dam_spp_client = DAMSettlementPointPricesClient(output_dir=output_dir, db_path=db_path)
    dam_lambda_client = DAMSystemLambdaClient(output_dir=output_dir, db_path=db_path)
    rt_spp_client = RTSettlementPointPricesClient(output_dir=output_dir, db_path=db_path)

    
    #
    # Fetch data from hourly reports
    #

    # Specify the hour ending we want (6:00 AM drop) for hourly reports
    hour_ending = "6:00"
    
    print("Fetching load forecast data...")
    load_df = load_client.get_load_forecast_data(
        posted_datetime_from=start_date,
        posted_datetime_to=today,
        posted_hour_ending=hour_ending
    )
    
    print("\nFetching solar generation data...")
    solar_df = solar_client.get_solar_generation_data(
        posted_datetime_from=start_date,
        posted_datetime_to=today,
        posted_hour_ending=hour_ending
    )
    
    print("\nFetching wind generation data...")
    wind_df = wind_client.get_wind_generation_data(
        posted_datetime_from=start_date,
        posted_datetime_to=today,
        posted_hour_ending=hour_ending
    )

    #
    # Fetch data from daily reports (e.g., DAM)
    #

    # Specify the delivery date range for DAM data
    print("\nFetching DAM Settlement Point Prices...")
    dam_spp_df = dam_spp_client.get_dam_spp_data(
        delivery_date_from=start_date,
        delivery_date_to=today,
    )
    
    print("\nFetching DAM System Lambda...")
    dam_lambda_df = dam_lambda_client.get_dam_lambda_data(
        delivery_date_from=start_date,
        delivery_date_to=today,
    )

    #
    # Fetch real-time data
    #
    
    print("\nFetching Real-Time Settlement Point Prices...")
    rt_spp_df = rt_spp_client.get_rt_spp_data(
        delivery_date_from=start_date,
        delivery_date_to=today,
    )
    
    print("\nData collection complete!")
    print(f"Load forecast records: {len(load_df)}")
    print(f"Solar generation records: {len(solar_df)}")
    print(f"Wind generation records: {len(wind_df)}")
    print(f"DAM Settlement Point Price records: {len(dam_spp_df)}")
    print(f"DAM System Lambda records: {len(dam_lambda_df)}")
    print(f"RT Settlement Point Price records: {len(rt_spp_df)}")

    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()