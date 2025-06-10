"""Script for fetching raw ERCOT data from public API endpoints."""

import os
import time
from datetime import datetime
from datetime import timedelta

from src.data.ercot.clients import DAMSettlementPointPricesClient
from src.data.ercot.clients import DAMSystemLambdaClient
from src.data.ercot.clients import LoadForecastClient
from src.data.ercot.clients import RTSettlementPointPricesClient
from src.data.ercot.clients import SolarGenerationClient
from src.data.ercot.clients import WindGenerationClient


def main():
    """Main function for fetching ERCOT data."""

    script_name = "00-kag-get-ercot-raw-data"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")

    # Set start date
    end_date = datetime.strptime("2024-03-11", "%Y-%m-%d")
    print(type(end_date))
    # Set a starting date as n days preceding start_date
    n_days = 2
    start_date = end_date - timedelta(days=n_days)
    print(type(start_date))

    # Directory setup
    root_dir = "/Users/kag/Documents/Projects"
    project_dir = os.path.join(root_dir, "ercot_dart")
    data_dir = os.path.join(project_dir, "data/raw")
    output_dir = os.path.join(data_dir, end_date.strftime("%Y-%m-%d"))
    os.makedirs(output_dir, exist_ok=True)

    # Set up database path
    db_path = os.path.join(output_dir, "ercot_data.db")

    # Data collection flags - set to True/False to control what data to fetch
    fetch_load_data = True
    fetch_solar_data = True
    fetch_wind_data = True
    fetch_dam_spp_data = True
    fetch_dam_lambda_data = True
    fetch_rt_spp_data = True

    # Specify the hour ending we want (6:00 AM drop) for hourly reports
    hour_ending = "6:00"

    #
    # Fetch real-time data
    #
    if fetch_rt_spp_data:
        print("\n ** Fetching Real-Time Settlement Point Prices...")
        rt_spp_client = RTSettlementPointPricesClient(
            output_dir=output_dir, db_path=db_path
        )
        rt_spp_df = rt_spp_client.get_rt_spp_data(
            delivery_date_from=start_date,
            delivery_date_to=end_date,
        )
        print(f"RT Settlement Point Price records: {len(rt_spp_df)}")

    #
    # Fetch data from daily reports (e.g., DAM)
    #

    if fetch_dam_lambda_data:
        print("\n ** Fetching DAM System Lambda...")
        dam_lambda_client = DAMSystemLambdaClient(
            output_dir=output_dir, db_path=db_path
        )
        dam_lambda_df = dam_lambda_client.get_dam_lambda_data(
            delivery_date_from=start_date,
            delivery_date_to=end_date,
        )
        print(f"DAM System Lambda records: {len(dam_lambda_df)}")

    # Specify the delivery date range for DAM data
    if fetch_dam_spp_data:
        print("\n ** Fetching DAM Settlement Point Prices...")
        dam_spp_client = DAMSettlementPointPricesClient(
            output_dir=output_dir, db_path=db_path
        )
        dam_spp_df = dam_spp_client.get_dam_spp_data(
            delivery_date_from=start_date,
            delivery_date_to=end_date,
        )
        print(f"DAM Settlement Point Price records: {len(dam_spp_df)}")

    #
    # Fetch data for a particular hour from reports generated hourly
    #
    if fetch_load_data:
        print("\n ** Fetching load forecast data...")
        load_client = LoadForecastClient(output_dir=output_dir, db_path=db_path)
        load_df = load_client.get_load_forecast_data(
            posted_datetime_from=start_date,
            posted_datetime_to=end_date,
            posted_hour_ending=hour_ending,
        )
        print(f"Load forecast records: {len(load_df)}")

    if fetch_solar_data:
        print("\n ** Fetching solar generation data...")
        solar_client = SolarGenerationClient(output_dir=output_dir, db_path=db_path)
        solar_df = solar_client.get_solar_generation_data(
            posted_datetime_from=start_date,
            posted_datetime_to=end_date,
            posted_hour_ending=hour_ending,
        )
        print(f"Solar generation records: {len(solar_df)}")

    if fetch_wind_data:
        print("\n ** Fetching wind generation data...")
        wind_client = WindGenerationClient(output_dir=output_dir, db_path=db_path)
        wind_df = wind_client.get_wind_generation_data(
            posted_datetime_from=start_date,
            posted_datetime_to=end_date,
            posted_hour_ending=hour_ending,
        )
        print(f"Wind generation records: {len(wind_df)}")

    print("\nData collection complete!")
    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()
