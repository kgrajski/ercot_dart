"""Script for fetching raw ERCOT data from public API endpoints."""

from datetime import datetime, timedelta
from ercot.clients.load import LoadForecastClient
from ercot.clients.solar import SolarGenerationClient
from ercot.clients.wind import WindGenerationClient
from ercot.clients.dam_spp import DAMSettlementPointPricesClient
from ercot.clients.dam_lambda import DAMSystemLambdaClient


def main():
    """Main function for fetching ERCOT data."""
    # Set up output directory for CSV files
    output_dir = "/Users/kag/Documents/Projects/ercot_dart/data/raw"
    
    # Create clients
    load_client = LoadForecastClient(output_dir=output_dir)
    solar_client = SolarGenerationClient(output_dir=output_dir)
    wind_client = WindGenerationClient(output_dir=output_dir)
    dam_spp_client = DAMSettlementPointPricesClient(output_dir=output_dir)
    dam_lambda_client = DAMSystemLambdaClient(output_dir=output_dir)
    
    # Get yesterday and today's dates
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Specify the hour ending we want (6:00 AM drop)
    hour_ending = "6:00"
    
    print("Fetching load forecast data...")
    load_df = load_client.get_load_forecast_data(
        posted_datetime_from=today,
        posted_datetime_to=today,
        posted_hour_ending=hour_ending
    )
    
    print("\nFetching solar generation data...")
    solar_df = solar_client.get_solar_generation_data(
        posted_datetime_from=today,
        posted_datetime_to=today,
        posted_hour_ending=hour_ending
    )
    
    print("\nFetching wind generation data...")
    wind_df = wind_client.get_wind_generation_data(
        posted_datetime_from=today,
        posted_datetime_to=today,
        posted_hour_ending=hour_ending
    )
    
    print("\nFetching DAM Settlement Point Prices...")
    dam_spp_df = dam_spp_client.get_dam_spp_data(
        delivery_date_from=today,
        delivery_date_to=today,
    )
    
    print("\nFetching DAM System Lambda...")
    dam_lambda_df = dam_lambda_client.get_dam_lambda_data(
        delivery_date_from=today,
        delivery_date_to=today,
    )
    
    print("\nData collection complete!")
    print(f"Load forecast records: {len(load_df)}")
    print(f"Solar generation records: {len(solar_df)}")
    print(f"Wind generation records: {len(wind_df)}")
    print(f"DAM Settlement Point Price records: {len(dam_spp_df)}")
    print(f"DAM System Lambda records: {len(dam_lambda_df)}")


if __name__ == "__main__":
    main() 