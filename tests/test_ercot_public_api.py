import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.ERCOT_Data import ERCOT_Data

def test_load_forecast():
    """Test the Seven-Day Load Forecast endpoint with posted datetime parameters."""
    
    # Initialize ERCOT_Data with output directory
    output_dir = "output/load_forecast"
    ercot = ERCOT_Data(output_dir)
    
    # Set up test parameters
    # We want the 6:00 drop for each day
    posted_datetime_from = "2024-01-01T06:00:00"
    posted_datetime_to = "2024-01-02T06:00:00"
    
    try:
        # Get load forecast data
        print(f"\nFetching load forecast data from {posted_datetime_from} to {posted_datetime_to}")
        df = ercot.get_hourly_load_forecast_by_weather_zone(
            posted_datetime_from=posted_datetime_from,
            posted_datetime_to=posted_datetime_to
        )
        
        # Basic validation
        assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        
        # Print summary
        print("\nData Summary:")
        print(f"Total records: {len(df)}")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
        
        # Print first few rows
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise

if __name__ == "__main__":
    test_load_forecast() 