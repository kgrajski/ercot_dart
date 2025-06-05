"""Base data module for ERCOT API data handling and processing.

This module provides base functionality for:
- Authentication and API access
- Data fetching with pagination
- JSON to DataFrame conversion
- Timestamp handling for both hourly and sub-hourly data
- CSV and database storage

For sub-hourly data (e.g., Real-Time Settlement Point Prices with 15-minute intervals),
the base timestamp handling preserves the interval information in the original data.
The interval-specific adjustments, if needed, are handled by the specific client classes.
"""

from typing import Dict, Optional
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
from .auth import ERCOTAuth
from .api import ERCOTApi
from .processors import ERCOTProcessor
from .database import DatabaseProcessor


class ERCOTBaseClient:
    """Base class for ERCOT API clients.
    
    This class provides core functionality for fetching and processing ERCOT data:
    - Authentication and API access
    - Data pagination handling
    - Basic timestamp conversion (UTC and local)
    - CSV and database storage
    
    For endpoints with sub-hourly data (e.g., RT SPP's 15-minute intervals):
    - Base timestamp handling works at the hourly level
    - Interval information (e.g., deliveryInterval) is preserved in the data
    - Specific interval handling, if needed, is done by client classes
    """
    
    def __init__(self, output_dir: Optional[str] = None, db_path: Optional[str] = None):
        """Initialize client with authentication, API, processor, and database handlers.
        
        Args:
            output_dir (str, optional): Directory path where CSV files will be saved
            db_path (str, optional): Path to SQLite database file
        """
        self.auth = ERCOTAuth()
        self.api = ERCOTApi(self.auth)
        self.processor = ERCOTProcessor(output_dir)
        self.db_processor = DatabaseProcessor(db_path)
    
    def estimate_download_time(self, total_pages: int) -> str:
        """Calculate and format the estimated download time based on number of pages and rate limit.
        
        Args:
            total_pages (int): Total number of pages to download
            
        Returns:
            str: Human readable time estimate (e.g., "2 hours, 15 minutes, 30 seconds")
        """
        if not total_pages or total_pages <= 1:
            return "less than 1 minute"
            
        # Time per page is at least MIN_REQUEST_INTERVAL
        # Add 20% buffer for processing and potential retries
        time_per_page = self.api.MIN_REQUEST_INTERVAL * 1.2
        total_estimated_seconds = time_per_page * (total_pages - 1)  # -1 because we already have first page
        
        # Convert to hours, minutes, seconds
        hours = int(total_estimated_seconds // 3600)
        minutes = int((total_estimated_seconds % 3600) // 60)
        seconds = int(total_estimated_seconds % 60)
        
        # Build time estimate string
        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        
        return ", ".join(parts)
    
    def print_pagination_info(self, total_records: int, records_per_page: int, total_pages: int) -> str:
        """Print information about pagination and estimated download time.
        
        Args:
            total_records (int): Total number of records available
            records_per_page (int): Number of records per page
            total_pages (int): Total number of pages
        """
        info = [
            f"Records: {total_records}",
            f"Pages: {total_pages}",
        ]
        
        if total_pages and total_pages > 1:
            time_estimate = self.estimate_download_time(total_pages)
            info.append(f"Est. time: {time_estimate}")
        
        return " | ".join(info)
    
    def _build_query_params(self, current_date: datetime, params: Dict) -> Dict:
        """Build query parameters for a specific date based on the original params.
        Base implementation handles hourly drops with posted datetime parameters.
        
        Args:
            current_date (datetime): The date to build parameters for
            params (dict): Original parameters including:
                - postedHourEnding: Hour ending for the data drop (e.g., "6:00")
                - hours_before: Number of hours before hour_ending to start the query
            
        Returns:
            dict: Query parameters for the specific date
            
        Note:
            This method can be overridden by subclasses to handle different parameter patterns,
            such as delivery date based queries for DAM endpoints.
        """
        # Extract time window parameters
        hour_ending = params.get('postedHourEnding', "6:00")  # Default to 6 AM if not specified
        hours_before = params.get('hours_before', 1)  # Default to 1 hour if not specified
        
        if hours_before < 1:
            raise ValueError("hours_before must be at least 1")
        
        # Convert hour_ending (e.g., "6:00") to 24-hour format with seconds (e.g., "06:00:00")
        try:
            hour, minute = hour_ending.split(":")
            hour_24 = f"{int(hour):02d}:{minute}:00"
            # Calculate start time based on hours_before parameter
            hour_24_start = f"{int(hour)-hours_before:02d}:{minute}:00"
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid hour_ending format. Expected 'H:mm' (e.g., '6:00'), got '{hour_ending}'") from e
        
        # Handle midnight boundary case
        if hour_24_start.startswith("-"):
            # If we cross midnight boundary, start at 00:00
            hour_24_start = f"00:{minute}:00"
            
        # Format datetime for the specific drop we want
        formatted_date = current_date.strftime('%Y-%m-%d')
        
        # Create API parameters dict without internal parameters
        current_params = {k: v for k, v in params.items() if k not in ['postedHourEnding', 'hours_before']}
        current_params['postedDatetimeFrom'] = f"{formatted_date}T{hour_24_start}"
        current_params['postedDatetimeTo'] = f"{formatted_date}T{hour_24}"
        
        return current_params

    def _save_data(self, df: pd.DataFrame, endpoint_key: str, params: Dict) -> pd.DataFrame:
        """Save data to CSV and database.
        
        Args:
            df: DataFrame to save
            endpoint_key: Key identifying the endpoint (for CSV naming)
            params: Query parameters used to fetch the data
            
        Returns:
            pandas.DataFrame: The input DataFrame (for chaining)
        """
        # Sort by UTC timestamp for consistent ordering before saving
        df = df.sort_values('utc_ts')
        
        # Save to CSV if output directory is set
        if self.processor.output_dir:
            csv_file = self.processor.save_to_csv(df, endpoint_key, params)
            tqdm.write(f"Data saved to: {csv_file}")
        
        # Save to database if configured
        if self.db_processor:
            self.save_to_database(df, endpoint_key)
            
        return df

    def get_data(self, endpoint_path: str, endpoint_key: str, params: Dict, 
                 save_output: bool = True, df: Optional[pd.DataFrame] = None,
                 add_utc: bool = True) -> pd.DataFrame:
        """Method to fetch and/or process data for a given endpoint.
        
        This method handles:
        1. Data fetching with pagination
        2. Basic timestamp conversion (UTC and local) if add_utc=True
        3. Data saving (CSV and database)
        
        For sub-hourly data (e.g., RT SPP):
        - Base timestamp handling works at the hourly level
        - Sub-hourly interval information is preserved in original columns
        - No automatic adjustment of timestamps for intervals
        - Some endpoints (like RT SPP) may need custom column mapping
          before timestamp processing
        
        Args:
            endpoint_path: Path component of the endpoint URL
            endpoint_key: Key identifying the endpoint (for CSV naming)
            params: Query parameters including date range and timing parameters
            save_output: Whether to save the output to CSV/DB. Defaults to True.
                        Set to False when handling intermediate data that will
                        be combined later (e.g., RT SPP settlement points)
            df: Optional pre-fetched DataFrame. If provided, skips API calls.
                Useful for saving combined data from multiple API calls.
            add_utc: Whether to add UTC timestamps. Defaults to True.
                    Set to False if timestamp columns need preprocessing.
            
        Returns:
            pandas.DataFrame: DataFrame containing the requested data with:
                - Original ERCOT columns including any interval information
                - Added UTC timestamps at hourly level (if add_utc=True)
                - Added local timestamps at hourly level (if add_utc=True)
        """
        if df is None:
            # Extract and validate date range
            date_from = params.get('postedDatetimeFrom') or params.get('deliveryDateFrom')
            date_to = params.get('postedDatetimeTo') or params.get('deliveryDateTo')
            
            if not date_from or not date_to:
                raise ValueError("Either postedDatetime or deliveryDate range parameters are required")
                
            start_date = pd.to_datetime(date_from).date()
            end_date = pd.to_datetime(date_to).date()
            
            # Calculate number of days
            date_range = pd.date_range(start_date, end_date, freq='D')
            num_days = len(date_range)
            
            tqdm.write(f"\nFetching data from {start_date} to {end_date}")
            
            # Initialize empty list to store all data
            all_data = []
            
            # Create progress bar for days
            with tqdm(total=num_days, desc="Processing days", unit="day", position=0) as day_pbar:
                # Process each day
                for current_date in date_range:
                    # Build query parameters for this date
                    current_params = self._build_query_params(current_date, params)
                    
                    # Make initial request to get pagination info
                    response = self.api.make_request(endpoint_path, current_params)
                    
                    if response.status_code != 200:
                        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                    
                    # Parse response
                    response_json = response.json()
                    
                    # Extract metadata
                    meta = response_json.get('_meta', {})
                    total_pages = meta.get('totalPages')
                    records_per_page = meta.get('pageSize')
                    total_records = meta.get('totalRecords')
                    
                    # Extract data from first page
                    data = response_json.get("data", [])
                    if data:
                        all_data.extend(data)
                    
                    # Update day progress with pagination info
                    day_pbar.set_postfix_str(f"date={current_date.strftime('%Y-%m-%d')} | " + 
                                           self.print_pagination_info(total_records, records_per_page, total_pages))
                    
                    # Handle pagination if needed
                    if total_pages and total_pages > 1:
                        with tqdm(total=total_pages-1, desc="Pages", unit="page", 
                                position=1, leave=False) as page_pbar:
                            for current_page in range(2, total_pages + 1):
                                # Always wait the minimum interval
                                time.sleep(self.api.MIN_REQUEST_INTERVAL)
                                
                                # Update params with current page
                                current_params['page'] = current_page
                                
                                # Make the API request with retry logic
                                response = self.api.make_request(endpoint_path, current_params)
                                
                                if response.status_code != 200:
                                    raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                                
                                # Parse response and extract data
                                response_json = response.json()
                                data = response_json.get('data', [])
                                if data:
                                    all_data.extend(data)
                                
                                # Update progress bar
                                page_pbar.update(1)
                    
                    # Update day progress
                    day_pbar.update(1)
            
            # Convert all collected data to DataFrame
            tqdm.write("\nProcessing collected data...")
            df = self.processor.json_to_df({
                "_meta": response_json["_meta"], 
                "report": response_json.get("report", {}),
                "fields": response_json["fields"],
                "data": all_data
            })
            
            # Add UTC timestamps if requested
            if add_utc:
                df = self.processor.add_utc_timestamps(df)
        
        # Save if requested (which will also sort by UTC timestamp)
        if save_output:
            df = self._save_data(df, endpoint_key, params)
            
        return df
    
    def save_to_database(self, df: pd.DataFrame, endpoint_key: str):
        """Save data to database with client-specific handling.
        This method can be overridden by subclasses to provide custom
        data preparation before saving to database.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            endpoint_key (str): Key identifying the endpoint
        """
        if self.db_processor:
            self.db_processor.save_to_database(df, endpoint_key) 