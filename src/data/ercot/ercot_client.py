"""Base client module for ERCOT API endpoints."""

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
    """Base class for ERCOT API clients."""
    
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
    
    def print_pagination_info(self, total_records: int, records_per_page: int, total_pages: int) -> None:
        """Print information about pagination and estimated download time.
        
        Args:
            total_records (int): Total number of records available
            records_per_page (int): Number of records per page
            total_pages (int): Total number of pages
        """
        print(f"\nPagination Info:")
        print(f"Total Records: {total_records}")
        print(f"Records per Page: {records_per_page}")
        print(f"Total Pages: {total_pages}")
        print(f"Rate Limit: {self.api.RATE_LIMIT} requests per minute")
        print(f"Minimum interval between requests: {self.api.MIN_REQUEST_INTERVAL:.2f} seconds")
        
        if total_pages and total_pages > 1:
            time_estimate = self.estimate_download_time(total_pages)
            print(f"Estimated download time: {time_estimate}")
        print()
    
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

    def get_data(self, endpoint_path: str, endpoint_key: str, params: Dict) -> pd.DataFrame:
        """Method to fetch data for a given endpoint for a given date range.
        
        Args:
            endpoint_path (str): Path component of the endpoint URL
            endpoint_key (str): Key identifying the endpoint (for CSV naming)
            params (dict): Query parameters including date range and timing parameters.
                         Can use either postedDatetimeFrom/To or deliveryDateFrom/To
                         for date range iteration.
            
        Returns:
            pandas.DataFrame: DataFrame containing the requested data
            
        Raises:
            ValueError: If required params are missing
            Exception: If API request fails
        """
        # Extract and validate date range - try posted dates first, then delivery dates
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
        with tqdm(total=num_days, desc="Processing days", unit="day") as day_pbar:
            # Process each day
            for current_date in date_range:
                formatted_date = current_date.strftime('%Y-%m-%d')
                
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
                
                # Print pagination information for this day
                self.print_pagination_info(total_records, records_per_page, total_pages)
                
                # Extract data from first page
                data = response_json.get('data', [])
                if data:
                    all_data.extend(data)
                
                # Handle pagination if needed
                if total_pages and total_pages > 1:
                    with tqdm(total=total_pages-1, desc=f"Fetching pages for {formatted_date}", 
                            unit="page", leave=False) as page_pbar:
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
                day_pbar.set_postfix({'date': formatted_date})
        
        # Convert all collected data to DataFrame
        tqdm.write("\nConverting data to DataFrame...")
        df = self.processor.json_to_df({
            "_meta": response_json["_meta"], 
            "report": response_json.get("report", {}),
            "fields": response_json["fields"],
            "data": all_data
        })
        
        # Save to CSV if output directory is set
        csv_file = None
        if self.processor.output_dir:
            tqdm.write("Saving to CSV...")
            csv_file = self.processor.save_to_csv(df, endpoint_key, params)
        
        # Save to database
        if self.db_processor:
            tqdm.write("Saving to database...")
            self.save_to_database(df, endpoint_key)
        
        # Verify and report on the data
        self.processor.verify_data(df, endpoint_key, csv_file)
            
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