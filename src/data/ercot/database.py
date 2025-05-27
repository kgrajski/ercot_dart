"""Database processing module for ERCOT data."""

import os
from typing import Dict, Optional, List
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path


class DatabaseProcessor:
    """Handles database operations for ERCOT data."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database processor.
        
        Args:
            db_path (str, optional): Path to SQLite database file.
                                   If None, will create 'ercot_data.db' in current directory.
        """
        if db_path is None:
            db_path = 'ercot_data.db'
        self.db_path = Path(db_path)
    
    def create_table_for_client(self, client_key: str, df: pd.DataFrame) -> str:
        """Create SQL table schema for a client's data.
        
        Args:
            client_key (str): Identifier for the client (e.g., 'load_forecast')
            df (pd.DataFrame): Sample DataFrame to derive schema from
            
        Returns:
            str: SQL create table statement
        """
        # Map pandas dtypes to SQL types
        dtype_map = {
            'object': 'TEXT',
            'int64': 'INTEGER',
            'float64': 'REAL',
            'datetime64[ns]': 'TIMESTAMP',
            'bool': 'BOOLEAN'
        }
        
        # Generate column definitions
        columns = []
        for col, dtype in df.dtypes.items():
            sql_type = dtype_map.get(str(dtype), 'TEXT')
            columns.append(f"{col} {sql_type}")
            
        # Create table statement
        create_table = f"""
        CREATE TABLE IF NOT EXISTS {client_key} (
            {', '.join(columns)}
        )
        """
        
        return create_table
    
    def deduplicate_data(self, df: pd.DataFrame, client_key: str) -> pd.DataFrame:
        """Basic deduplication of data using pandas drop_duplicates.
        
        Args:
            df (pd.DataFrame): DataFrame to deduplicate
            client_key (str): Identifier for the client (unused for now)
            
        Returns:
            pd.DataFrame: Deduplicated DataFrame
        """
        # Get count before deduplication
        original_count = len(df)
        
        # Deduplicate
        deduped_df = df.drop_duplicates(keep='last')
        
        # Get count after deduplication
        final_count = len(deduped_df)
        
        # Print warning if duplicates were found
        if final_count < original_count:
            duplicate_count = original_count - final_count
            print(f"\nWARNING: Found {duplicate_count} duplicate rows in {client_key} data")
            print(f"Original count: {original_count}, After deduplication: {final_count}")
        
        return deduped_df
    
    def save_to_database(self, df: pd.DataFrame, client_key: str):
        """Save DataFrame to SQLite database.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            client_key (str): Identifier for the client
        """
        # Connect to database
        with sqlite3.connect(self.db_path) as conn:
            # Create table if it doesn't exist
            create_table_sql = self.create_table_for_client(client_key, df)
            conn.execute(create_table_sql)
            
            # Deduplicate and save data
            deduped_df = self.deduplicate_data(df, client_key)
            deduped_df.to_sql(client_key, conn, if_exists='append', index=False) 