import sqlite3
import pandas as pd
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, db_path: str = 'data/pipeline.db'):
        """
        Initialize the DataLoader with a database path.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.engine = None
        self._ensure_db_directory()

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Database directory ready: {db_dir}")

    def connect(self) -> sqlite3.Connection:
        """
        Establish a connection to the SQLite database.
        
        Returns:
            sqlite3.Connection: Database connection object
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.engine = create_engine(f'sqlite:///{self.db_path}')
            logger.info(f"Connected to database: {self.db_path}")
            return self.connection
        except sqlite3.OperationalError as e:
            if 'disk I/O error' in str(e):
                logger.warning(f"Disk I/O error with {self.db_path}, using temp directory")
                temp_path = os.path.join(tempfile.gettempdir(), 'pipeline.db')
                self.connection = sqlite3.connect(temp_path)
                self.engine = create_engine(f'sqlite:///{temp_path}')
                self.db_path = temp_path
                logger.info(f"Connected to database (fallback): {self.db_path}")
                return self.connection
            else:
                logger.error(f"Error connecting to database: {e}")
                raise
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")

    def load_to_sqlite(self, df: pd.DataFrame, table_name: str, 
                       if_exists: str = 'replace', index: bool = False) -> None:
        """
        Save a pandas DataFrame to a SQLite table.
        
        Args:
            df: Pandas DataFrame to save
            table_name: Name of the table to create/replace
            if_exists: What to do if table exists: 'fail', 'replace', 'append'
            index: Whether to write DataFrame index as a column
        """
        if self.connection is None:
            self.connect()
        
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            logger.info(f"Successfully loaded {len(df)} rows to table '{table_name}'")
        except Exception as e:
            if 'disk I/O error' in str(e):
                logger.warning(f"Disk I/O error, trying temp directory")
                self.close()
                temp_path = os.path.join(tempfile.gettempdir(), 'pipeline.db')
                self.connection = sqlite3.connect(temp_path)
                self.engine = create_engine(f'sqlite:///{temp_path}')
                self.db_path = temp_path
                df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
                logger.info(f"Successfully loaded {len(df)} rows to table '{table_name}' (using temp)")
            else:
                logger.error(f"Error loading data to SQLite: {e}")
                raise

    def read_from_sqlite(self, query: str) -> pd.DataFrame:
        """
        Read data from SQLite using a SQL query.
        
        Args:
            query: SQL query to execute
        
        Returns:
            pd.DataFrame: Query results as a DataFrame
        """
        if self.connection is None:
            self.connect()
        
        try:
            df = pd.read_sql_query(query, self.connection)
            logger.info(f"Successfully read {len(df)} rows from query")
            return df
        except Exception as e:
            logger.error(f"Error reading from SQLite: {e}")
            raise

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            pd.DataFrame: Table schema information
        """
        if self.connection is None:
            self.connect()
        
        query = f"PRAGMA table_info({table_name})"
        return self.read_from_sqlite(query)

    def list_tables(self) -> pd.DataFrame:
        """
        List all tables in the database.
        
        Returns:
            pd.DataFrame: List of tables
        """
        if self.connection is None:
            self.connect()
        
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        return self.read_from_sqlite(query)

    def get_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            int: Number of rows in the table
        """
        if self.connection is None:
            self.connect()
        
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        logger.info(f"Table '{table_name}' has {count} rows")
        return count

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_dataframe_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str, 
                             if_exists: str = 'replace', index: bool = False) -> None:
    """
    Convenience function to load a DataFrame to SQLite.
    
    Args:
        df: Pandas DataFrame to save
        db_path: Path to the SQLite database
        table_name: Name of the table
        if_exists: What to do if table exists: 'fail', 'replace', 'append'
        index: Whether to write DataFrame index as a column
    """
    with DataLoader(db_path) as loader:
        loader.load_to_sqlite(df, table_name, if_exists=if_exists, index=index)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load data from CSV to SQLite')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--table', type=str, required=True, help='Table name in SQLite')
    parser.add_argument('--db', type=str, default='data/pipeline.db', help='SQLite database path')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    
    with DataLoader(args.db) as loader:
        loader.load_to_sqlite(df, args.table)
        
        tables = loader.list_tables()
        print(f"\nTables in database: {', '.join(tables['name'].tolist())}")
        
        count = loader.get_row_count(args.table)
        print(f"Rows in '{args.table}': {count}")
        
        schema = loader.get_table_info(args.table)
        print(f"\nTable schema for '{args.table}':")
        print(schema[['name', 'type']])
