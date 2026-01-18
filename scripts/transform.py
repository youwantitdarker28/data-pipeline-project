import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.transformations = []

    def clean_null_values(self, strategy: str = 'drop', fill_value: Optional[Union[str, int, float, Dict[str, Union[str, int, float]]]] = None, 
                          threshold: Optional[float] = None) -> 'DataTransformer':
        """
        Clean null values from the dataframe.
        
        Args:
            strategy: 'drop', 'fill', or 'drop_columns'
                - 'drop': Drop rows with any null values
                - 'fill': Fill null values with specified value
                - 'drop_columns': Drop columns with null values above threshold
            fill_value: Value to use for filling (required for 'fill' strategy)
                Can be a single value or a dict mapping column names to values
            threshold: For 'drop_columns' strategy, minimum percentage of non-null values required
                (e.g., 0.7 means keep columns with at least 70% non-null values)
        
        Returns:
            DataTransformer: self for method chaining
        """
        initial_rows = len(self.df)
        initial_cols = len(self.df.columns)
        
        if strategy == 'drop':
            self.df.dropna(inplace=True)
            dropped_rows = initial_rows - len(self.df)
            self.transformations.append(f"Dropped {dropped_rows} rows with null values")
            logger.info(f"Dropped {dropped_rows} rows with null values")
            
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value must be provided for 'fill' strategy")
            
            if isinstance(fill_value, dict):
                for col, val in fill_value.items():
                    if col in self.df.columns:
                        null_count = self.df[col].isnull().sum()
                        self.df[col].fillna(val, inplace=True)
                        logger.info(f"Filled {null_count} null values in column '{col}' with {val}")
                self.transformations.append(f"Filled null values in specified columns")
            else:
                null_count = self.df.isnull().sum().sum()
                self.df.fillna(fill_value, inplace=True)
                self.transformations.append(f"Filled {null_count} null values with {fill_value}")
                logger.info(f"Filled {null_count} null values with {fill_value}")
                
        elif strategy == 'drop_columns':
            if threshold is None:
                threshold = 0.5
            
            initial_cols = len(self.df.columns)
            threshold_count = int(threshold * len(self.df))
            self.df.dropna(thresh=threshold_count, axis=1, inplace=True)
            dropped_cols = initial_cols - len(self.df.columns)
            self.transformations.append(f"Dropped {dropped_cols} columns with null values above {threshold*100:.0f}%")
            logger.info(f"Dropped {dropped_cols} columns with null values above {threshold*100:.0f}%")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'drop', 'fill', or 'drop_columns'")
        
        return self

    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataTransformer':
        """
        Remove duplicate rows from the dataframe.
        
        Args:
            subset: List of column names to consider for duplicates
            keep: 'first', 'last', or False
        
        Returns:
            DataTransformer: self for method chaining
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        dropped_rows = initial_rows - len(self.df)
        self.transformations.append(f"Removed {dropped_rows} duplicate rows")
        logger.info(f"Removed {dropped_rows} duplicate rows")
        return self

    def standardize_text(self, columns: List[str], case: str = 'lower', strip_whitespace: bool = True) -> 'DataTransformer':
        """
        Standardize text columns.
        
        Args:
            columns: List of column names to standardize
            case: 'lower', 'upper', 'title', or None
            strip_whitespace: Whether to strip leading/trailing whitespace
        
        Returns:
            DataTransformer: self for method chaining
        """
        for col in columns:
            if col in self.df.columns and self.df[col].dtype == 'object':
                if strip_whitespace:
                    self.df[col] = self.df[col].str.strip()
                if case == 'lower':
                    self.df[col] = self.df[col].str.lower()
                elif case == 'upper':
                    self.df[col] = self.df[col].str.upper()
                elif case == 'title':
                    self.df[col] = self.df[col].str.title()
                self.transformations.append(f"Standardized text in column '{col}'")
                logger.info(f"Standardized text in column '{col}'")
        return self

    def get_summary(self) -> Dict:
        """
        Get a summary of the transformation.
        
        Returns:
            Dict: Summary of transformations applied
        """
        null_counts = self.df.isnull().sum()
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'null_values_remaining': null_counts.sum(),
            'null_values_by_column': null_counts[null_counts > 0].to_dict(),
            'transformations_applied': self.transformations
        }

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the transformed dataframe.
        
        Returns:
            pd.DataFrame: The transformed dataframe
        """
        return self.df.copy()


def load_data(source: str, file_type: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from various sources.
    
    Args:
        source: File path or data source
        file_type: Optional file type ('csv', 'json', 'excel'). If None, auto-detected from extension.
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if file_type is None:
        if source.endswith('.csv'):
            file_type = 'csv'
        elif source.endswith('.json'):
            file_type = 'json'
        elif source.endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        else:
            raise ValueError("Cannot auto-detect file type. Please specify file_type parameter.")
    
    logger.info(f"Loading data from {source} as {file_type}")
    
    if file_type == 'csv':
        return pd.read_csv(source)
    elif file_type == 'json':
        return pd.read_json(source)
    elif file_type == 'excel':
        return pd.read_excel(source)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def save_data(df: pd.DataFrame, destination: str, file_type: Optional[str] = None) -> None:
    """
    Save dataframe to various destinations.
    
    Args:
        df: Dataframe to save
        destination: File path or data destination
        file_type: Optional file type ('csv', 'json', 'excel'). If None, auto-detected from extension.
    """
    if file_type is None:
        if destination.endswith('.csv'):
            file_type = 'csv'
        elif destination.endswith('.json'):
            file_type = 'json'
        elif destination.endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        else:
            raise ValueError("Cannot auto-detect file type. Please specify file_type parameter.")
    
    logger.info(f"Saving data to {destination} as {file_type}")
    
    if file_type == 'csv':
        df.to_csv(destination, index=False)
    elif file_type == 'json':
        df.to_json(destination, orient='records', indent=2)
    elif file_type == 'excel':
        df.to_excel(destination, index=False)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform data by cleaning null values')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--strategy', type=str, default='drop', 
                        choices=['drop', 'fill', 'drop_columns'], help='Null handling strategy')
    parser.add_argument('--fill-value', type=str, help='Value to use for filling (for fill strategy)')
    parser.add_argument('--threshold', type=float, help='Threshold for drop_columns strategy (0-1)')
    
    args = parser.parse_args()
    
    df = load_data(args.input)
    transformer = DataTransformer(df)
    
    if args.strategy == 'fill' and args.fill_value:
        try:
            fill_value = json.loads(args.fill_value)
        except json.JSONDecodeError:
            fill_value = args.fill_value
    else:
        fill_value = None
    
    if args.strategy == 'drop_columns' and args.threshold:
        transformer.clean_null_values(strategy=args.strategy, threshold=args.threshold)
    elif args.strategy == 'fill':
        transformer.clean_null_values(strategy=args.strategy, fill_value=fill_value)
    else:
        transformer.clean_null_values(strategy=args.strategy)
    
    summary = transformer.get_summary()
    print("\n=== Transformation Summary ===")
    for key, value in summary.items():
        if key != 'null_values_by_column':
            print(f"{key.replace('_', ' ').title()}: {value}")
    print("\n=== Null Values Remaining by Column ===")
    for col, count in summary['null_values_by_column'].items():
        print(f"  {col}: {count}")
    
    save_data(transformer.get_dataframe(), args.output)
    print(f"\nTransformed data saved to {args.output}")
