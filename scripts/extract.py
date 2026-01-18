import requests
import pandas as pd
import json
import logging
from typing import Optional, Dict, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.data = None
        self.response = None

    def fetch_data(self, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> 'DataExtractor':
        """
        Fetch data from the API.
        
        Args:
            params: Query parameters for the request
            headers: Headers for the request
        
        Returns:
            DataExtractor: self for method chaining
        """
        try:
            logger.info(f"Fetching data from {self.api_url}")
            self.response = requests.get(self.api_url, params=params, headers=headers, timeout=30)
            self.response.raise_for_status()
            
            content_type = self.response.headers.get('content-type', '')
            if 'application/json' in content_type:
                self.data = self.response.json()
                logger.info(f"Successfully fetched {len(self.data) if isinstance(self.data, list) else 1} records")
            else:
                raise ValueError(f"Unexpected content type: {content_type}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        
        return self

    def flatten_data(self) -> 'DataExtractor':
        """
        Flatten nested JSON structures.
        
        Returns:
            DataExtractor: self for method chaining
        """
        if not self.data:
            raise ValueError("No data to flatten. Call fetch_data() first.")
        
        def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                elif isinstance(v, list):
                    items[new_key] = json.dumps(v)
                else:
                    items[new_key] = v
            return items
        
        if isinstance(self.data, list):
            self.data = [flatten_dict(item) for item in self.data]
        elif isinstance(self.data, dict):
            self.data = [flatten_dict(self.data)]
        
        logger.info("Data flattened successfully")
        return self

    def save_to_csv(self, output_path: str, index: bool = False) -> None:
        """
        Save fetched data to a CSV file.
        
        Args:
            output_path: Path where the CSV file will be saved
            index: Whether to include row numbers in the CSV
        """
        if not self.data:
            raise ValueError("No data to save. Call fetch_data() first.")
        
        try:
            df = pd.DataFrame(self.data)
            df.to_csv(output_path, index=index)
            logger.info(f"Data saved to {output_path}")
            logger.info(f"Saved {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def save_to_json(self, output_path: str, orient: str = 'records', indent: int = 2) -> None:
        """
        Save fetched data to a JSON file.
        
        Args:
            output_path: Path where the JSON file will be saved
            orient: Format of JSON output
            indent: Number of spaces for indentation
        """
        if not self.data:
            raise ValueError("No data to save. Call fetch_data() first.")
        
        try:
            df = pd.DataFrame(self.data)
            df.to_json(output_path, orient=orient, indent=indent)
            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def get_data(self) -> Any:
        """
        Get the fetched data.
        
        Returns:
            The fetched data (list or dict)
        """
        return self.data

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the fetched data as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The fetched data as a DataFrame
        """
        if not self.data:
            raise ValueError("No data available. Call fetch_data() first.")
        return pd.DataFrame(self.data)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the fetched data.
        
        Returns:
            Dict: Summary of the data
        """
        if not self.data:
            return {}
        
        df = self.get_dataframe()
        return {
            'url': self.api_url,
            'status_code': self.response.status_code if self.response else None,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict()
        }


def extract_from_api(api_url: str, output_path: str, flatten: bool = True, 
                     params: Optional[Dict[str, Any]] = None, 
                     headers: Optional[Dict[str, str]] = None) -> None:
    """
    Convenience function to extract data from API and save to CSV.
    
    Args:
        api_url: URL of the API endpoint
        output_path: Path where the CSV file will be saved
        flatten: Whether to flatten nested JSON structures
        params: Query parameters for the request
        headers: Headers for the request
    """
    extractor = DataExtractor(api_url)
    extractor.fetch_data(params=params, headers=headers)
    
    if flatten:
        extractor.flatten_data()
    
    extractor.save_to_csv(output_path)
    
    summary = extractor.get_summary()
    print("\n=== Extraction Summary ===")
    for key, value in summary.items():
        if key != 'columns':
            print(f"{key.replace('_', ' ').title()}: {value}")
    print(f"Columns: {', '.join(summary['columns'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract data from a public API and save to CSV')
    parser.add_argument('--url', type=str, default='https://jsonplaceholder.typicode.com/users',
                        help='API URL to fetch data from (default: JSONPlaceholder users)')
    parser.add_argument('--output', type=str, default='data/extracted_data.csv',
                        help='Output CSV file path (default: data/extracted_data.csv)')
    parser.add_argument('--no-flatten', action='store_true',
                        help='Do not flatten nested JSON structures')
    parser.add_argument('--resource', type=str, 
                        choices=['users', 'posts', 'comments', 'albums', 'photos', 'todos'],
                        help='JSONPlaceholder resource to fetch (overrides --url)')
    
    args = parser.parse_args()
    
    if args.resource:
        args.url = f'https://jsonplaceholder.typicode.com/{args.resource}'
    
    print(f"Extracting data from: {args.url}")
    print(f"Saving to: {args.output}")
    
    extract_from_api(
        api_url=args.url,
        output_path=args.output,
        flatten=not args.no_flatten
    )
