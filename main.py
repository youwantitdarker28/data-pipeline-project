import argparse
from pathlib import Path

from scripts.extract import extract_from_api
from scripts.transform import DataTransformer, load_data, save_data
from scripts.load import DataLoader

DEFAULT_API_URL = 'https://jsonplaceholder.typicode.com/users'
DEFAULT_EXTRACTED_FILE = 'data/extracted_users.csv'
DEFAULT_FINAL_OUTPUT = 'data/final_output.csv'
DEFAULT_DB_PATH = 'data/pipeline.db'
DEFAULT_TABLE_NAME = 'users'


def run_pipeline(
    api_url: str,
    extracted_file: str,
    final_output: str,
    db_path: str,
    table_name: str,
) -> None:
    Path(extracted_file).parent.mkdir(parents=True, exist_ok=True)
    Path(final_output).parent.mkdir(parents=True, exist_ok=True)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    print('=== ETL Pipeline Started ===')

    print(f'\nStep 1: Extracting data from {api_url}')
    extract_from_api(api_url=api_url, output_path=extracted_file)

    print(f'\nStep 2: Loading extracted data from {extracted_file}')
    df = load_data(extracted_file)
    print(f'Loaded {len(df)} rows and {len(df.columns)} columns')

    print(f'\nStep 3: Initializing DataTransformer')
    transformer = DataTransformer(df)

    print('\nStep 4: Cleaning null values')
    transformer.clean_null_values(strategy='drop')

    print('\nStep 5: Standardizing text in name and username columns')
    transformer.standardize_text(columns=['name', 'username'], case='title')

    print('\nStep 6: Getting transformation summary')
    summary = transformer.get_summary()
    for key, value in summary.items():
        if key != 'null_values_by_column':
            print(f'  {key}: {value}')

    print(f'\nStep 7: Saving transformed data to {final_output}')
    cleaned_df = transformer.get_dataframe()
    save_data(cleaned_df, final_output)

    print(f'\nStep 8: Loading data to SQLite database')
    with DataLoader(db_path) as loader:
        loader.load_to_sqlite(cleaned_df, table_name)

        tables = loader.list_tables()
        print(f"  Tables in database: {', '.join(tables['name'].tolist())}")

        row_count = loader.get_row_count(table_name)
        print(f"  Rows in '{table_name}' table: {row_count}")

    print(f'\n=== ETL Pipeline Completed Successfully ===')
    print(f'CSV output: {final_output}')
    print(f'SQLite database: {db_path}')
    print(f'Rows: {len(cleaned_df)}, Columns: {len(cleaned_df.columns)}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full ETL pipeline.')
    parser.add_argument('--url', type=str, default=DEFAULT_API_URL, help='API URL to fetch data from')
    parser.add_argument('--extracted', type=str, default=DEFAULT_EXTRACTED_FILE, help='Extracted CSV path')
    parser.add_argument('--output', type=str, default=DEFAULT_FINAL_OUTPUT, help='Final output CSV path')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_PATH, help='SQLite database path')
    parser.add_argument('--table', type=str, default=DEFAULT_TABLE_NAME, help='SQLite table name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        api_url=args.url,
        extracted_file=args.extracted,
        final_output=args.output,
        db_path=args.db,
        table_name=args.table,
    )
