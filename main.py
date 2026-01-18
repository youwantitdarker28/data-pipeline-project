from scripts.extract import extract_from_api
from scripts.transform import DataTransformer, load_data, save_data
from scripts.load import DataLoader

API_URL = 'https://jsonplaceholder.typicode.com/users'
EXTRACTED_FILE = 'data/extracted_users.csv'
FINAL_OUTPUT = 'data/final_output.csv'
DB_PATH = 'data/pipeline.db'
TABLE_NAME = 'users'

print('=== ETL Pipeline Started ===')

print(f'\nStep 1: Extracting data from {API_URL}')
extract_from_api(api_url=API_URL, output_path=EXTRACTED_FILE)

print(f'\nStep 2: Loading extracted data from {EXTRACTED_FILE}')
df = load_data(EXTRACTED_FILE)
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

print(f'\nStep 7: Saving transformed data to {FINAL_OUTPUT}')
cleaned_df = transformer.get_dataframe()
save_data(cleaned_df, FINAL_OUTPUT)

print(f'\nStep 8: Loading data to SQLite database')
with DataLoader(DB_PATH) as loader:
    loader.load_to_sqlite(cleaned_df, TABLE_NAME)
    
    tables = loader.list_tables()
    print(f"  Tables in database: {', '.join(tables['name'].tolist())}")
    
    row_count = loader.get_row_count(TABLE_NAME)
    print(f"  Rows in '{TABLE_NAME}' table: {row_count}")

print(f'\n=== ETL Pipeline Completed Successfully ===')
print(f'CSV output: {FINAL_OUTPUT}')
print(f'SQLite database: {DB_PATH}')
print(f'Rows: {len(cleaned_df)}, Columns: {len(cleaned_df.columns)}')
