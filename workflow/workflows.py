from auth.auth import GoogleAuth
from data import sheet_data
import pandas as pd
import asyncio
import os
import pickle
from data.vendors.avesapi import AvesAPIClient

def get_keyword_portfolio_as_df(spreadsheet_id: str, sheet_name: str, header_row: int = 5, data_start_row: int = 6) -> pd.DataFrame:
    """
    Get keyword portfolio data from Google Sheets with improved error handling.
    
    Args:
        spreadsheet_id: ID of the Google Sheet
        sheet_name: Name of the sheet to read
        header_row: Row index of the headers (0-based)
        data_start_row: Row index where data starts (0-based)
        
    Returns:
        DataFrame containing keyword portfolio data
    """
    try:
        auth = GoogleAuth(credentials_path='client_secret.json')
        client = sheet_data.GSheetsClient(auth)
        
        # Get all sheets as dataframes
        sheets_dict = client.get_all_as_dataframes(
            spreadsheet_id, 
            sheet_name,
            header_row=header_row,
            data_start_row=data_start_row
        )
        
        # Check if the requested sheet exists
        if sheet_name not in sheets_dict:
            print(f"Error: Sheet '{sheet_name}' not found in spreadsheet")
            return pd.DataFrame()
        
        df = sheets_dict[sheet_name]
        
        # Validate the dataframe
        if df.empty:
            print(f"Warning: No data found in sheet '{sheet_name}'")
            return df
            
        # Check for required columns
        required_columns = ['Keyword']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            
        # Format URLs in 'Current Landing Page' column if it exists
        if 'Current Landing Page' in df.columns:
            df['Current Landing Page'] = df['Current Landing Page'].apply(
                lambda x: (f'https://{x.strip()}' 
                        if x and isinstance(x, str) and x.strip() and not x.strip().startswith(('http://', 'https://')) 
                        else x)
            )
        
        # Remove rows with empty keywords
        if 'Keyword' in df.columns:
            initial_rows = len(df)
            df = df.dropna(subset=['Keyword'])
            rows_dropped = initial_rows - len(df)
            
            if rows_dropped > 0:
                print(f"Removed {rows_dropped} rows with empty keywords")
        
        print(f"Successfully loaded {len(df)} keyword rows from Google Sheets")
        return df
        
    except Exception as e:
        print(f"Error loading keyword portfolio: {str(e)}")
        return pd.DataFrame()

async def get_avesapi_data(api_key: str, queries: list[str], num_results: int = 3):
    client = AvesAPIClient(api_key)
    batch_results = await client.batch_search(queries=queries, num_results = num_results)
    
    for query, result in batch_results.items():
        if "error" in result:
            print(f"Query '{query}' failed: {result['error']}")
        else:
            print(f"Query '{query}' found {len(result.get('results', []))} results")
            
    return batch_results

async def get_avesapi_data_with_monitoring(api_key: str, queries: list[str], num_results: int = 3, batch_size: int = 20):
    """
    Get data from Aves API with improved monitoring and error handling.
    
    Args:
        api_key: Aves API key
        queries: List of search queries
        num_results: Number of results to return per query
        batch_size: Number of queries to process in each batch
        
    Returns:
        Dictionary of API results keyed by query
    """
    from data.vendors.avesapi import AvesAPIClient, BatchRequestManager
    
    client = AvesAPIClient(
        api_key=api_key,
        max_concurrent_requests=10,
        rate_limit_per_min=60,
        retry_attempts=3
    )
    
    batch_manager = BatchRequestManager(
        client=client,
        chunk_size=batch_size,
        pause_between_chunks=2.0
    )
    
    # Progress tracking callback
    async def progress_callback(completed, total, current_results):
        success_count = sum(1 for result in current_results.values() if 'error' not in result)
        error_count = sum(1 for result in current_results.values() if 'error' in result)
        
        print(f"Progress: {completed}/{total} queries processed")
        print(f"Success: {success_count}, Errors: {error_count}")
    
    print(f"Starting API requests for {len(queries)} queries...")
    
    try:
        batch_results = await batch_manager.process_batch(
            queries=queries, 
            callback=progress_callback,
            num_results=num_results
        )
        
        success_count = sum(1 for result in batch_results.values() if 'error' not in result)
        error_count = sum(1 for result in batch_results.values() if 'error' in result)
        
        print(f"API requests complete: {len(batch_results)}/{len(queries)} queries processed")
        print(f"Success: {success_count}, Errors: {error_count}")
        
        return batch_results
        
    except Exception as e:
        print(f"Error in API batch processing: {str(e)}")
        return {}

def extract_competitor_data_from_pickled_api_results(api_cache_path: str = 'aves_cache'):
    """
    Extract competitor data from pickled API results with improved error handling and validation.
    
    Args:
        api_cache_path: Path to the directory containing the API cache files
        
    Returns:
        DataFrame containing competitor data
    """
    organic_competition = {}
    error_count = 0
    success_count = 0
    empty_count = 0
    no_results_count = 0

    print(f"Extracting competitor data from {api_cache_path}...")
    
    # Ensure the cache directory exists
    if not os.path.exists(api_cache_path):
        print(f"Warning: Cache directory '{api_cache_path}' does not exist")
        return pd.DataFrame()
    
    # Get list of cache files
    cache_files = [f for f in os.listdir(api_cache_path) if f.endswith('.pkl')]
    
    if not cache_files:
        print(f"Warning: No pickle files found in '{api_cache_path}'")
        return pd.DataFrame()
    
    print(f"Found {len(cache_files)} cache files")
    
    for p in cache_files:
        try:
            with open(f'{api_cache_path}/{p}', 'rb') as f:
                avesapi_data = pickle.load(f)
                
                # Extract keyword from filename if not found in data
                filename_keyword = p.replace('search_', '').replace('.pkl', '').replace('_', ' ')
                
                # Handle different possible data structures
                if isinstance(avesapi_data, dict) and 'search_parameters' in avesapi_data:
                    keyword = avesapi_data['search_parameters'].get('query', filename_keyword)
                    results = avesapi_data.get('result', {}).get('organic_results', [])
                elif isinstance(avesapi_data, dict) and 'organic_results' in avesapi_data:
                    # Alternative structure
                    keyword = filename_keyword
                    results = avesapi_data.get('organic_results', [])
                else:
                    print(f"Warning: Unexpected data structure in {p}")
                    error_count += 1
                    continue
                
                if not keyword:
                    print(f"Warning: No keyword found in {p}")
                    error_count += 1
                    continue
                
                if not results:
                    print(f"Warning: No results found for keyword '{keyword}' in {p}")
                    no_results_count += 1
                    
                    # Add keyword to the competition dict even with no results
                    # This ensures we have a record of all keywords we tried
                    if keyword not in organic_competition:
                        organic_competition[keyword] = {
                            'position': [],
                            'url': [],
                            'title': []
                        }
                    
                    continue
                
                # Initialize entry for this keyword if it doesn't exist
                if keyword not in organic_competition:
                    organic_competition[keyword] = {
                        'position': [],
                        'url': [],
                        'title': []
                    }
                
                # Add top 3 results for this keyword
                for result in results[:3]:
                    if not isinstance(result, dict):
                        continue
                        
                    url = result.get('url', '')
                    title = result.get('title', '')
                    position = result.get('position', 0)
                    
                    if not url:
                        continue
                    
                    organic_competition[keyword]['url'].append(url)
                    organic_competition[keyword]['title'].append(title)
                    organic_competition[keyword]['position'].append(position)
                
                success_count += 1
                
        except Exception as e:
            print(f"Error processing file {p}: {str(e)}")
            error_count += 1

    print(f"Successfully processed {success_count} files")
    print(f"Encountered {error_count} errors")
    print(f"Found {no_results_count} files with no results")
    print(f"Extracted data for {len(organic_competition)} keywords")

    # Convert to DataFrame - first create a flat structure
    data = []
    for keyword, results in organic_competition.items():
        # Skip keywords with no results
        if not results['url']:
            empty_count += 1
            continue
            
        for i in range(len(results['url'])):
            if i < len(results['position']) and i < len(results['title']):
                data.append({
                    'keyword': keyword,
                    'position': results['position'][i],
                    'url': results['url'][i],
                    'title': results['title'][i]
                })

    organic_competition_df = pd.DataFrame(data)
    
    print(f"Created DataFrame with {len(organic_competition_df)} rows")
    print(f"Keywords with no results (skipped): {empty_count}")
    
    return organic_competition_df