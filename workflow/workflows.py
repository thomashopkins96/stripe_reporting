from auth.auth import GoogleAuth
from data import sheet_data
import pandas as pd
import asyncio
import os
import pickle
from data.vendors.avesapi import AvesAPIClient

def get_keyword_portfolio_as_df(spreadsheet_id: str, sheet_name: str, header_row: int = 5, data_start_row: int = 6) -> pd.DataFrame:
    auth = GoogleAuth(credentials_path='client_secret.json')
    df = sheet_data.GSheetsClient(auth).get_all_as_dataframes(spreadsheet_id, 
                                                         sheet_name,
                                                         header_row = header_row,
                                                         data_start_row = data_start_row)
    
    df = df['Keyword Portfolio']
    if 'Current Landing Page' in df.columns:
        df['Current Landing Page'] = df['Current Landing Page'].apply(
        lambda x: (f'https://{x.strip()}' 
                if x and x.strip() and not x.strip().startswith(('http://', 'https://')) 
                else x)
    )
    return df

async def get_avesapi_data(api_key: str, queries: list[str], num_results: int = 3):
    client = AvesAPIClient(api_key)
    batch_results = await client.batch_search(queries=queries, num_results = num_results)
    
    for query, result in batch_results.items():
        if "error" in result:
            print(f"Query '{query}' failed: {result['error']}")
        else:
            print(f"Query '{query}' found {len(result.get('results', []))} results")
            
    return batch_results

def extract_competitor_data_from_pickled_api_results(api_cache_path: str = 'aves_cache'):
    organic_competition = {}

    for p in os.listdir(api_cache_path):
        with open(f'{api_cache_path}/{p}', 'rb') as f:
            avesapi_data = pickle.load(f)
            keyword = avesapi_data['search_parameters']['query']
            
            # Initialize entry for this keyword if it doesn't exist
            if keyword not in organic_competition:
                organic_competition[keyword] = {
                    'position': [],
                    'url': [],
                    'title': []
                }
            
            # Add top 3 results for this keyword
            for result in avesapi_data['result']['organic_results'][:3]:
                organic_competition[keyword]['url'].append(result['url'])
                organic_competition[keyword]['title'].append(result['title'])
                organic_competition[keyword]['position'].append(result['position'])

    # Convert to DataFrame - first create a flat structure
    data = []
    for keyword, results in organic_competition.items():
        for i in range(len(results['url'])):
            data.append({
                'keyword': keyword,
                'position': results['position'][i],
                'url': results['url'][i],
                'title': results['title'][i]
            })

    organic_competition_df = pd.DataFrame(data)
    
    return organic_competition_df