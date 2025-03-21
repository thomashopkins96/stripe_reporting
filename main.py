from auth.auth import GoogleAuth
from datetime import datetime
from test import *
import pandas as pd
import os
import pickle
from content_engineering.tokenize import tokenize_for_bm25, clean_content
from rank_bm25 import BM25Plus
import nltk
from workflow.workflows import *
from utils.clean import remove_stopwords
from content_engineering.bm25 import BM25Scorer

nltk.download('stopwords')
nltk.download('punkt')

AVES_API_KEY = '68DCNVPG4P4BAZMZGSY266RWCRYG'
CREDENTIALS_PATH = 'client_secret.json'
SPREADSHEET_ID = '1EA9DOhRqySnP2xssBHZ9LA6YdLfGelgISYYvlv_suX4'
SHEET_NAME = 'Keyword Portfolio'
COMPETITOR_DATA_FILE_PATH = f'competitor_data_for_stripe_reporting_{datetime.now().strftime('%d-%m-%Y')}.csv'

if __name__ == "__main__":
    auth = GoogleAuth(credentials_path=CREDENTIALS_PATH)
    df = get_keyword_portfolio_as_df(spreadsheet_id=SPREADSHEET_ID, sheet_name=SHEET_NAME)
    #avesapi_data = asyncio.run(get_avesapi_data(api_key=AVES_API_KEY, queries=df['Keyword'].tolist()))
    competitor_df = extract_competitor_data_from_pickled_api_results()
    
    if not os.path.exists(COMPETITOR_DATA_FILE_PATH):
        competitor_df.to_csv(COMPETITOR_DATA_FILE_PATH, index=False)

    comparator = HTMLComparator(directory=r"C:\Users\whopk\Documents\Coding\Work\stripe_reporting_v2\html_from_sf")
    comparator.find_files()
    comparator.pair_files()
    comparator.compare_file_lengths()
    if not os.path.exists('all_embeddings.pkl'):
        results = comparator.process_all_files()
        
        with open('all_embeddings.pkl', 'wb') as f:
            pickle.dump(results, f)
        
    embedding_df = pd.DataFrame().from_dict(pd.read_pickle('all_embeddings.pkl'), orient='index')[['content', 'embedding', 'url']]
    
    columns_to_merge = ['url', 'embedding', 'content']
    competitor_df = competitor_df.merge(
        embedding_df[columns_to_merge],
        on='url',
        how='left'
    )
    
    competitor_df = competitor_df[~competitor_df['url'].str.endswith('pdf')]
    competitor_df = competitor_df[competitor_df['embedding'].notna()]
    competitor_df['clean_content'] = competitor_df['content'].apply(clean_content)
    competitor_df.to_csv('claude_input.csv')
    
    bm25_scorer = BM25Scorer()
    
    grouped_df = competitor_df.groupby('keyword').agg({
    'clean_content': list,
    'content': list,
    'url': list,
    'position': list
    # Add any other columns you want to group
    }).reset_index()
    
    if not os.path.exists('bm25_scores.pkl'):
        bm25_scores = {'keyword': [],
                       'position': [],
                        'url': [],
                        'clean_content': [],
                        'bm25_score': []}
        
        for keyword, position, url, text in zip(grouped_df['keyword'].tolist(), 
                                                grouped_df['position'].tolist(),
                                                grouped_df['url'].tolist(), 
                                                grouped_df['clean_content'].tolist()):
            
            bm25_scores['keyword'].append(keyword)
            bm25_scores['position'].append(position)
            bm25_scores['url'].append(url)
            corpus = bm25_scorer.fit(text)
            
            bm25_scores['bm25_score'].append(corpus.score_all(keyword))
            bm25_scores['clean_content'].append(text)
            
        with open('bm25_scores.pkl', 'wb') as f:
            pickle.dump(bm25_scores, f)
    
    # First, load your pickled dictionary
    bm25_dict = pd.read_pickle('bm25_scores.pkl')
    # Create a properly structured dataframe instead of melting
    rows = []

    # Assuming bm25_dict has 'keyword', 'url', and 'bm25_score' keys
    keywords = bm25_dict['keyword']
    urls = bm25_dict['url']
    scores = bm25_dict['bm25_score']
    positions = bm25_dict['position']
    content = bm25_dict['clean_content']

    # For each entry, extract information and create proper rows
    for i in range(len(keywords)):
        keyword = keywords[i]
        url_list = urls[i]
        score_list = scores[i]  # This is a list of (doc_idx, score) tuples
        position_list = positions[i]
        content_list = content[i]

        # For each document in the result list
        for position, text, (doc_idx, score) in zip(position_list, content_list, score_list):
            if doc_idx < len(url_list):  # Make sure the index is valid
                url = url_list[doc_idx]
                rows.append({
                    'keyword': keyword,
                    'position': position,
                    'url': url,
                    'content': text,
                    'bm25_score': score,
                })

    # Create a proper dataframe
    bm25_df = pd.DataFrame(rows)
    print(competitor_df)
    competitor_df = competitor_df.merge(bm25_df, how = 'left', on = ['url', 'keyword', 'position'])
    print(competitor_df[competitor_df['bm25_score'].isna()])
    
    
    