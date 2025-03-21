# main.py - Restructured version
from auth.auth import GoogleAuth
from datetime import datetime
import pandas as pd
import os
import pickle
import nltk
from workflow.workflows import get_keyword_portfolio_as_df
from content_engineering.embedding import HTMLComparator
from content_engineering.bm25 import BM25Scorer
from utils.clean import clean_content

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Configuration constants
CREDENTIALS_PATH = 'client_secret.json'
SPREADSHEET_ID = '1EA9DOhRqySnP2xssBHZ9LA6YdLfGelgISYYvlv_suX4'
SHEET_NAME = 'Keyword Portfolio'
COMPETITOR_DATA_FILE_PATH = f'competitor_data_for_stripe_reporting_{datetime.now().strftime("%d-%m-%Y")}.csv'
HTML_DIRECTORY = r"html_from_sf"
EMBEDDINGS_FILE = 'all_embeddings.pkl'
BM25_SCORES_FILE = 'bm25_scores.pkl'

def load_keyword_data():
    """Load keyword portfolio data from Google Sheets"""
    auth = GoogleAuth(credentials_path=CREDENTIALS_PATH)
    return get_keyword_portfolio_as_df(spreadsheet_id=SPREADSHEET_ID, sheet_name=SHEET_NAME)

def load_competitor_data():
    """Load or extract competitor data from API results"""
    from workflow.workflows import extract_competitor_data_from_pickled_api_results
    competitor_df = extract_competitor_data_from_pickled_api_results()
    
    if not os.path.exists(COMPETITOR_DATA_FILE_PATH):
        competitor_df.to_csv(COMPETITOR_DATA_FILE_PATH, index=False)
    
    return competitor_df

def process_html_content():
    """Process HTML content and generate embeddings with fallback handling"""
    comparator = HTMLComparator(directory=HTML_DIRECTORY)
    comparator.find_files()
    comparator.pair_files()
    comparator.compare_file_lengths()
    
    # Check if embeddings already exist
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Loading existing embeddings from {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            results = pickle.load(f)
    else:
        try:
            print("Processing HTML files to generate embeddings...")
            results = comparator.process_all_files()
            
            # Save embeddings
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(results, f)
                
            print(f"Embeddings saved to {EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"Error processing HTML files: {str(e)}")
            # Return empty results if processing fails
            results = {}
    
    # Create DataFrame from results
    if results:
        df = pd.DataFrame.from_dict(results, orient='index')
        
        # Filter for required columns and handle missing columns
        required_cols = ['content', 'embedding', 'url']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if set(available_cols) != set(required_cols):
            missing = set(required_cols) - set(available_cols)
            print(f"Warning: Missing columns in embeddings data: {missing}")
            
            # Add missing columns with None values
            for col in missing:
                df[col] = None
        
        # Select only the required columns
        return df[required_cols]
    else:
        print("Warning: No embedding results available")
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['content', 'embedding', 'url'])

def enrich_competitor_data(competitor_df, embedding_df):
    """Add embeddings and clean content to competitor data, handling missing values"""
    # Handle empty embedding_df case
    if embedding_df.empty:
        print("Warning: No embeddings available for enrichment")
        
        # Add empty columns for embedding and content if needed
        if 'embedding' not in competitor_df.columns:
            competitor_df['embedding'] = None
        if 'content' not in competitor_df.columns:
            competitor_df['content'] = None
        if 'clean_content' not in competitor_df.columns:
            competitor_df['clean_content'] = None
            
        return competitor_df
    
    # Merge embeddings with competitor data
    print(f"Merging embeddings with competitor data...")
    print(f"Competitor data: {len(competitor_df)} rows")
    print(f"Embedding data: {len(embedding_df)} rows")
    
    # Check for invalid or missing URLs before merge
    invalid_urls = competitor_df[~competitor_df['url'].str.startswith(('http://', 'https://'))]['url'].nunique()
    if invalid_urls > 0:
        print(f"Warning: Found {invalid_urls} invalid URLs in competitor data")
    
    # Count URLs in both dataframes to estimate merge quality
    common_urls = set(competitor_df['url']).intersection(set(embedding_df['url']))
    print(f"URLs in both datasets: {len(common_urls)} out of {competitor_df['url'].nunique()} competitor URLs")
    
    # Perform the merge
    columns_to_merge = ['url', 'embedding', 'content']
    merged_df = competitor_df.merge(
        embedding_df[columns_to_merge], 
        on='url', 
        how='left'
    )
    
    # Report merge results
    null_embeddings = merged_df['embedding'].isna().sum()
    print(f"Rows with missing embeddings after merge: {null_embeddings} ({null_embeddings/len(merged_df)*100:.1f}%)")
    
    # Filter and clean the data
    filtered_df = merged_df[~merged_df['url'].str.endswith('pdf', na=False)]
    print(f"Removed {len(merged_df) - len(filtered_df)} PDF URLs")
    
    # Only filter by embeddings if we actually have some
    if merged_df['embedding'].notna().any():
        filtered_df = filtered_df[filtered_df['embedding'].notna()]
        print(f"Removed {len(merged_df) - len(filtered_df)} rows with missing embeddings")
    else:
        print("Warning: All embeddings are missing, skipping embedding filter")
    
    # Clean content
    filtered_df['clean_content'] = filtered_df['content'].apply(
        lambda x: clean_content(x) if isinstance(x, str) else ""
    )
    
    return filtered_df

def calculate_bm25_scores(competitor_df):
    """Calculate BM25 scores for each keyword-document pair with improved error handling"""
    bm25_scorer = BM25Scorer(verbose=True)
    
    # Group data by keyword
    grouped_df = competitor_df.groupby('keyword').agg({
        'clean_content': list,
        'content': list,
        'url': list,
        'position': list
    }).reset_index()
    
    if os.path.exists(BM25_SCORES_FILE):
        # Ensure we return a DataFrame, not a dictionary
        bm25_data = pd.read_pickle(BM25_SCORES_FILE)
        if isinstance(bm25_data, dict):
            # Convert dictionary to DataFrame if needed
            return transform_bm25_dict_to_df(bm25_data)
        return bm25_data
    
    # Calculate BM25 scores with direct URL mapping
    bm25_results = []
    
    for _, row in grouped_df.iterrows():
        keyword = row['keyword']
        positions = row['position']
        urls = row['url']
        texts = row['clean_content']
        
        # Skip if any required data is missing
        if not all([keyword, len(texts) > 0, len(urls) > 0, len(positions) > 0]):
            print(f"Skipping keyword '{keyword}' due to missing data")
            continue
            
        # Ensure all arrays have the same length
        if not (len(texts) == len(urls) == len(positions)):
            print(f"Warning: Mismatched array lengths for keyword '{keyword}'")
            print(f"  texts: {len(texts)}, urls: {len(urls)}, positions: {len(positions)}")
            
            # Use the minimum length to avoid indexing errors
            min_length = min(len(texts), len(urls), len(positions))
            texts = texts[:min_length]
            urls = urls[:min_length]
            positions = positions[:min_length]
        
        try:
            # Fit the BM25 model with the texts for this keyword
            bm25_scorer.fit(texts)
            
            # Score the query against all documents
            scores = bm25_scorer.score_all(keyword)
            
            # Map scores directly to URLs and positions
            for doc_idx, score in scores:
                if doc_idx < len(urls):  # Ensure index is valid
                    bm25_results.append({
                        'keyword': keyword,
                        'url': urls[doc_idx],
                        'position': positions[doc_idx],
                        'bm25_score': score
                    })
                else:
                    print(f"Warning: Invalid doc_idx {doc_idx} for keyword '{keyword}' (max {len(urls)-1})")
        except Exception as e:
            print(f"Error calculating BM25 scores for keyword '{keyword}': {str(e)}")
    
    # Convert to DataFrame and save
    bm25_df = pd.DataFrame(bm25_results)
    
    # Save the DataFrame
    bm25_df.to_pickle(BM25_SCORES_FILE)
    
    return bm25_df

def transform_bm25_dict_to_df(bm25_dict):
    """Transform BM25 scores dictionary to a properly structured DataFrame"""
    if not isinstance(bm25_dict, dict):
        print(f"Warning: Expected dict but got {type(bm25_dict).__name__}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    rows = []
    
    # Check if the dictionary has the expected keys
    required_keys = ['keyword', 'url', 'position', 'bm25_score']
    if not all(key in bm25_dict for key in required_keys):
        print(f"Warning: Dictionary is missing required keys. Available keys: {list(bm25_dict.keys())}")
        print("Attempting to extract data with available keys...")
    
    try:
        keywords = bm25_dict.get('keyword', [])
        urls = bm25_dict.get('url', [])
        scores = bm25_dict.get('bm25_score', [])
        positions = bm25_dict.get('position', [])
        
        # Validate data
        if len(keywords) == 0:
            print("No keywords found in the dictionary")
            return pd.DataFrame()
            
        # Ensure all lists have the same length
        length = min(len(item) for item in [keywords, urls, scores, positions] if len(item) > 0)
        
        for i in range(length):
            keyword = keywords[i] if i < len(keywords) else None
            url_list = urls[i] if i < len(urls) else []
            score_list = scores[i] if i < len(scores) else []
            position_list = positions[i] if i < len(positions) else []
            
            # For shorter lists, use the available items
            max_items = max(
                len(url_list) if isinstance(url_list, list) else 0,
                len(score_list) if isinstance(score_list, list) else 0,
                len(position_list) if isinstance(position_list, list) else 0
            )
            
            for j in range(max_items):
                url = url_list[j] if isinstance(url_list, list) and j < len(url_list) else None
                
                # Handle score tuples
                score = None
                if isinstance(score_list, list) and j < len(score_list):
                    if isinstance(score_list[j], tuple) and len(score_list[j]) == 2:
                        doc_idx, score_value = score_list[j]
                        score = score_value
                    else:
                        score = score_list[j]
                
                # Get position
                position = position_list[j] if isinstance(position_list, list) and j < len(position_list) else None
                
                if keyword and url and score is not None:
                    rows.append({
                        'keyword': keyword,
                        'url': url,
                        'position': position,
                        'bm25_score': score
                    })
    
    except Exception as e:
        print(f"Error transforming BM25 dictionary to DataFrame: {str(e)}")
        
    return pd.DataFrame(rows)

def main():
    """Main execution flow with improved BM25 handling"""
    # Step 1: Load data
    print("Loading keyword data...")
    keyword_df = load_keyword_data()
    
    print("Loading competitor data...")
    competitor_df = load_competitor_data()
    
    # Step 2: Process HTML content and get embeddings
    print("Processing HTML content...")
    embedding_df = process_html_content()
    
    # Step 3: Enrich competitor data with embeddings and clean content
    print("Enriching competitor data...")
    enriched_df = enrich_competitor_data(competitor_df, embedding_df)
    enriched_df.to_csv('claude_input.csv', index=False)
    
    # Step 4: Calculate BM25 scores - now returns a DataFrame directly
    print("Calculating BM25 scores...")
    bm25_df = calculate_bm25_scores(enriched_df)
    
    # Step 5: Merge BM25 scores with competitor data
    print("Merging BM25 scores with competitor data...")
    if len(bm25_df) > 0:
        final_df = enriched_df.merge(bm25_df, how='left', on=['url', 'keyword', 'position'])
    else:
        print("Warning: No BM25 scores available. Returning enriched DataFrame without scores.")
        final_df = enriched_df
    
    # Step 6: Debug missing scores
    missing_scores = final_df[final_df['bm25_score'].isna()]
    print("Data processing complete!")
    print(f"Rows without BM25 score: {missing_scores.shape[0]} out of {final_df.shape[0]} ({missing_scores.shape[0]/final_df.shape[0]*100:.2f}%)")
    
    if missing_scores.shape[0] > 0:
        # Analyze the rows with missing scores
        print("\nAnalyzing rows with missing BM25 scores:")
        print(f"Number of unique keywords with missing scores: {missing_scores['keyword'].nunique()}")
        print(f"Number of unique URLs with missing scores: {missing_scores['url'].nunique()}")
        
        # Check if any keywords are completely missing scores
        all_keywords = set(final_df['keyword'].unique())
        missing_keywords = set(missing_scores['keyword'].unique())
        scored_keywords = all_keywords - missing_keywords
        
        print(f"Keywords with at least one scored row: {len(scored_keywords)} out of {len(all_keywords)}")
        
        # Sample a few problematic rows
        if missing_scores.shape[0] > 0:
            print("\nSample of rows with missing BM25 scores:")
            sample = missing_scores.sample(min(5, missing_scores.shape[0]))
            for _, row in sample.iterrows():
                print(f"Keyword: {row['keyword']}")
                print(f"URL: {row['url']}")
                print(f"Has embedding: {row['embedding'] is not None}")
                print(f"Clean content length: {len(row['clean_content']) if isinstance(row['clean_content'], str) else 'N/A'}")
                print("-" * 50)
    
    return final_df

if __name__ == "__main__":
    main()