import os
import re
import trafilatura
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any, Set
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)


class URLExtractor:
    """Extract URLs from filenames following specific patterns."""
    
    @staticmethod
    def extract_from_filename(filename: str) -> Optional[str]:
        """
        Extract the URL from a filename that follows the pattern:
        prefix_https_domain_path_components.extension
        
        Args:
            filename: The filename to extract URL from
            
        Returns:
            Extracted URL or None if not found
        """
        # Split the filename by underscore after the prefix
        parts = filename.split('_', 1)
        if len(parts) < 2:
            return None
            
        remaining = parts[1]
        
        # Check if it starts with http or https
        if remaining.startswith('http_'):
            protocol = "http"
            path = remaining[5:]  # Remove 'http_'
        elif remaining.startswith('https_'):
            protocol = "https"
            path = remaining[6:]  # Remove 'https_'
        else:
            return None
            
        # Replace underscores with appropriate characters
        # First underscore separates domain from path
        domain_parts = path.split('_', 1)
        if len(domain_parts) < 2:
            domain = domain_parts[0]
            path = ""
        else:
            domain = domain_parts[0]
            path = domain_parts[1]
        
        # Replace remaining underscores with slashes for path components
        path = path.replace('_', '/')
        
        # Remove file extension if present
        path = re.sub(r'\.[a-z]+$', '', path)
        
        return f"{protocol}://{domain}/{path}"


class FileFinder:
    """Find and categorize files based on prefixes."""
    
    def __init__(self, directory: str = '.'):
        """
        Initialize with a directory path.
        
        Args:
            directory: Path to the directory containing files
        """
        self.directory = directory
    
    def find_html_files(self, prefixes: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Find all HTML/text files in the directory and categorize them by prefixes.
        
        Args:
            prefixes: List of prefixes to categorize files (e.g., ['original', 'rendered'])
            
        Returns:
            Dictionary mapping prefixes to dictionaries of files
        """
        files = [f for f in os.listdir(self.directory) 
                if f.endswith('.txt') or f.endswith('.html')]
        
        result = {prefix: {} for prefix in prefixes}
        patterns = {prefix: re.compile(f'^{prefix}_(.+)$') for prefix in prefixes}
        
        for file in files:
            file_path = os.path.join(self.directory, file)
            
            for prefix, pattern in patterns.items():
                match = pattern.match(file)
                if match:
                    key = match.group(1)
                    result[prefix][key] = file_path
                    break
                    
        return result


class ContentExtractor:
    """Extract and process content from HTML files."""
    
    @staticmethod
    def extract_from_file(file_path: str) -> Optional[str]:
        """
        Extract content from an HTML file using trafilatura.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Extracted text content or None if extraction failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return trafilatura.extract(html_content)
        except Exception as e:
            print(f"Error extracting content from {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_length(file_path: str) -> int:
        """
        Get the length of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        return os.path.getsize(file_path)


class TextChunker:
    """Split text into chunks of approximately equal token length."""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks of approximately equal token length.
        
        Args:
            text: Text to split into chunks
            chunk_size: Target size of each chunk in tokens (approximate)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Estimate token count (rough approximation)
        def estimate_tokens(sentence):
            return len(sentence.split())
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            
            if current_size + sentence_tokens > chunk_size and current_chunk:
                # Current chunk is full, start a new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_tokens
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_size += sentence_tokens
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks


class EmbeddingGenerator:
    """Generate embeddings from text using sentence transformers with fallback options."""
    
    # List of models to try in order of preference
    DEFAULT_MODELS = [
        "mixedbread-ai/mxbai-embed-large-v1",  # First choice
        "all-MiniLM-L6-v2",                    # Fallback option 1
        "all-mpnet-base-v2"                    # Fallback option 2
    ]
    
    def __init__(self, model_name: str = None):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use (optional)
                        If None, will try models in DEFAULT_MODELS list
        """
        self.model_name = model_name
        self.model = None
        self.tried_models = []
    
    def load_model(self):
        """Load the sentence transformer model with fallback options."""
        if self.model is not None:
            return
            
        # If model_name is specified, try only that model
        if self.model_name:
            models_to_try = [self.model_name]
        else:
            models_to_try = self.DEFAULT_MODELS
            
        # Try models in order until one succeeds
        for model_name in models_to_try:
            if model_name in self.tried_models:
                continue
                
            self.tried_models.append(model_name)
            
            try:
                print(f"Attempting to load model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name
                print(f"Successfully loaded model: {model_name}")
                return
            except OSError as e:
                print(f"Error loading model {model_name}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error loading model {model_name}: {str(e)}")
                continue
                
        # If we get here, all models failed
        raise ValueError(f"Failed to load any embedding model. Tried: {', '.join(self.tried_models)}")
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks and average them.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            Averaged embedding vector or None if generation fails
        """
        if not chunks:
            return None
            
        try:
            self.load_model()
            
            # Generate embeddings for each chunk
            embeddings = self.model.encode(chunks)
            
            # Average the embeddings
            return np.mean(embeddings, axis=0)
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return None

class HTMLComparator:
    """
    A class to compare 'original' and 'rendered' HTML documents and extract their content.
    
    This class identifies pairs of HTML documents with matching names (one prefixed with 'original'
    and one with 'rendered'), compares their lengths, extracts content using trafilatura,
    chunks the longer document, and creates embeddings with SentenceTransformer.
    """
    
    def __init__(self, directory: str = '.', model_name: str = "mixedbread-ai/mxbai-embed-large-v1", chunk_size: int = 512):
        """
        Initialize the comparator with a directory containing HTML files.
        
        Args:
            directory: Path to the directory containing HTML files
            model_name: Name of the SentenceTransformer model to use
            chunk_size: Size of chunks in tokens (approximate)
        """
        self.directory = directory
        self.file_finder = FileFinder(directory)
        self.content_extractor = ContentExtractor()
        self.text_chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.url_extractor = URLExtractor()
        self.chunk_size = chunk_size
        
        self.file_categories = {}
        self.paired_files = {}
        self.url_mapping = {}
        self.comparison_results = {}
        self.content_cache = {}
        self.embedding_results = {}
    
    def find_files(self):
        """
        Find and categorize HTML files in the directory.
        Also extract URLs from the filenames.
        """
        self.file_categories = self.file_finder.find_html_files(['original', 'rendered'])
        
        # Extract URLs from filenames
        for category, files in self.file_categories.items():
            for key, file_path in files.items():
                filename = os.path.basename(file_path)
                url = self.url_extractor.extract_from_filename(filename)
                if url:
                    self.url_mapping[key] = url
        
        return self.file_categories
    
    def pair_files(self):
        """
        Find pairs of original and rendered files with matching keys.
        
        Returns:
            Dictionary mapping keys to pairs of file paths (original, rendered)
        """
        original_files = self.file_categories.get('original', {})
        rendered_files = self.file_categories.get('rendered', {})
        
        # Find keys that exist in both original and rendered
        common_keys = set(original_files.keys()) & set(rendered_files.keys())
        
        self.paired_files = {
            key: {
                'original': original_files[key],
                'rendered': rendered_files[key],
                'url': self.url_mapping.get(key)
            }
            for key in common_keys
        }
        
        return self.paired_files
    
    def compare_file_lengths(self):
        """
        Compare the lengths of paired original and rendered files.
        
        Returns:
            Dictionary mapping keys to comparison results
        """
        for key, files in self.paired_files.items():
            original_length = self.content_extractor.get_file_length(files['original'])
            rendered_length = self.content_extractor.get_file_length(files['rendered'])
            
            longer_type = 'original' if original_length >= rendered_length else 'rendered'
            
            self.comparison_results[key] = {
                'original_length': original_length,
                'rendered_length': rendered_length,
                'longer_type': longer_type,
                'longer_file': files[longer_type],
                'url': files.get('url')
            }
        
        return self.comparison_results
    
    def extract_content(self, file_path: str) -> Optional[str]:
        """
        Extract content from an HTML file using trafilatura with robust fallback mechanisms.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Extracted text content or None if extraction failed
        """
        if file_path in self.content_cache:
            return self.content_cache[file_path]
        
        # Try multiple extraction methods in sequence
        extracted_content = None
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                self.content_cache[file_path] = None
                return None
            
            # First attempt: trafilatura with UTF-8 encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Try trafilatura extraction with explicit text format
                extracted_content = trafilatura.extract(
                    html_content
                )
            except Exception as e:
                print(f"First extraction attempt failed for {file_path}: {str(e)}")
            
            # Second attempt: trafilatura with latin-1 encoding
            if not extracted_content:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        html_content = f.read()
                    
                    extracted_content = trafilatura.extract(
                        html_content,
                        output_format='text',
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False
                    )
                except Exception as e:
                    print(f"Second extraction attempt failed for {file_path}: {str(e)}")
            
            # Third attempt: use BeautifulSoup as fallback
            if not extracted_content:
                try:
                    from bs4 import BeautifulSoup
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Get text
                    text = soup.get_text()
                    
                    # Break into lines and remove leading and trailing space on each
                    lines = (line.strip() for line in text.splitlines())
                    # Break multi-headlines into a line each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    # Remove blank lines
                    extracted_content = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    print(f"Used BeautifulSoup fallback for {file_path}")
                except Exception as e:
                    print(f"Third extraction attempt failed for {file_path}: {str(e)}")
            
            # Final fallback: simple regex if all else fails
            if not extracted_content:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    
                    # Remove all HTML tags
                    extracted_content = re.sub(r'<[^>]+>', ' ', html_content)
                    # Remove extra whitespace
                    extracted_content = re.sub(r'\s+', ' ', extracted_content).strip()
                    
                    print(f"Used regex fallback for {file_path}")
                except Exception as e:
                    print(f"Final extraction attempt failed for {file_path}: {str(e)}")
            
            # Log result
            if extracted_content:
                content_preview = extracted_content[:50] + "..." if len(extracted_content) > 50 else extracted_content
                print(f"Successfully extracted content from {os.path.basename(file_path)} ({len(extracted_content)} chars)")
            else:
                print(f"Warning: All extraction methods failed for {file_path}")
            
            # Cache and return result
            self.content_cache[file_path] = extracted_content
            return extracted_content
            
        except Exception as e:
            print(f"Unexpected error extracting content from {file_path}: {str(e)}")
            self.content_cache[file_path] = None
            return None
    
    def process_file(self, key: str) -> Dict[str, Any]:
        """
        Process a paired file: extract content, chunk longer document, and generate embeddings.
        
        Args:
            key: The key identifying the file pair
            
        Returns:
            Dictionary with processing results
        """
        if key not in self.comparison_results:
            raise KeyError(f"No comparison results for key: {key}")
            
        result = self.comparison_results[key]
        longer_file = result['longer_file']
        
        # Extract content
        content = self.extract_content(longer_file)
        if not content:
            print(f"Warning: Failed to extract content from {longer_file}")
            return {
                'status': 'error',
                'error': f"Failed to extract content from {longer_file}",
                'url': result.get('url')
            }
        
        # Chunk content
        chunks = self.text_chunker.chunk_text(content, self.chunk_size)
        
        # Generate embeddings
        try:
            embedding = self.embedding_generator.generate_embeddings(chunks)
        except Exception as e:
            print(f"Error generating embeddings for {key}: {str(e)}")
            embedding = None
        
        self.embedding_results[key] = {
            'content': content,
            'chunks': chunks,
            'embedding': embedding,
            'longer_type': result['longer_type'],
            'url': result.get('url')
        }
        
        return self.embedding_results[key]
    
    def process_all_files(self):
        """
        Process all paired files and generate embeddings.
        
        Returns:
            Dictionary mapping keys to processing results
        """
        if not self.comparison_results:
            self.find_files()
            self.pair_files()
            self.compare_file_lengths()
        
        for key in self.comparison_results:
            self.process_file(key)
        
        return self.embedding_results
    
    def get_url(self, key: str) -> Optional[str]:
        """
        Get the URL for a specific key.
        
        Args:
            key: The key identifying the file pair
            
        Returns:
            URL or None if not found
        """
        return self.url_mapping.get(key)