from rank_bm25 import BM25Plus
from typing import List, Dict, Tuple, Any, Optional
import re
import numpy as np
import time
import sys
from tqdm import tqdm


class BM25Scorer:
    """
    A wrapper class for the rank_bm25 library's BM25Plus implementation with progress reporting.
    
    This class provides convenient methods for document tokenization,
    corpus processing, and query scoring using the BM25Plus algorithm.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0, verbose: bool = True) -> None:
        """
        Initialize the BM25Scorer with BM25Plus parameters.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Document length normalization parameter (default: 0.75)
            delta: Parameter for BM25Plus's term frequency normalization (default: 1.0)
            verbose: Whether to show progress information (default: True)
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.verbose = verbose
        self.bm25 = None
        self.corpus_tokenized = []
        self.original_documents = []
    
    def _print_progress(self, message: str) -> None:
        """Print progress message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] {message}")
            sys.stdout.flush()  # Ensure message is displayed immediately
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into terms using simple whitespace splitting and lowercasing.
        
        Args:
            text: The input text to tokenize
            
        Returns:
            A list of tokens
        """
        # Convert to lowercase and replace punctuation with spaces
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split on whitespace and filter out empty tokens
        tokens = [token for token in text.split() if token]
        return tokens
    
    def fit(self, documents: List[str]) -> 'BM25Scorer':
        """
        Process a corpus of documents and create the BM25Plus model with progress reporting.
        
        Args:
            documents: List of document strings
            
        Returns:
            Self (for method chaining)
        """
        start_time = time.time()
        self._print_progress(f"Starting to process {len(documents)} documents")
        
        # Check for empty documents
        empty_docs = [i for i, doc in enumerate(documents) if not doc.strip()]
        if empty_docs:
            self._print_progress(f"WARNING: Found {len(empty_docs)} empty documents. This may affect scoring.")
            if len(empty_docs) <= 5:
                self._print_progress(f"Empty document indices: {empty_docs}")
        
        self.original_documents = documents
        
        # Tokenize documents with progress bar
        self._print_progress("Tokenizing documents...")
        self.corpus_tokenized = []
        
        if self.verbose:
            for doc_idx, doc in enumerate(tqdm(documents, desc="Tokenizing", unit="doc")):
                tokens = self.tokenize(doc)
                self.corpus_tokenized.append(tokens)
                
                # Warn about empty documents after tokenization
                if not tokens and doc.strip():
                    self._print_progress(f"WARNING: Document {doc_idx} tokenized to empty list: '{doc[:50]}...'")
        else:
            self.corpus_tokenized = [self.tokenize(doc) for doc in documents]
        
        # Check for empty token lists
        empty_token_docs = [i for i, tokens in enumerate(self.corpus_tokenized) if not tokens]
        if empty_token_docs:
            self._print_progress(f"WARNING: {len(empty_token_docs)} documents tokenized to empty lists.")
            if len(empty_token_docs) <= 5:
                self._print_progress(f"Documents with empty token lists: {empty_token_docs}")
        
        self._print_progress(f"Tokenization complete. Processed {len(documents)} documents.")
        
        # Build BM25Plus model
        self._print_progress("Building BM25Plus index...")
        index_start_time = time.time()
        
        # Check for issues before building the model
        if not self.corpus_tokenized:
            self._print_progress("ERROR: No documents to index!")
            return self
        if all(not tokens for tokens in self.corpus_tokenized):
            self._print_progress("ERROR: All documents tokenized to empty lists!")
            return self
        
        self.bm25 = BM25Plus(
            self.corpus_tokenized, 
            k1=self.k1, 
            b=self.b, 
            delta=self.delta
        )
        
        index_time = time.time() - index_start_time
        self._print_progress(f"BM25Plus index built in {index_time:.2f} seconds")
        
        # Report on vocabulary size and document statistics
        vocab = set()
        token_counts = []
        for doc in self.corpus_tokenized:
            vocab.update(doc)
            token_counts.append(len(doc))
        
        # Compute vocabulary statistics
        total_time = time.time() - start_time
        self._print_progress(f"Fitting complete. Vocabulary size: {len(vocab)} unique terms")
        
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            self._print_progress(f"Document statistics:")
            self._print_progress(f"  Average tokens per document: {avg_tokens:.2f}")
            self._print_progress(f"  Min tokens: {min_tokens}, Max tokens: {max_tokens}")
        
        # Display some top terms by document frequency for debugging
        if vocab:
            term_doc_counts = {}
            for term in vocab:
                doc_count = sum(1 for doc in self.corpus_tokenized if term in doc)
                term_doc_counts[term] = doc_count
            
            # Sort terms by document frequency
            top_terms = sorted(term_doc_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            self._print_progress("Top 10 most frequent terms in corpus:")
            for term, count in top_terms:
                self._print_progress(f"  '{term}' appears in {count}/{len(self.corpus_tokenized)} documents")
        
        self._print_progress(f"Total processing time: {total_time:.2f} seconds")
        
        return self
    
    def score(self, query: str, doc_idx: int) -> float:
        """
        Calculate BM25Plus score for a query against a specific document.
        
        Args:
            query: The search query
            doc_idx: Index of the document to score
            
        Returns:
            The BM25Plus score for the query and document
        """
        if self.bm25 is None:
            raise ValueError("BM25Scorer has not been initialized with documents. Call fit() first.")
        
        # Tokenize the query
        self._print_progress(f"Scoring document {doc_idx} for query: '{query}'")
        query_tokens = self.tokenize(query)
        
        # Get score for the specific document
        start_time = time.time()
        doc_scores = self.bm25.get_scores(query_tokens)
        score_time = time.time() - start_time
        
        if self.verbose:
            self._print_progress(f"Document {doc_idx} scored in {score_time:.4f} seconds. Score: {doc_scores[doc_idx]:.4f}")
        
        return doc_scores[doc_idx]
    
    def score_all(self, query: str) -> List[Tuple[int, float]]:
        """
        Score a query against all documents in the corpus.
        
        Args:
            query: The search query
            
        Returns:
            List of (doc_idx, score) tuples sorted by descending score
        """
        if self.bm25 is None:
            raise ValueError("BM25Scorer has not been initialized with documents. Call fit() first.")
        
        # Tokenize the query
        self._print_progress(f"Scoring all documents for query: '{query}'")
        start_time = time.time()
        query_tokens = self.tokenize(query)
        
        # Debug: Check if query tokens exist
        if self.verbose:
            if not query_tokens:
                self._print_progress("WARNING: Query tokenized to empty list! No matches possible.")
            else:
                self._print_progress(f"Query tokenized to: {query_tokens}")
                
                # Check if query terms appear in the corpus
                term_doc_counts = {}
                for term in query_tokens:
                    doc_count = sum(1 for doc in self.corpus_tokenized if term in doc)
                    term_doc_counts[term] = doc_count
                
                self._print_progress("Query term document frequencies:")
                for term, count in term_doc_counts.items():
                    self._print_progress(f"  '{term}' appears in {count}/{len(self.corpus_tokenized)} documents")
                
                if all(count == 0 for count in term_doc_counts.values()):
                    self._print_progress("WARNING: None of the query terms appear in any document!")
        
        # Score all documents
        scoring_start = time.time()
        doc_scores = self.bm25.get_scores(query_tokens)
        scoring_time = time.time() - scoring_start
        
        # Create (doc_idx, score) tuples and sort by descending score
        sorting_start = time.time()
        scored_docs = [(i, score) for i, score in enumerate(doc_scores)]
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        sorting_time = time.time() - sorting_start
        
        total_time = time.time() - start_time
        
        # Report statistics
        if self.verbose:
            positive_scores = sum(1 for _, score in scored_docs if score > 0)
            max_score = max(doc_scores) if doc_scores.size > 0 else 0
            
            self._print_progress(f"Scored {len(doc_scores)} documents in {scoring_time:.4f} seconds")
            self._print_progress(f"Sorting completed in {sorting_time:.4f} seconds")
            self._print_progress(f"Documents with positive scores: {positive_scores}/{len(doc_scores)}")
            self._print_progress(f"Maximum score: {max_score:.4f}")
            
            # Debug: Show top document matches if any had positive scores
            if positive_scores > 0:
                self._print_progress("\nTop document matches:")
                for idx, (doc_idx, score) in enumerate(sorted_docs[:3]):
                    if score > 0:
                        doc_preview = self.original_documents[doc_idx][:100] + "..." if len(self.original_documents[doc_idx]) > 100 else self.original_documents[doc_idx]
                        self._print_progress(f"  {idx+1}. Doc {doc_idx}: {doc_preview}")
                        self._print_progress(f"     Score: {score:.4f}")
            elif self.corpus_tokenized:
                # Debug: If no matches, show a sample document for reference
                sample_doc_idx = 0
                self._print_progress("\nNo matches found. Sample document for reference:")
                self._print_progress(f"Document {sample_doc_idx}: {self.original_documents[sample_doc_idx]}")
                self._print_progress(f"Tokenized as: {self.corpus_tokenized[sample_doc_idx]}")
            
            self._print_progress(f"Total scoring and sorting time: {total_time:.4f} seconds")
        
        return sorted_docs
    
    def search(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching a query and return the top results.
        
        Args:
            query: The search query
            top_n: Maximum number of results to return (default: 5)
            
        Returns:
            List of dictionaries containing document info and scores
        """
        if self.bm25 is None:
            raise ValueError("BM25Scorer has not been initialized with documents. Call fit() first.")
        
        self._print_progress(f"Searching for: '{query}' (top {top_n} results)")
        start_time = time.time()
        
        # Get sorted (doc_idx, score) tuples
        scores = self.score_all(query)
        
        # Limit to top_n results
        top_results = scores[:top_n]
        
        # Format the results
        results = [
            {
                "document_idx": idx,
                "score": score,
                "text": self.original_documents[idx]
            }
            for idx, score in top_results
            if score > 0  # Only include documents with positive scores
        ]
        
        total_time = time.time() - start_time
        self._print_progress(f"Search completed in {total_time:.4f} seconds. Found {len(results)} relevant documents.")
        
        return results