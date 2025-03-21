import asyncio
import aiohttp
import json
import os
import time
from utils.async_pickler import async_pickle_output
from typing import List, Dict, Any, Optional, Union, Tuple

class AvesAPIClient:
    """
    Asynchronous client for the Aves API search endpoint.
    Handles concurrent requests, rate limiting, and error handling.
    """
    
    BASE_URL = "https://api.avesapi.com/search"
    
    def __init__(
        self, 
        api_key: str, 
        max_concurrent_requests: int = 5,
        rate_limit_per_min: int = 60,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        cache_directory: str = "aves_cache",
        cache_expiry: Optional[int] = 86400  # 24 hours default cache expiry
    ):
        """
        Initialize the AvesAPIClient.
        
        Args:
            api_key: Your Aves API key
            max_concurrent_requests: Maximum number of concurrent requests
            rate_limit_per_min: Maximum number of requests per minute (rate limit)
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
            cache_directory: Directory to store cached results
            cache_expiry: Time in seconds after which cache expires (None for no expiry)
        """
        self.api_key = api_key
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_min = rate_limit_per_min
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.cache_directory = cache_directory
        self.cache_expiry = cache_expiry
        
        # Rate limiting tracking
        self.request_timestamps: List[float] = []
        
        # Semaphore to control concurrent requests
        self._semaphore = None
        
        # Create cache directory
        os.makedirs(cache_directory, exist_ok=True)
    
    async def _init_semaphore(self):
        """Initialize the semaphore for controlling concurrent requests."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    
    async def _wait_for_rate_limit(self):
        """
        Wait if necessary to comply with the rate limit.
        """
        now = time.time()
        
        # Remove timestamps older than 60 seconds
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        # If we've hit the rate limit, wait until we can make another request
        if len(self.request_timestamps) >= self.rate_limit_per_min:
            oldest_timestamp = min(self.request_timestamps)
            sleep_time = 60 - (now - oldest_timestamp)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Add current timestamp
        self.request_timestamps.append(time.time())
    
    @async_pickle_output(directory="aves_cache", expire_after=86400)  # Cache for 24 hours
    async def search(
        self, 
        query: str,
        google_domain: str = "google.com",
        gl: str = "us",
        hl: str = "en",
        device: str = "desktop",
        num_results: int = 10,
        **additional_params
    ) -> Dict[str, Any]:
        """
        Perform a single search request to the Aves API.
        Results are cached to avoid repeated API calls.
        
        Args:
            query: The search query
            google_domain: Google domain to use
            gl: Geographic location
            hl: Language
            device: Device type (desktop, mobile, tablet)
            num_results: Number of results to return
            **additional_params: Additional parameters to include in the request
            
        Returns:
            Response data from the API
        
        Raises:
            Exception: If the request fails after all retry attempts
        """
        await self._init_semaphore()
        
        params = {
            "apikey": self.api_key,
            "type": "web",
            "query": query,
            "google_domain": google_domain,
            "gl": gl,
            "hl": hl,
            "device": device,
            "output": "json",
            "num": num_results
        }
        
        # Add any additional parameters
        params.update(additional_params)
        
        async with self._semaphore:
            # Check rate limit before making request
            await self._wait_for_rate_limit()
            
            for attempt in range(self.retry_attempts):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(self.BASE_URL, params=params) as response:
                            if response.status == 200:
                                return await response.json()
                            elif response.status == 429:  # Too Many Requests
                                # Wait longer for rate limit reset
                                wait_time = self.retry_delay * (2 ** attempt)
                                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                # Log error and retry
                                error_text = await response.text()
                                print(f"Error {response.status}: {error_text}")
                                await asyncio.sleep(self.retry_delay)
                except aiohttp.ClientError as e:
                    # Network error, retry
                    print(f"Network error: {str(e)}. Retrying...")
                    await asyncio.sleep(self.retry_delay)
            
            # If we've exhausted all retries
            raise Exception(f"Failed to get results for query '{query}' after {self.retry_attempts} attempts")
    
    @async_pickle_output(directory="aves_cache", expire_after=86400)  # Cache for 24 hours
    async def batch_search(
        self, 
        queries: List[str],
        google_domain: str = "google.com",
        gl: str = "us",
        hl: str = "en",
        device: str = "desktop",
        num_results: int = 10,
        **additional_params
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform multiple search requests concurrently.
        Results are cached to avoid repeated API calls.
        
        Args:
            queries: List of search queries to process
            google_domain: Google domain to use
            gl: Geographic location
            hl: Language
            device: Device type (desktop, mobile, tablet)
            num_results: Number of results to return
            **additional_params: Additional parameters to include in the requests
            
        Returns:
            Dictionary with queries as keys and their respective search results as values
        """
        await self._init_semaphore()
        
        async def process_query(query):
            try:
                result = await self.search(
                    query=query,
                    google_domain=google_domain,
                    gl=gl,
                    hl=hl,
                    device=device,
                    num_results=num_results,
                    **additional_params
                )
                return query, result
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                return query, {"error": str(e)}
        
        # Create tasks for all queries
        tasks = [process_query(query) for query in queries]
        
        # Run all tasks concurrently and collect results
        results = {}
        for future in asyncio.as_completed(tasks):
            query, result = await future
            results[query] = result
        
        return results


class BatchRequestManager:
    """
    Manager for handling large batches of requests with progress tracking and error handling.
    """
    
    def __init__(
        self, 
        client: AvesAPIClient,
        chunk_size: int = 20,
        pause_between_chunks: float = 2.0
    ):
        """
        Initialize the BatchRequestManager.
        
        Args:
            client: AvesAPIClient instance to use for requests
            chunk_size: Number of requests to process in each chunk
            pause_between_chunks: Pause time between chunks in seconds
        """
        self.client = client
        self.chunk_size = chunk_size
        self.pause_between_chunks = pause_between_chunks
    
    @async_pickle_output(directory="aves_batch_cache", expire_after=86400)  # Cache for 24 hours
    async def process_batch(
        self,
        queries: List[str],
        callback = None,
        **search_params
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process a large batch of queries in chunks.
        Results are cached to avoid repeated processing.
        
        Args:
            queries: List of search queries to process
            callback: Optional callback function to call after each chunk with
                     (completed_count, total_count, current_results) parameters
            **search_params: Parameters to pass to the search method
            
        Returns:
            Combined results from all chunks
        """
        total_queries = len(queries)
        results = {}
        
        # Process queries in chunks
        for i in range(0, total_queries, self.chunk_size):
            chunk = queries[i:i + self.chunk_size]
            
            # Process current chunk
            chunk_results = await self.client.batch_search(queries=chunk, **search_params)
            results.update(chunk_results)
            
            # Call callback if provided
            if callback:
                completed = min(i + self.chunk_size, total_queries)
                await callback(completed, total_queries, results)
            
            # Pause between chunks if not the last chunk
            if i + self.chunk_size < total_queries:
                await asyncio.sleep(self.pause_between_chunks)
        
        return results