import pickle
import functools
import time
import os

from typing import Callable, Any, Optional, Dict, Union
from datetime import datetime

def async_pickle_output(
    directory: str = "cache",
    include_args: bool = True,
    expire_after: Optional[int] = None,  # Cache expiration in seconds
    protocol: int = pickle.HIGHEST_PROTOCOL
):
    """
    Decorator that caches the output of an async function to a pickle file.
    
    Args:
        directory: Directory to save pickles
        include_args: Whether to include function arguments in the filename
        expire_after: Number of seconds after which the cache expires
        protocol: Pickle protocol version to use
    
    Returns:
        The decorated async function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Generate filename based on function name and arguments
            func_name = func.__name__
            
            if include_args:
                # For the search method, use query as the key identifier
                if func_name == "search" and "query" in kwargs:
                    query = kwargs["query"]
                    # Sanitize filename
                    sanitized_query = "".join(c if c.isalnum() or c == "_" else "_" for c in query)
                    filename = f"{func_name}_{sanitized_query[:50]}.pkl"
                # For batch_search, use a hash of the queries
                elif func_name == "batch_search" and "queries" in kwargs:
                    queries_str = "_".join(kwargs["queries"][:3])  # Use first 3 queries for name
                    if len(kwargs["queries"]) > 3:
                        queries_str += f"_plus_{len(kwargs['queries'])-3}"
                    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in queries_str)
                    filename = f"{func_name}_{sanitized[:50]}.pkl"
                else:
                    # Generic argument-based filename
                    args_str = "_".join(str(arg)[:10] for arg in args 
                   if not (hasattr(arg, '__class__') and 
                          arg.__class__.__name__ == 'AvesAPIClient'))
                    kwargs_str = "_".join(f"{k}_{v}"[:10] for k, v in kwargs.items() 
                                        if k not in ["google_domain", "gl", "hl", "device"])
                    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in f"{args_str}_{kwargs_str}")
                    filename = f"{func_name}_{sanitized[:50]}.pkl"
            else:
                filename = f"{func_name}.pkl"
            
            full_path = os.path.join(directory, filename)
            
            # Check if pickle exists and is not expired
            if os.path.exists(full_path):
                try:
                    # Check if the cache is expired
                    if expire_after is not None:
                        modified_time = os.path.getmtime(full_path)
                        if time.time() - modified_time > expire_after:
                            print(f"Cache expired for {filename}, fetching fresh data...")
                        else:
                            # Load from pickle if not expired
                            with open(full_path, 'rb') as f:
                                cached_data = pickle.load(f)
                                print(f"Using cached data from {filename}")
                                return cached_data
                    else:
                        # No expiration set, just use the cache
                        with open(full_path, 'rb') as f:
                            cached_data = pickle.load(f)
                            print(f"Using cached data from {filename}")
                            return cached_data
                except Exception as e:
                    print(f"Error reading cache {filename}: {str(e)}")
            
            # Execute the function if cache doesn't exist or is expired
            result = await func(*args, **kwargs)
            
            # Save the result to pickle
            try:
                with open(full_path, 'wb') as f:
                    pickle.dump(result, f, protocol=protocol)
                print(f"Saved result to cache: {filename}")
            except Exception as e:
                print(f"Error saving to cache {filename}: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator