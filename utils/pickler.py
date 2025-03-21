"""
Pickle Output Decorator

This module provides a decorator that automatically serializes function output to a pickle file.
It can be used to cache function results, save intermediate data, or persist output data.
"""

import os
import pickle
import functools
import time
from typing import Callable, Any, Optional, Dict, Union
from datetime import datetime

def pickle_output(
    file_path: Optional[str] = None,
    directory: str = "pickles",
    include_args: bool = False,
    include_timestamp: bool = False,
    overwrite: bool = True,
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> Callable:
    """
    Decorator that saves the output of a function to a pickle file.
    
    Args:
        file_path: Path where to save the pickle. If None, a name is generated based on the function name.
        directory: Directory to save pickle files when using auto-generated paths.
        include_args: Whether to include function arguments in the filename when auto-generating.
        include_timestamp: Whether to add a timestamp to the filename when auto-generating.
        overwrite: Whether to overwrite existing pickle files.
        protocol: Pickle protocol version to use.
    
    Returns:
        The decorated function.
    
    Examples:
        >>> @pickle_output()
        >>> def compute_data(x, y):
        >>>     # Some expensive computation
        >>>     return result
        
        >>> @pickle_output(file_path='custom_path.pkl')
        >>> def get_results():
        >>>     return results
        
        >>> @pickle_output(include_args=True, include_timestamp=True)
        >>> def process_dataset(dataset_name, limit=None):
        >>>     return processed_data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the result from the function
            result = func(*args, **kwargs)
            
            # Determine the file path
            output_path = _get_file_path(
                func.__name__, 
                file_path, 
                directory, 
                include_args, 
                include_timestamp, 
                args, 
                kwargs
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Check if file exists and handle according to overwrite setting
            if not overwrite and os.path.exists(output_path):
                print(f"Pickle file already exists at {output_path} and overwrite=False. Skipping save.")
                return result
            
            # Save to pickle
            try:
                with open(output_path, 'wb') as f:
                    pickle.dump(result, f, protocol=protocol)
                print(f"Output saved to {output_path}")
            except Exception as e:
                print(f"Error saving pickle to {output_path}: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator


def _get_file_path(
    func_name: str,
    file_path: Optional[str],
    directory: str,
    include_args: bool,
    include_timestamp: bool,
    args: tuple,
    kwargs: Dict[str, Any]
) -> str:
    """
    Determine the file path for the pickle output.
    
    Args:
        func_name: Name of the decorated function.
        file_path: User-specified file path or None.
        directory: Directory to save pickle files.
        include_args: Whether to include arguments in the filename.
        include_timestamp: Whether to include timestamp in the filename.
        args: Function positional arguments.
        kwargs: Function keyword arguments.
        
    Returns:
        The file path to use for the pickle.
    """
    if file_path:
        return file_path
    
    # Build an auto-generated filename
    filename_parts = [func_name]
    
    # Add args and kwargs to filename if requested
    if include_args:
        arg_parts = []
        
        # Include positional args
        if args:
            arg_parts.append('_'.join(str(arg).replace('/', '_').replace('\\', '_')[:10] for arg in args))
        
        # Include keyword args
        if kwargs:
            for key, value in sorted(kwargs.items()):
                arg_parts.append(f"{key}_{str(value).replace('/', '_').replace('\\', '_')[:10]}")
        
        if arg_parts:
            filename_parts.append('_'.join(arg_parts))
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts.append(timestamp)
    
    # Create the filename
    filename = '_'.join(filename_parts) + '.pkl'
    
    # Sanitize the filename
    filename = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in filename)
    
    # Join with the directory
    return os.path.join(directory, filename)


class PickleOutputManager:
    """
    A class for managing pickle output, with support for metadata and automatic loading.
    
    This provides a more advanced alternative to the simple decorator, with features like
    metadata tracking, versioning, and automatic loading.
    """
    
    def __init__(
        self, 
        directory: str = "pickles",
        create_dir: bool = True,
        protocol: int = pickle.HIGHEST_PROTOCOL
    ):
        """
        Initialize the PickleOutputManager.
        
        Args:
            directory: Directory to save pickle files.
            create_dir: Whether to create the directory if it doesn't exist.
            protocol: Pickle protocol version to use.
        """
        self.directory = directory
        self.protocol = protocol
        
        if create_dir:
            os.makedirs(directory, exist_ok=True)
    
    def save(
        self, 
        data: Any, 
        filename: str, 
        metadata: Optional[Dict[str, Any]] = None, 
        overwrite: bool = True
    ) -> str:
        """
        Save data to a pickle file with optional metadata.
        
        Args:
            data: The data to save.
            filename: The filename to use (without directory).
            metadata: Optional metadata to save alongside the data.
            overwrite: Whether to overwrite existing files.
            
        Returns:
            The full path to the saved file.
        """
        # Add .pkl extension if missing
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        full_path = os.path.join(self.directory, filename)
        
        if not overwrite and os.path.exists(full_path):
            print(f"File {full_path} already exists and overwrite=False. Skipping save.")
            return full_path
        
        # Prepare the data package (with metadata if provided)
        if metadata:
            package = {
                'data': data,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
        else:
            package = data
        
        # Save to pickle
        try:
            with open(full_path, 'wb') as f:
                pickle.dump(package, f, protocol=self.protocol)
            print(f"Data saved to {full_path}")
        except Exception as e:
            print(f"Error saving data to {full_path}: {str(e)}")
        
        return full_path
    
    def load(
        self, 
        filename: str, 
        extract_data: bool = True
    ) -> Any:
        """
        Load data from a pickle file.
        
        Args:
            filename: The filename to load (without directory).
            extract_data: If True and the pickle contains a metadata package,
                         extract and return just the data.
                         
        Returns:
            The loaded data.
        """
        # Add .pkl extension if missing
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        full_path = os.path.join(self.directory, filename)
        
        try:
            with open(full_path, 'rb') as f:
                loaded = pickle.load(f)
            
            # Check if this is a metadata package and extract if requested
            if extract_data and isinstance(loaded, dict) and 'data' in loaded and 'metadata' in loaded:
                return loaded['data']
            
            return loaded
        except Exception as e:
            print(f"Error loading data from {full_path}: {str(e)}")
            return None
    
    def output_decorator(
        self, 
        filename: Optional[str] = None,
        include_args: bool = False,
        include_timestamp: bool = False,
        add_metadata: bool = True,
        overwrite: bool = True
    ) -> Callable:
        """
        Create a decorator that uses this manager to save function output.
        
        Args:
            filename: Filename to use (without directory). If None, use function name.
            include_args: Whether to include args in the auto-generated filename.
            include_timestamp: Whether to add timestamp to the auto-generated filename.
            add_metadata: Whether to save metadata about the function call.
            overwrite: Whether to overwrite existing files.
            
        Returns:
            A decorator function.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Get function result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Determine filename
                if filename:
                    output_filename = filename
                else:
                    output_filename = func.__name__
                    
                    if include_args:
                        args_str = '_'.join(str(arg)[:10] for arg in args)
                        kwargs_str = '_'.join(f"{k}_{v}"[:10] for k, v in kwargs.items())
                        if args_str or kwargs_str:
                            output_filename += f"_{args_str}_{kwargs_str}"
                    
                    if include_timestamp:
                        output_filename += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Clean filename
                output_filename = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in output_filename)
                if not output_filename.endswith('.pkl'):
                    output_filename += '.pkl'
                
                # Prepare metadata if requested
                metadata = None
                if add_metadata:
                    metadata = {
                        'function': func.__name__,
                        'args': str(args),
                        'kwargs': str(kwargs),
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Save the result
                self.save(result, output_filename, metadata, overwrite)
                
                return result
            
            return wrapper
        
        return decorator