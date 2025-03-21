import numpy as np
import re
import os
from typing import Dict, Optional, List

def convert_float_str_to_array(string: str) -> np.array:
    if string.startswith('[') and string.endswith(']'):
        string = string[1:-1]
        
    return np.array(string.split(','), dtype=float)
        
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