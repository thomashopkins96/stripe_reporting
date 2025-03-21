"""
Google Sheets Authentication Module

This module provides a modular authentication flow for accessing Google Sheets
using OAuth2 credentials or service account authentication.
"""

import os
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import pandas as pd
from auth.auth import GoogleAuth
from googleapiclient.discovery import Resource

class GSheetsClient:
    """
    A client for interacting with Google Sheets.
    Uses the GSheetsAuth class for authentication.
    """
    
    def __init__(self, auth: GoogleAuth = None, **auth_kwargs):
        """
        Initialize the GSheetsClient.
        
        Args:
            auth: A GSheetsAuth object for authentication
            **auth_kwargs: Arguments to create a new GSheetsAuth object if not provided
        """
        if auth is None:
            auth = GoogleAuth(**auth_kwargs)
        
        self.auth = auth
        self.service = None
    
    def connect(self) -> Resource:
        """
        Connect to the Google Sheets API.
        
        Returns:
            Google Sheets API service object
        """
        if self.service is None:
            self.service = self.auth.authenticate()
        return self.service
    
    def get_spreadsheet(self, spreadsheet_id: str) -> Dict[str, Any]:
        """
        Get information about a spreadsheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet to retrieve
            
        Returns:
            The spreadsheet information
        """
        if self.service is None:
            self.connect()
        
        return self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    
    def get_sheet_values(
        self,
        spreadsheet_id: str,
        range_name: str,
        value_render_option: str = 'FORMATTED_VALUE',
        date_time_render_option: str = 'FORMATTED_STRING'
    ) -> Dict[str, Any]:
        """
        Get values from a specific range in a spreadsheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_name: The A1 notation of the values to retrieve
            value_render_option: How values should be rendered
            date_time_render_option: How dates, times, and durations should be rendered
            
        Returns:
            The values in the specified range
        """
        if self.service is None:
            self.connect()
        
        return self.service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueRenderOption=value_render_option,
            dateTimeRenderOption=date_time_render_option
        ).execute()
        
    def get_all_values(
        self,
        spreadsheet_id: str,
        sheet_name: Optional[str] = None,
        value_render_option: str = 'FORMATTED_VALUE',
        date_time_render_option: str = 'FORMATTED_STRING'
    ) -> Dict[str, List[List[Any]]]:
        """
        Get all values from a spreadsheet or a specific sheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            sheet_name: If provided, get values only from this sheet,
                       otherwise get values from all sheets
            value_render_option: How values should be rendered
            date_time_render_option: How dates, times, and durations should be rendered
            
        Returns:
            Dictionary with sheet names as keys and their values as values
        """
        if self.service is None:
            self.connect()
        
        # Get spreadsheet info to know the sheet names
        spreadsheet_info = self.get_spreadsheet(spreadsheet_id)
        sheets = spreadsheet_info.get('sheets', [])
        sheet_names = [sheet['properties']['title'] for sheet in sheets]
        
        # If specific sheet requested, check if it exists
        if sheet_name is not None:
            if sheet_name not in sheet_names:
                raise ValueError(f"Sheet '{sheet_name}' not found in the spreadsheet")
            sheet_names = [sheet_name]
        
        # Get values from each sheet
        result = {}
        for name in sheet_names:
            try:
                range_name = f"'{name}'"  # Wrapping sheet name in quotes to handle special characters
                response = self.get_sheet_values(
                    spreadsheet_id,
                    range_name,
                    value_render_option,
                    date_time_render_option
                )
                result[name] = response.get('values', [])
            except Exception as e:
                result[name] = []
                print(f"Warning: Could not get values from sheet '{name}': {str(e)}")
        
        return result
    
    def get_as_dataframe(
        self,
        spreadsheet_id: str,
        range_name: str,
        has_header: bool = True,
        value_render_option: str = 'FORMATTED_VALUE',
        date_time_render_option: str = 'FORMATTED_STRING'
    ) -> pd.DataFrame:
        """
        Get data from a spreadsheet range as a pandas DataFrame.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_name: The A1 notation of the values to retrieve
            has_header: Whether the first row contains headers
            value_render_option: How values should be rendered
            date_time_render_option: How dates, times, and durations should be rendered
            
        Returns:
            A pandas DataFrame containing the data
        """
        response = self.get_sheet_values(
            spreadsheet_id,
            range_name,
            value_render_option,
            date_time_render_option
        )
        
        values = response.get('values', [])
        
        if not values:
            return pd.DataFrame()
        
        if has_header:
            headers = values[0]
            data = values[1:]
            return pd.DataFrame(data, columns=headers)
        else:
            return pd.DataFrame(values)
            
            
    def _get_unique_headers(self, headers):
        """
        Ensure all headers are unique and handle empty headers.
        
        Args:
            headers: List of header strings
            
        Returns:
            List of unique header strings
        """
        unique_headers = []
        header_counts = {}
        
        for i, header in enumerate(headers):
            # Handle empty header
            if header == '':
                unique_headers.append(f"Unnamed_{i}")
                continue
                
            # Handle duplicate headers
            if header in header_counts:
                header_counts[header] += 1
                unique_headers.append(f"{header}_{header_counts[header]}")
            else:
                header_counts[header] = 0
                unique_headers.append(header)
        
        return unique_headers
    
    def get_all_as_dataframes(
        self,
        spreadsheet_id: str,
        sheet_name: Optional[str] = None,
        has_header: bool = True,
        header_row: int = 0,
        data_start_row: Optional[int] = None,
        value_render_option: str = 'FORMATTED_VALUE',
        date_time_render_option: str = 'FORMATTED_STRING'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all data from a spreadsheet or a specific sheet as pandas DataFrames.
        
        Args:
            sp readsheet_id: The ID of the spreadsheet
            sheet_name: If provided, get values only from this sheet,
                       otherwise get values from allsheets
            has_header: Whether a specific row contains headers
            header_row: The index of the row containing headers (default is 0, the first row)
            data_start_row: The index of the first row of data (default is header_row + 1)
                           Set to a specific value to skip rows between header and data
            value_render_option: How values should be rendered
            date_time_render_option: How dates, times, and durations should be rendered
            
        Returns:
            Dictionary with sheet names as keys and DataFrames as values
        """
        all_values = self.get_all_values(
            spreadsheet_id,
            sheet_name,
            value_render_option,
            date_time_render_option
        )
        
        result = {}
        for sheet_name, values in all_values.items():
            if not values:
                result[sheet_name] = pd.DataFrame()
                continue
            
            # Set the default data start row if not specified
            if data_start_row is None:
                current_data_start_row = header_row + 1 if has_header else 0
            else:
                current_data_start_row = data_start_row
                
            # Validate indices to prevent out of range errors
            if header_row >= len(values) and has_header:
                print(f"Warning: Header row index {header_row} is out of range for sheet '{sheet_name}'. Skipping.")
                result[sheet_name] = pd.DataFrame()
                continue
                
            if current_data_start_row >= len(values):
                # Return empty DataFrame with headers if possible
                if has_header and header_row < len(values):
                    headers = values[header_row]
                    unique_headers = self._get_unique_headers(headers)
                    result[sheet_name] = pd.DataFrame(columns=unique_headers)
                else:
                    result[sheet_name] = pd.DataFrame()
                continue
            
            # Find the maximum row length to handle irregular data
            max_length = max(len(row) for row in values) if values else 0
            
            # Ensure all rows have the same length
            normalized_values = []
            for row in values:
                # Pad rows with empty strings if they're shorter than max_length
                normalized_values.append(row + [''] * (max_length - len(row)))
            
            if has_header:
                # Get headers from the specified header row
                headers = normalized_values[header_row]
                
                # Get data starting from the specified data start row
                data = normalized_values[current_data_start_row:]
                
                # Ensure headers are unique (pandas requirement)
                unique_headers = self._get_unique_headers(headers)
                
                result[sheet_name] = pd.DataFrame(data, columns=unique_headers)
            else:
                # If no header, just return data from the specified start row
                result[sheet_name] = pd.DataFrame(normalized_values[current_data_start_row:])
                
        return result
    
    def update_values(
        self,
        spreadsheet_id: str,
        range_name: str,
        values: List[List[Any]],
        value_input_option: str = 'RAW'
    ) -> Dict[str, Any]:
        """
        Update values in a specific range in a spreadsheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_name: The A1 notation of the values to update
            values: The new values to apply
            value_input_option: How the input data should be interpreted
            
        Returns:
            The update response
        """
        if self.service is None:
            self.connect()
        
        body = {
            'values': values
        }
        
        return self.service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            body=body
        ).execute()
    
    def append_values(
        self,
        spreadsheet_id: str,
        range_name: str,
        values: List[List[Any]],
        value_input_option: str = 'RAW',
        insert_data_option: str = 'INSERT_ROWS'
    ) -> Dict[str, Any]:
        """
        Append values to a specific range in a spreadsheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_name: The A1 notation of where to append values
            values: The values to append
            value_input_option: How the input data should be interpreted
            insert_data_option: How the input data should be inserted
            
        Returns:
            The append response
        """
        if self.service is None:
            self.connect()
        
        body = {
            'values': values
        }
        
        return self.service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption=value_input_option,
            insertDataOption=insert_data_option,
            body=body
        ).execute()
    
    def clear_values(
        self,
        spreadsheet_id: str,
        range_name: str
    ) -> Dict[str, Any]:
        """
        Clear values from a specific range in a spreadsheet.
        
        Args:
            spreadsheet_id: The ID of the spreadsheet
            range_name: The A1 notation of the values to clear
            
        Returns:
            The clear response
        """
        if self.service is None:
            self.connect()
        
        return self.service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            body={}
        ).execute()