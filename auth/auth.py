import os
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import pandas as pd
from google.oauth2.service_account import Credentials as ServiceAccountCreds
from google.oauth2.credentials import Credentials as OAuthCreds
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build, Resource
from google.auth.exceptions import RefreshError


class GoogleAuth:
    """
    A class that handles authentication with Google Sheets API.
    Supports both OAuth2 and Service Account authentication methods.
    """
    
    # Define the scope needed for Google Sheets access
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    
    def __init__(
        self, 
        credentials_path: str = None,
        token_path: str = None,
        use_service_account: bool = False
    ):
        """
        Initialize the GSheetsAuth object.
        
        Args:
            credentials_path: Path to the credentials file (OAuth client_secret.json or service_account.json)
            token_path: Path to save/load the OAuth token (for user auth only)
            use_service_account: Whether to use service account auth instead of OAuth
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.use_service_account = use_service_account
        self.credentials = None
        self.service = None
    
    def authenticate(self) -> Resource:
        """
        Authenticate with Google Sheets API and return the service object.
        
        Returns:
            Google Sheets API service object
        
        Raises:
            FileNotFoundError: If credentials file is not found
            ValueError: If authentication fails
        """
        if self.use_service_account:
            return self._authenticate_service_account()
        else:
            return self._authenticate_oauth()
    
    def _authenticate_service_account(self) -> Resource:
        """
        Authenticate using a service account.
        
        Returns:
            Google Sheets API service object
        
        Raises:
            FileNotFoundError: If service account file is not found
            ValueError: If authentication fails
        """
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Service account credentials file not found at: {self.credentials_path}")
        
        try:
            self.credentials = ServiceAccountCreds.from_service_account_file(
                self.credentials_path, scopes=self.SCOPES
            )
            self.service = build('sheets', 'v4', credentials=self.credentials)
            return self.service
        except Exception as e:
            raise ValueError(f"Failed to authenticate with service account: {str(e)}")
    
    def _authenticate_oauth(self) -> Resource:
        """
        Authenticate using OAuth2 flow.
        
        Returns:
            Google Sheets API service object
        
        Raises:
            FileNotFoundError: If client secrets file is not found
            ValueError: If authentication fails
        """
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"OAuth client secrets file not found at: {self.credentials_path}")
        
        if not self.token_path:
            self.token_path = os.path.join(os.path.dirname(self.credentials_path), 'token.json')
        
        try:
            # Check if we have a token file and try to load credentials
            if os.path.exists(self.token_path):
                self.credentials = OAuthCreds.from_authorized_user_info(
                    json.loads(Path(self.token_path).read_text()), 
                    self.SCOPES
                )
            
            # If there are no valid credentials available, go through the OAuth flow
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    try:
                        self.credentials.refresh(Request())
                    except RefreshError:
                        # If refresh fails, go through the flow again
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, self.SCOPES
                        )
                        self.credentials = flow.run_local_server(port=0)
                else:
                    # No valid credentials, start fresh
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.SCOPES
                    )
                    self.credentials = flow.run_local_server(port=0)
                
                # Save the credentials for next run
                with open(self.token_path, 'w') as token:
                    token.write(self.credentials.to_json())
            
            self.service = build('sheets', 'v4', credentials=self.credentials)
            return self.service
        
        except Exception as e:
            raise ValueError(f"Failed to authenticate with OAuth: {str(e)}")