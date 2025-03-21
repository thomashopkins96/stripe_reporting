from auth.auth import GoogleAuth
from sheets.sheets import GSheetsClient
import pandas as pd

def get_data_from_sheets(spreadsheet_id: str, range_name: str) -> pd.DataFrame:
    auth = GoogleAuth()
    client = GSheetsClient(auth)
    
    return client.get_as_dataframe(spreadsheet_id, range_name)