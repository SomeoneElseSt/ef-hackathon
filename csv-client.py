"""
CSV Client for DataFrame processing and analysis.
Handles DataFrame operations with proper error handling and separation of concerns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


def normalize_phone_number(phone_number: str) -> str:
    """
    Normalizes phone number to E.164 format with +1 prefix for US numbers
    
    Args:
        phone_number: Raw phone number string
        
    Returns:
        Normalized phone number with +1 prefix
    """
    if not phone_number:
        return ""
    
    # Remove all non-digit characters
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    if not digits_only:
        return ""
    
    # Handle different US phone number formats
    if len(digits_only) == 10:
        # 10 digits - assume US number, add +1
        return f"+1{digits_only}"
    elif len(digits_only) == 11 and digits_only.startswith('1'):
        # 11 digits starting with 1 - US number with country code
        return f"+{digits_only}"
    elif len(digits_only) == 11 and not digits_only.startswith('1'):
        # 11 digits not starting with 1 - assume US number, add +1
        return f"+1{digits_only}"
    elif len(digits_only) > 11:
        # More than 11 digits - assume already has country code, add +
        return f"+{digits_only}"
    else:
        # Less than 10 digits - invalid
        return ""

# Constants
DEFAULT_PREVIEW_ROWS = 5
MAX_DISPLAY_COLUMNS = 20
NUMERIC_TYPES = ['int64', 'float64', 'int32', 'float32']
TEXT_TYPES = ['object', 'string']

class CSVProcessor:
    """Handles CSV file processing and DataFrame operations"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filepath: Optional[str] = None
        
    def load_file(self, filepath: str) -> bool:
        """
        Load CSV or Excel file into DataFrame
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        if not Path(filepath).exists():
            return False
            
        try:
            if filepath.endswith('.csv'):
                self.df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(filepath)
            else:
                return False
                
            self.filepath = filepath
            return True
            
        except Exception:
            self.df = None
            self.filepath = None
            return False
    
    def get_preview(self, rows: int = DEFAULT_PREVIEW_ROWS) -> Optional[List[Dict[str, Any]]]:
        """
        Get preview of DataFrame as list of dictionaries
        
        Args:
            rows: Number of rows to preview
            
        Returns:
            List of dictionaries representing rows, or None if no data
        """
        if self.df is None:
            return None
            
        preview_df = self.df.head(rows)
        return preview_df.to_dict('records')
    
    def get_column_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about DataFrame columns
        
        Returns:
            Dictionary with column information, or None if no data
        """
        if self.df is None:
            return None
            
        column_info = {}
        
        for col in self.df.columns:
            dtype_str = str(self.df[col].dtype)
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()
            
            col_info = {
                'dtype': dtype_str,
                'null_count': int(null_count),
                'unique_count': int(unique_count),
                'is_numeric': dtype_str in NUMERIC_TYPES,
                'is_text': dtype_str in TEXT_TYPES
            }
            
            # Add sample values for text columns
            if col_info['is_text'] and unique_count > 0:
                sample_values = self.df[col].dropna().unique()[:5].tolist()
                col_info['sample_values'] = sample_values
            
            column_info[col] = col_info
            
        return column_info
    
    def get_basic_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get basic statistics about the DataFrame
        
        Returns:
            Dictionary with basic stats, or None if no data
        """
        if self.df is None:
            return None
            
        return {
            'shape': self.df.shape,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'has_duplicates': self.df.duplicated().any(),
            'duplicate_count': self.df.duplicated().sum()
        }
    
    def clean_data(self, remove_duplicates: bool = False, fill_na: bool = False) -> bool:
        """
        Clean DataFrame data
        
        Args:
            remove_duplicates: Whether to remove duplicate rows
            fill_na: Whether to fill NA values
            
        Returns:
            True if successful, False otherwise
        """
        if self.df is None:
            return False
            
        try:
            if remove_duplicates:
                self.df = self.df.drop_duplicates()
                
            if fill_na:
                # Fill numeric columns with median
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                
                # Fill text columns with empty string
                text_cols = self.df.select_dtypes(include=['object']).columns
                for col in text_cols:
                    self.df[col] = self.df[col].fillna('')
                    
            return True
            
        except Exception:
            return False
    
    def filter_data(self, column: str, value: Any, operation: str = 'equals') -> bool:
        """
        Filter DataFrame based on column value
        
        Args:
            column: Column name to filter on
            value: Value to filter by
            operation: Filter operation ('equals', 'contains', 'greater', 'less')
            
        Returns:
            True if successful, False otherwise
        """
        if self.df is None or column not in self.df.columns:
            return False
            
        try:
            if operation == 'equals':
                self.df = self.df[self.df[column] == value]
            elif operation == 'contains' and self.df[column].dtype == 'object':
                self.df = self.df[self.df[column].str.contains(str(value), na=False)]
            elif operation == 'greater' and pd.api.types.is_numeric_dtype(self.df[column]):
                self.df = self.df[self.df[column] > value]
            elif operation == 'less' and pd.api.types.is_numeric_dtype(self.df[column]):
                self.df = self.df[self.df[column] < value]
            else:
                return False
                
            return True
            
        except Exception:
            return False
    
    def export_to_csv(self, output_path: str) -> bool:
        """
        Export DataFrame to CSV
        
        Args:
            output_path: Path for output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        if self.df is None:
            return False
            
        try:
            self.df.to_csv(output_path, index=False)
            return True
        except Exception:
            return False

def load_and_preview(filepath: str, preview_rows: int = DEFAULT_PREVIEW_ROWS) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Load file and return preview with basic info
    
    Args:
        filepath: Path to the CSV/Excel file
        preview_rows: Number of rows to preview
        
    Returns:
        Tuple of (preview_data, basic_stats) or None if failed
    """
    processor = CSVProcessor()
    
    if not processor.load_file(filepath):
        return None
        
    preview = processor.get_preview(preview_rows)
    stats = processor.get_basic_stats()
    
    if preview is None or stats is None:
        return None
        
    return preview, stats

# --- Dynamic row object utilities ---

def is_non_empty_cell_value(value: Any) -> bool:
    """Return True if the cell value is considered non-empty."""
    if value is None:
        return False
    
    # Handles NaN, NaT, None-like across numpy/pandas
    if pd.isna(value):
        return False
    
    if isinstance(value, str):
        return bool(value.strip())
    
    # Numbers, booleans and other non-empty objects
    return True if isinstance(value, (int, float, bool)) else bool(value)


def normalize_value_for_json(value: Any) -> Any:
    """Normalize pandas/numpy/scalar values for JSON serialization."""
    if value is None or pd.isna(value):
        return None
    
    try:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:
        pass
    
    # Format timestamps
    if hasattr(value, 'isoformat'):
        try:
            return value.isoformat()  # pandas.Timestamp, datetime, etc.
        except Exception:
            return str(value)
    
    if isinstance(value, str):
        return value.strip()
    
    return value


def dataframe_to_dynamic_objects(dataframe: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame into a list of dynamic JSON-like dicts wrapped by 'lead'.
    - For each row, include only columns whose values are non-empty.
    - Keys are the original column names; values are normalized for JSON.
    - Each resulting row object is wrapped as {"lead": row_object}.
    
    Args:
        dataframe: Input pandas DataFrame
    
    Returns:
        List of dictionaries, one per row, each under a 'lead' key.
    """
    if dataframe is None or dataframe.empty:
        return []
    
    row_dicts: List[Dict[str, Any]] = []
    records = dataframe.to_dict(orient='records')
    
    for record in records:
        dynamic_row: Dict[str, Any] = {}
        for column_name, raw_value in record.items():
            if not is_non_empty_cell_value(raw_value):
                continue
            dynamic_row[str(column_name)] = normalize_value_for_json(raw_value)
        if dynamic_row:
            row_dicts.append({"lead": dynamic_row})
    
    return row_dicts


def extract_phone_numbers_from_leads(lead_data: List[Dict[str, Any]]) -> List[str]:
    """
    Extract and normalize phone numbers from enriched lead data.
    
    Args:
        lead_data: List of enriched lead dictionaries
    
    Returns:
        List of normalized phone numbers as strings (E.164 format with +1)
    """
    phone_numbers = []
    
    for lead_item in lead_data:
        if not isinstance(lead_item, dict):
            continue
            
        raw_phone = None
        
        # Check for phone in structured_data (from enrichment)
        structured_data = lead_item.get("structured_data", {})
        if isinstance(structured_data, dict):
            phone = structured_data.get("phone")
            if phone and str(phone).strip():
                raw_phone = str(phone).strip()
        
        # Check for phone in lead data (from CSV) if not found in structured_data
        if not raw_phone:
            lead = lead_item.get("lead", {})
            if isinstance(lead, dict):
                phone = lead.get("phone") or lead.get("Phone") or lead.get("phone_number") or lead.get("Phone Number")
                if phone and str(phone).strip():
                    raw_phone = str(phone).strip()
        
        # Check top-level phone field (backward compatibility) if still not found
        if not raw_phone:
            phone = lead_item.get("phone") or lead_item.get("Phone") or lead_item.get("phone_number")
            if phone and str(phone).strip():
                raw_phone = str(phone).strip()
        
        # Normalize the phone number if we found one
        if raw_phone:
            normalized_phone = normalize_phone_number(raw_phone)
            if normalized_phone:  # Only add if normalization was successful
                phone_numbers.append(normalized_phone)
    
    return phone_numbers


def prepare_lead_data_for_calling(enriched_results: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Prepare lead data for VAPI calling by extracting phone numbers and organizing complete lead data.
    
    Args:
        enriched_results: List of enriched lead results from SixtyFour API
    
    Returns:
        Tuple of (phone_numbers, complete_lead_data)
    """
    print(f"DEBUG CSV: Preparing {len(enriched_results)} enriched results for calling")
    phone_numbers = []
    complete_lead_data = []
    
    for i, result in enumerate(enriched_results):
        print(f"DEBUG CSV: Processing result {i+1}: {type(result)}")
        print(f"DEBUG CSV: Result {i+1} keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if not isinstance(result, dict):
            print(f"DEBUG CSV: Result {i+1} skipped - not a dict")
            continue
        
        # Extract phone number
        phone = None
        
        # Priority 1: From structured_data (enriched)
        structured_data = result.get("structured_data", {})
        print(f"DEBUG CSV: Result {i+1} structured_data: {structured_data}")
        if isinstance(structured_data, dict):
            phone = structured_data.get("phone")
            if phone:
                print(f"DEBUG CSV: Result {i+1} found phone in structured_data: {phone}")
        
        # Priority 2: From original lead data (CSV)
        if not phone:
            lead = result.get("lead", {})
            print(f"DEBUG CSV: Result {i+1} lead data: {lead}")
            if isinstance(lead, dict):
                phone = (lead.get("phone") or lead.get("Phone") or 
                        lead.get("phone_number") or lead.get("Phone Number"))
                if phone:
                    print(f"DEBUG CSV: Result {i+1} found phone in lead data: {phone}")
        
        # Priority 3: Top-level phone field
        if not phone:
            phone = (result.get("phone") or result.get("Phone") or 
                    result.get("phone_number"))
            if phone:
                print(f"DEBUG CSV: Result {i+1} found phone at top level: {phone}")
        
        print(f"DEBUG CSV: Result {i+1} final phone: {phone}")
        
        # Only include leads with valid phone numbers
        if phone and str(phone).strip():
            phone_str = str(phone).strip()
            phone_numbers.append(phone_str)
            
            # Create complete lead data object
            lead_data_item = {
                "phone_number": phone_str,
                "original_lead": result.get("lead", {}),
                "enriched_data": structured_data,
                "notes": result.get("notes"),
                "findings": result.get("findings", []),
                "references": result.get("references", {}),
                "confidence_score": result.get("confidence_score")
            }
            complete_lead_data.append(lead_data_item)
            print(f"DEBUG CSV: Added lead {i+1} with phone {phone_str}")
        else:
            print(f"DEBUG CSV: Result {i+1} skipped - no valid phone number")
    
    print(f"DEBUG CSV: Final results - {len(phone_numbers)} phone numbers, {len(complete_lead_data)} complete leads")
    return phone_numbers, complete_lead_data
