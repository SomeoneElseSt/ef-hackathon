"""
CSV Client for DataFrame processing and analysis.
Handles DataFrame operations with proper error handling and separation of concerns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

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
