# CSV File Processor

A Streamlit-based app for uploading and previewing CSV and Excel files with a clean, simple interface.

## Features

- **File Upload**: Support for CSV, Excel (.xlsx, .xls) files
- **Data Preview**: View the first 5 rows of uploaded data in a table format
- **File Management**: Easy file removal and replacement
- **Responsive Design**: Dark theme with hover effects and smooth transitions
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Drag & Drop**: Intuitive drag-and-drop file upload interface

## Technical Stack

- **App**: Streamlit (Python)
- **Data Processing**: Pandas, NumPy
- **Excel Support**: openpyxl/xlrd

## Project Structure

```
EF/
├── app.py                 # Streamlit application
├── csv-client.py          # Optional DataFrame utilities
├── requirements.txt       # Python dependencies
```

## Installation

1. Install dependencies using uv (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py --server.port 8501
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

3. Upload a CSV or Excel file using the drag-and-drop interface or file picker

4. View the data preview showing the first 5 rows

5. Remove files when no longer needed

## Notes

- Files are not written to disk. Everything is processed in-memory.
- Caching is enabled for parsing to speed up repeated uploads of the same file.

## Code Guidelines Followed

1. **Explicit Error Handling**: Early returns and continues preferred over try/catch blocks
2. **Flat Code Structure**: Avoided nested ifs and arrow-shaped functions
3. **Separation of Concerns**: Dedicated functions for each step, separate client files
4. **Package Management**: UV recommended for Python dependencies
5. **English Variable Names**: All variables and constants named in English

## File Processing Features

The `csv-client.py` module provides:
- DataFrame loading and validation
- Data preview generation
- Column information and statistics
- Data cleaning utilities
- Export functionality

## Browser Support

- Modern browsers with ES6+ support
- Responsive design for mobile and desktop
- Drag-and-drop API support

## Security

- File type validation (CSV/XLS/XLSX)
- Upload size guard (16MB)
