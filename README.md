# CSV File Processor

A Flask-based web application for uploading and previewing CSV and Excel files with a clean, dark YC-style interface.

## Features

- **File Upload**: Support for CSV, Excel (.xlsx, .xls) files
- **Data Preview**: View the first 5 rows of uploaded data in a table format
- **File Management**: Easy file removal and replacement
- **Responsive Design**: Dark theme with hover effects and smooth transitions
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Drag & Drop**: Intuitive drag-and-drop file upload interface

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy
- **File Handling**: Werkzeug
- **Excel Support**: openpyxl

## Project Structure

```
EF/
├── app.py                 # Flask application
├── csv-client.py          # DataFrame processing utilities
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Dark YC-style CSS
│   └── js/
│       └── main.js       # Frontend JavaScript
└── uploads/              # File upload directory (auto-created)
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

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload a CSV or Excel file using the drag-and-drop interface or file picker

4. View the data preview showing the first 5 rows

5. Remove files when no longer needed

## API Endpoints

- `GET /` - Main application page
- `POST /api/upload` - Upload file and get preview
- `POST /api/remove` - Remove uploaded file
- `GET /api/status` - Check current file status
- `GET /api/dataframe` - Get complete DataFrame data

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

## Security Features

- File type validation
- Secure filename handling
- File size limits (16MB max)
- Temporary file cleanup
