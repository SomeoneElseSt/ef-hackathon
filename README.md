# Hire Line Data Uploader & Enricher

A Streamlit-based app for uploading CSV and Excel files, previewing data, and performing lead enrichment and calling workflows entirely in memory.

## Features

- **Upload CSV/Excel**: Drag-and-drop or file picker support for `.csv`, `.xlsx`, and `.xls` files.
- **In-Memory Processing**: All parsing and data handling occur in memory with caching for repeated uploads.
- **Data Preview**: See the first few rows of your uploaded data in a table.
- **File Management**: Quick removal and replacement of the current dataset.
- **AI-Generated Prompts** (optional): Generate recruitment prompts using a configured OpenAI API key.
- **Lead Extraction & Enrichment**: Extract leads from your data and enrich them via the SixtyFour API (if configured).
- **VAPI Calls (Preview)**: Prepare and preview outbound AI-assisted calls with a user-provided prompt.
- **Dynamic Lead Transformation**: Convert raw CSV data into structured lead objects suitable for downstream workflows.
- **Security & Validation**: Validate file types and enforce a maximum upload size; all data remains in memory.

## Technical Stack

- **App**: `Streamlit` (Python)
- **Data Processing**: `pandas`, `numpy`
- **Excel Support**: `openpyxl` / `xlrd` (via pandas)
- **Environment**: `python-dotenv` for environment variables
- **Auxiliary Clients**: `chatgpt-client.py`, `csv-client.py`, `sixtyfour-client.py`, `vapi-client.py` (dynamically loaded)

## Project Structure

```
EF/
├── app.py                 # Streamlit application (main UI and flow)
├── csv-client.py          # DataFrame utilities and exports (optional)
├── sixtyfour-client.py     # Lead enrichment client (SixtyFour API integration)
├── vapi-client.py          # VAPI integration client
├── chatgpt-client.py       # ChatGPT integration (prompt generation)
├── requirements.txt        # Python dependencies
```

Notes: In this repo, the app is run from `app.py` and handles uploads, previews, prompt generation, enrichment, and VAPI preview calls.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py --server.port 8501
   ```

3. Open your browser and navigate to:
   `http://localhost:8501`

## How to Use

- Upload a CSV or Excel file via drag-and-drop or file picker.
- The UI shows a preview of the data (first 5 rows) and metadata (rows/columns).
- If OpenAI API keys are configured, you can generate an agent prompt for later use.
- Enrich leads by triggering the SixtyFour API (if configured) to obtain structured data.
- Prepare VAPI outbound calls by supplying a prompt and reviewing the listed leads.

All actions are designed to work in-memory for speed and to avoid disk I/O unless exporting data is explicitly implemented.

## Environment & Security

- File type validation: CSV, XLSX, XLS
- Upload size guard: 16 MB maximum per file
- OpenAI, SixtyFour, and VAPI credentials are read from environment variables when configured (via `python-dotenv` or your shell).


