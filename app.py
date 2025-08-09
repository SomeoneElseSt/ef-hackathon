"""
Streamlit app for CSV/Excel upload and preview.
Uses in-memory DataFrame handling (no file paths) and caching.
"""

from typing import Optional, Tuple
import io
import pandas as pd
import streamlit as st

# Constants
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
PREVIEW_ROWS = 5
MAX_UPLOAD_MB = 16


def get_extension_from_filename(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def is_allowed_extension(extension: str) -> bool:
    return extension in ALLOWED_EXTENSIONS


@st.cache_data(show_spinner=False)
def parse_dataframe_cached(file_bytes: bytes, extension: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    if not file_bytes or not extension:
        return False, None, "Invalid file input"

    try:
        if extension == "csv":
            dataframe = pd.read_csv(io.BytesIO(file_bytes))
            return True, dataframe, None

        if extension in ("xlsx", "xls"):
            dataframe = pd.read_excel(io.BytesIO(file_bytes))
            return True, dataframe, None

        return False, None, "Unsupported file type"
    except Exception as e:
        # Return error as value instead of raising
        return False, None, f"Could not read file: {str(e)}"


def load_dataframe_from_upload(uploaded_file) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    if uploaded_file is None:
        return False, None, "No file provided"

    filename = uploaded_file.name
    extension = get_extension_from_filename(filename)

    if not is_allowed_extension(extension):
        return False, None, "File type not allowed (use CSV or Excel)"

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        return False, None, "Empty file"

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return False, None, f"File too large. Max {MAX_UPLOAD_MB}MB"

    return parse_dataframe_cached(file_bytes, extension)


def render_header() -> None:
    st.set_page_config(page_title="Rip em' if you got 'em!", page_icon="📁", layout="centered")
    st.title("Rip em' if you got 'em 📞 !")
    st.caption("Upload your sales list .csv data. We'll find your list's phone numbers and call them with a script you provide.")


def render_upload_section() -> None:
    uploaded_file = st.file_uploader(
        label="Upload your file",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        help="Supported: .csv, .xlsx, .xls",
    )

    if uploaded_file is None:
        return

    success, dataframe, error_message = load_dataframe_from_upload(uploaded_file)

    if not success or dataframe is None:
        st.error(error_message or "Upload failed")
        return

    st.session_state["uploaded_name"] = uploaded_file.name
    st.session_state["dataframe"] = dataframe


def render_file_info_and_preview() -> None:
    dataframe: Optional[pd.DataFrame] = st.session_state.get("dataframe")
    uploaded_name: Optional[str] = st.session_state.get("uploaded_name")

    if dataframe is None:
        return

    top_bar = st.columns([3, 1])
    with top_bar[0]:
        st.subheader("File loaded")
        st.write(f"**Name**: {uploaded_name}")
        st.write(f"**Rows**: {len(dataframe):,}")
        st.write(f"**Columns**: {len(dataframe.columns)}")
    with top_bar[1]:
        if st.button("Remove file"):
            st.session_state.pop("dataframe", None)
            st.session_state.pop("uploaded_name", None)
            st.cache_data.clear()
            st.experimental_rerun()

    st.divider()
    st.subheader("Data preview")
    st.dataframe(dataframe.head(PREVIEW_ROWS), use_container_width=True, height=240)


def render_prompt_input() -> str:
    st.divider()
    st.subheader("Call script prompt")
    default_value = st.session_state.get("vapi_prompt", "")
    vapi_prompt = st.text_area(
        label="Enter the prompt for the agents to call your list with!",
        value=default_value,
        placeholder="Your script goes here...",
        help="Saved locally in session. Used later for calling.",
        height=160,
    )
    st.session_state["vapi_prompt"] = vapi_prompt
    return vapi_prompt


def main() -> None:
    render_header()
    render_upload_section()
    render_file_info_and_preview()
    vapi_prompt = render_prompt_input()
    if vapi_prompt.strip():
        st.caption("Prompt saved.")


if __name__ == "__main__":
    main()
