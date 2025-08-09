"""
Streamlit app for CSV/Excel upload and preview.
Uses in-memory DataFrame handling (no file paths) and caching.
"""

from typing import Optional, Tuple, List, Dict, Any
import io
import os
import asyncio
import importlib.util
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


def render_prompt_input() -> str:
    st.divider()
    st.subheader("Call Prompt")
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

def render_file_info_and_preview() -> None:
    dataframe: Optional[pd.DataFrame] = st.session_state.get("dataframe")
    uploaded_name: Optional[str] = st.session_state.get("uploaded_name")

    if dataframe is None:
        return

    top_bar = st.columns([3, 1])
    with top_bar[0]:
        st.subheader("File loaded and read")
        st.info(
            f"**Name:** {uploaded_name} &nbsp; | &nbsp; "
            f"**Rows:** {len(dataframe):,} &nbsp; | &nbsp; "
            f"**Columns:** {len(dataframe.columns)}",
            icon="📋"
        )

    st.divider()
    st.subheader("Data preview")
    st.dataframe(dataframe.head(PREVIEW_ROWS), use_container_width=True, height=240)


def _load_csv_client_module():
    module_key = "_csv_client_module"
    if module_key in st.session_state:
        return st.session_state[module_key]
    
    base_dir = os.path.dirname(__file__)
    target_path = os.path.join(base_dir, "csv-client.py")
    if not os.path.exists(target_path):
        return None
    
    spec = importlib.util.spec_from_file_location("csv_client", target_path)
    if spec is None or spec.loader is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        return None
    
    st.session_state[module_key] = module
    return module


def dataframe_to_dynamic_objects_safe(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    if dataframe is None or dataframe.empty:
        return []
    module = _load_csv_client_module()
    if module is None:
        return []
    func = getattr(module, "dataframe_to_dynamic_objects", None)
    if func is None:
        return []
    try:
        result = func(dataframe)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _load_sixtyfour_client_module():
    module_key = "_sixtyfour_client_module"
    if module_key in st.session_state:
        return st.session_state[module_key]
    
    base_dir = os.path.dirname(__file__)
    target_path = os.path.join(base_dir, "sixtyfour-client.py")
    if not os.path.exists(target_path):
        return None
    
    spec = importlib.util.spec_from_file_location("sixtyfour_client", target_path)
    if spec is None or spec.loader is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        return None
    
    st.session_state[module_key] = module
    return module


def get_dynamic_leads_for_api(dataframe: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if dataframe is None or dataframe.empty:
        return []
    dynamic_objects = dataframe_to_dynamic_objects_safe(dataframe)
    return dynamic_objects if isinstance(dynamic_objects, list) else []


def enrich_leads_and_transform(dataframe: Optional[pd.DataFrame]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if dataframe is None or dataframe.empty:
        return [], []
    client_module = _load_sixtyfour_client_module()
    if client_module is None:
        return [], []

    leads = get_dynamic_leads_for_api(dataframe)
    if not leads:
        return [], []

    try:
        # Use env-configured API key inside client
        results = asyncio.run(client_module.enrich_leads_with_env(leads))
    except Exception:
        return [], []

    # Extract successes and failures using provided helpers
    try:
        successful = client_module.extract_successful_phones(results)  # List[dict] with 'phone'
    except Exception:
        successful = []
    try:
        failed = client_module.extract_failed_leads(results)  # List[dict] with 'error'
    except Exception:
        failed = []

    # Transform successful to desired shape: {"lead": {metadata...}, "phone_number": phone}
    transformed_success: List[Dict[str, Any]] = []
    for item in successful:
        if not isinstance(item, dict):
            continue
        phone_value = item.get("phone")
        # Prefer inner 'lead' if present to avoid double-wrapping
        inner_lead = item.get("lead") if isinstance(item.get("lead"), dict) else {k: v for k, v in item.items() if k != "phone"}
        transformed_success.append({
            "lead": inner_lead,
            "phone_number": phone_value,
        })

    return transformed_success, failed


def render_enrichment_section() -> None:
    dataframe: Optional[pd.DataFrame] = st.session_state.get("dataframe")
    if dataframe is None:
        return
    
    st.divider()
    st.subheader("Find phone numbers")
    client_module = _load_sixtyfour_client_module()
    if client_module is None:
        st.error("SixtyFour client not available")
        return

    api_ready = False
    try:
        api_ready = bool(client_module.has_api_key_configured())
    except Exception:
        api_ready = False

    if not api_ready:
        st.info("SIXTYFOUR_API_KEY is not configured in the environment.")

    if st.button("Enrich phone numbers"):
        if dataframe is None or dataframe.empty:
            st.error("No data to process")
            return
        if not api_ready:
            st.error("API key not configured. Set SIXTYFOUR_API_KEY in your environment.")
            return
        with st.spinner("Enriching phone numbers..."):
            success_objs, failed_objs = enrich_leads_and_transform(dataframe)
        st.session_state["enrich_success_objects"] = success_objs
        st.session_state["enrich_failed_objects"] = failed_objs

        if not success_objs and not failed_objs:
            st.warning("No results returned")
            return

        if success_objs:
            st.success(f"Enriched {len(success_objs)} leads")
            # Show a sample of the successful objects
            sample_to_show = success_objs[:min(len(success_objs), PREVIEW_ROWS)]
            st.json(sample_to_show)
        else:
            st.info("No successful enrichments")

### VAPI Pseudo code
# Inputs: 
# succesful leads gotten from extract succesful phones function
# prompt, provided by the user on the streamlit dashboard 
# 
# Processing:
# Call VAPI API with the prompt and the to phone number
# 
# Output: 
# Save the call outcome transcript to a list/object of all sucessful calls
# Show on streamlit

def _load_vapi_client_module():
    """Load VAPI client module dynamically"""
    module_key = "_vapi_client_module"
    if module_key in st.session_state:
        return st.session_state[module_key]
    
    base_dir = os.path.dirname(__file__)
    target_path = os.path.join(base_dir, "vapi-client.py")
    if not os.path.exists(target_path):
        return None
    
    spec = importlib.util.spec_from_file_location("vapi_client", target_path)
    if spec is None or spec.loader is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        return None
    
    st.session_state[module_key] = module
    return module


def make_vapi_calls(
    success_objs: List[Dict[str, Any]], 
    prompt: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Make VAPI calls to the enriched leads with phone numbers
    
    Args:
        success_objs: List of successful enrichment objects with phone numbers
        prompt: User-provided prompt for the AI agent
        
    Returns:
        Tuple of (successful_calls, failed_calls)
    """
    if not success_objs or not prompt.strip():
        return [], []
    
    client_module = _load_vapi_client_module()
    if client_module is None:
        return [], []
    
    # Check if VAPI API key is configured
    try:
        api_ready = bool(client_module.has_api_key_configured())
    except Exception:
        api_ready = False
    
    if not api_ready:
        return [], []
    
    # Prepare call requests
    call_requests = []
    for obj in success_objs:
        phone_number = obj.get("phone_number")
        if phone_number and isinstance(phone_number, str):
            call_requests.append({
                "phone_number": phone_number,
                "prompt": prompt
            })
    
    if not call_requests:
        return [], []
    
    try:
        # Make the calls using the VAPI client
        results = asyncio.run(client_module.make_outbound_calls_with_env(call_requests))
        
        # Extract successful and failed calls
        successful_calls = client_module.extract_successful_calls(results)
        failed_calls = client_module.extract_failed_calls(results)
        
        return successful_calls, failed_calls
        
    except Exception as e:
        st.error(f"Error making VAPI calls: {str(e)}")
        return [], []


def render_vapi_calling_section() -> None:
    """Render the VAPI calling section in the UI"""
    success_objs = st.session_state.get("enrich_success_objects", [])
    vapi_prompt = st.session_state.get("vapi_prompt", "")
    
    if not success_objs:
        return
    
    if not vapi_prompt.strip():
        st.info("Enter a call prompt above to enable VAPI calling")
        return
    
    st.divider()
    st.subheader("Make VAPI Calls")
    
    # Check VAPI client availability
    vapi_client = _load_vapi_client_module()
    if vapi_client is None:
        st.error("VAPI client not available")
        return
    
    # Check VAPI API key configuration
    try:
        vapi_api_ready = bool(vapi_client.has_api_key_configured())
    except Exception:
        vapi_api_ready = False
    
    if not vapi_api_ready:
        st.info("VAPI_API_TOKEN is not configured in the environment.")
        return
    
    if st.button("Start VAPI Calls"):
        if not success_objs:
            st.error("No enriched leads available for calling")
            return
        
        with st.spinner("Making VAPI calls..."):
            successful_calls, failed_calls = make_vapi_calls(success_objs, vapi_prompt)
        
        # Store results in session state
        st.session_state["vapi_successful_calls"] = successful_calls
        st.session_state["vapi_failed_calls"] = failed_calls
        
        # Display results
        if successful_calls:
            st.success(f"Successfully initiated {len(successful_calls)} calls")
            # Show call details
            for i, call in enumerate(successful_calls[:min(len(successful_calls), PREVIEW_ROWS)]):
                with st.expander(f"Call {i+1}: {call.get('phone_number', 'Unknown')}"):
                    st.json(call)
        else:
            st.info("No successful calls initiated")
        
        if failed_calls:
            st.warning(f"Failed to initiate {len(failed_calls)} calls")
            # Show failure details
            for i, call in enumerate(failed_calls[:min(len(failed_calls), PREVIEW_ROWS)]):
                with st.expander(f"Failed Call {i+1}: {call.get('phone_number', 'Unknown')}"):
                    st.json(call)
    
    # Display previous call results if available
    vapi_successful_calls = st.session_state.get("vapi_successful_calls", [])
    vapi_failed_calls = st.session_state.get("vapi_failed_calls", [])
    
    if vapi_successful_calls or vapi_failed_calls:
        st.divider()
        st.subheader("Call Results Summary")
        
        if vapi_successful_calls:
            st.success(f"Total Successful Calls: {len(vapi_successful_calls)}")
        
        if vapi_failed_calls:
            st.error(f"Total Failed Calls: {len(vapi_failed_calls)}")


def main() -> None:
    render_header()
    render_upload_section()
    render_file_info_and_preview()
    vapi_prompt = render_prompt_input()
    if vapi_prompt.strip():
        st.caption("Prompt saved.")
    render_enrichment_section()
    render_vapi_calling_section()


if __name__ == "__main__":
    main()
