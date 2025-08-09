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
        print("DEBUG: DataFrame is None or empty")
        return [], []
    client_module = _load_sixtyfour_client_module()
    if client_module is None:
        print("DEBUG: SixtyFour client module not loaded")
        return [], []

    leads = get_dynamic_leads_for_api(dataframe)
    print(f"DEBUG: Generated {len(leads)} leads for API:")
    for i, lead in enumerate(leads[:3]):  # Print first 3 leads
        print(f"DEBUG: Lead {i+1}: {lead}")
    
    if not leads:
        print("DEBUG: No leads generated from dataframe")
        return [], []

    try:
        # Use env-configured API key inside client
        print("DEBUG: Calling SixtyFour API...")
        results = client_module.enrich_leads_with_env(leads)
        print(f"DEBUG: Received {len(results)} results from SixtyFour API")
    except Exception as e:
        print(f"DEBUG: Exception calling SixtyFour API: {e}")
        return [], []

    # Process raw enrichment results directly
    successful = []
    failed = []
    
    print("DEBUG: Processing enrichment results...")
    for i, result in enumerate(results):
        print(f"DEBUG: Result {i+1} type: {type(result)}")
        print(f"DEBUG: Result {i+1} has success attr: {hasattr(result, 'success')}")
        
        if not hasattr(result, 'success'):
            print(f"DEBUG: Result {i+1} skipped - no success attribute")
            continue
            
        print(f"DEBUG: Result {i+1} success: {result.success}")
        
        if result.success:
            print(f"DEBUG: Result {i+1} structured_data: {result.structured_data}")
            print(f"DEBUG: Result {i+1} original_lead: {result.original_lead}")
            
            # Create enriched lead object with all data
            enriched_lead = {
                "lead": result.original_lead or {},
                "structured_data": result.structured_data or {},
                "notes": result.notes,
                "findings": result.findings or [],
                "references": result.references or {},
                "confidence_score": result.confidence_score,
                "raw_response": result.raw_response
            }
            successful.append(enriched_lead)
            print(f"DEBUG: Added successful lead {i+1}")
        else:
            print(f"DEBUG: Result {i+1} failed with error: {result.error}")
            # Create failed lead object
            failed_lead = {
                "lead": result.original_lead or {},
                "error": result.error
            }
            failed.append(failed_lead)

    print(f"DEBUG: Final results - Successful: {len(successful)}, Failed: {len(failed)}")
    return successful, failed


def render_enrichment_section() -> Tuple[List[str], List[Dict[str, Any]]]:
    dataframe: Optional[pd.DataFrame] = st.session_state.get("dataframe")
    if dataframe is None:
        return [], []
    
    st.divider()
    st.subheader("Enrich lead data")
    client_module = _load_sixtyfour_client_module()
    csv_module = _load_csv_client_module()
    
    if client_module is None:
        st.error("SixtyFour client not available")
        return [], []
    
    if csv_module is None:
        st.error("CSV client not available")
        return [], []

    api_ready = False
    try:
        api_ready = bool(client_module.has_api_key_configured())
    except Exception:
        api_ready = False

    if not api_ready:
        st.info("SIXTYFOUR_API_KEY is not configured in the environment.")

    # Check if we have existing enriched data
    existing_phone_numbers = st.session_state.get("phone_numbers", [])
    existing_lead_data = st.session_state.get("complete_lead_data", [])
    
    if existing_phone_numbers and existing_lead_data:
        st.info(f"Using {len(existing_phone_numbers)} previously enriched leads")
        return existing_phone_numbers, existing_lead_data

    if st.button("Enrich lead data"):
        if dataframe is None or dataframe.empty:
            st.error("No data to process")
            return [], []
        if not api_ready:
            st.error("API key not configured. Set SIXTYFOUR_API_KEY in your environment.")
            return [], []
        
        with st.spinner("Enriching lead data..."):
            success_objs, failed_objs = enrich_leads_and_transform(dataframe)
        
        st.session_state["enrich_success_objects"] = success_objs
        st.session_state["enrich_failed_objects"] = failed_objs

        if not success_objs and not failed_objs:
            st.warning("No results returned")
            return [], []

        if success_objs:
            st.success(f"Enriched {len(success_objs)} leads")
            
            # Use CSV client to prepare data for calling
            try:
                phone_numbers, complete_lead_data = csv_module.prepare_lead_data_for_calling(success_objs)
                
                # Save to session state
                st.session_state["phone_numbers"] = phone_numbers
                st.session_state["complete_lead_data"] = complete_lead_data
                
                # Show preview
                st.write(f"**Found {len(phone_numbers)} leads with phone numbers:**")
                sample_to_show = complete_lead_data[:min(len(complete_lead_data), PREVIEW_ROWS)]
                st.json(sample_to_show)
                
                return phone_numbers, complete_lead_data
                
            except Exception as e:
                st.error(f"Error preparing lead data: {str(e)}")
                return [], []
        else:
            st.info("No successful enrichments")
            return [], []
    
    return [], []


def call_vapi_api(prompt: str, phone_numbers: List[str], lead_data: List[Dict[str, Any]]) -> None:
    """
    Call VAPI API with the provided prompt, phone numbers, and complete lead data.
    
    Args:
        prompt: The script/prompt for the agents to use during calls
        phone_numbers: List of phone numbers to call
        lead_data: Complete lead data including enriched information
    """
    if not prompt.strip():
        return
    
    if not phone_numbers:
        st.warning("No phone numbers available for calling")
        return
    
    if not lead_data:
        st.warning("No lead data available for calling")
        return
    
    st.divider()
    st.subheader("VAPI Call Results")
    st.info(f"Ready to call {len(phone_numbers)} leads with enriched data and provided prompt")
    
    # Show prompt
    st.write("**Call Script:**")
    st.text_area("Prompt", value=prompt, disabled=True, height=100)
    
    # Show lead data preview
    st.write("**Leads to Call:**")
    for i, lead in enumerate(lead_data, 1):
        with st.expander(f"Lead {i}: {lead.get('phone_number', 'No phone')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Data:**")
                original = lead.get("original_lead", {})
                if original:
                    st.json(original)
                else:
                    st.write("No original data")
            
            with col2:
                st.write("**Enriched Data:**")
                enriched = lead.get("enriched_data", {})
                if enriched:
                    st.json(enriched)
                else:
                    st.write("No enriched data")
            
            # Show additional enrichment info
            if lead.get("notes"):
                st.write("**Notes:**")
                st.write(lead["notes"])
            
            if lead.get("findings"):
                st.write("**Key Findings:**")
                for finding in lead["findings"]:
                    st.write(f"• {finding}")
    
    # Placeholder for actual VAPI implementation
    st.info("🚀 Ready to start calling! (VAPI integration pending)")
    # TODO: Implement actual VAPI API calls with complete lead data


def main() -> None:
    render_header()
    render_upload_section()
    render_file_info_and_preview()
    vapi_prompt = render_prompt_input()
    if vapi_prompt.strip():
        st.caption("Prompt saved.")
    phone_numbers, lead_data = render_enrichment_section()
    call_vapi_api(vapi_prompt, phone_numbers, lead_data)



if __name__ == "__main__":
    main()
