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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    st.set_page_config(page_title="Hire Line 👨‍💻👩‍💻", page_icon="📁", layout="centered")
    st.title("Hire Line 👨‍💻👩‍💻")
    st.caption("Upload your hiring prospects list .csv data with a phone number and we'll call them to screen them with ascript you provide.")


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


def _load_chatgpt_client_module():
    """Load ChatGPT client module dynamically"""
    module_key = "_chatgpt_client_module"
    if module_key in st.session_state:
        return st.session_state[module_key]
    
    base_dir = os.path.dirname(__file__)
    target_path = os.path.join(base_dir, "chatgpt-client.py")
    if not os.path.exists(target_path):
        return None
    
    spec = importlib.util.spec_from_file_location("chatgpt_client", target_path)
    if spec is None or spec.loader is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        return None
    
    st.session_state[module_key] = module
    return module


def render_prompt_input() -> str:
    st.divider()
    st.subheader("Call Prompt")
    
    # Generate prompt button
    dataframe: Optional[pd.DataFrame] = st.session_state.get("dataframe")
    if dataframe is not None and not dataframe.empty:
        chatgpt_module = _load_chatgpt_client_module()
        
        if chatgpt_module and chatgpt_module.has_openai_key_configured():
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🤖 Generate Prompt", type="secondary", help="Use ChatGPT to generate a recruitment prompt based on your CSV data"):
                    with st.spinner("Generating prompt with ChatGPT..."):
                        try:
                            result = chatgpt_module.generate_prompt_with_env(dataframe)
                            # Save candidate fitness list if present (simple text)
                            try:
                                fitness_text = getattr(result, "candidate_fitness_list", None)
                                st.session_state["candidate_fitness_list"] = (fitness_text or "").strip()
                                st.session_state["fitness_tokens_used"] = getattr(result, "fitness_tokens_used", None)
                            except Exception:
                                st.session_state["candidate_fitness_list"] = ""
                                st.session_state["fitness_tokens_used"] = None

                            if result.success and result.generated_prompt:
                                st.session_state["vapi_prompt"] = result.generated_prompt
                                st.session_state["prompt_generated"] = True
                                st.session_state["prompt_tokens_used"] = result.tokens_used
                                st.success(f"✅ Prompt generated! ({result.tokens_used} tokens used)")
                            else:
                                st.error(f"Failed to generate prompt: {result.error}")
                        except Exception as e:
                            st.error(f"Error generating prompt: {str(e)}")
        else:
            if not chatgpt_module:
                st.info("💡 ChatGPT client not available for prompt generation")
            else:
                st.info("💡 Set OPENAI_API_KEY in environment to enable AI prompt generation")
    
    # Show generation status if prompt was just generated
    if st.session_state.get("prompt_generated", False):
        tokens_used = st.session_state.get("prompt_tokens_used", 0)
        st.info(f"✨ AI-generated recruitment prompt ready! ({tokens_used} tokens used)")
        st.session_state["prompt_generated"] = False  # Reset the flag
    
    # Prompt input
    default_value = st.session_state.get("vapi_prompt", "")
    vapi_prompt = st.text_area(
        label="Enter the prompt for the agents to call your list with!",
        value=default_value,
        placeholder="Your recruitment script goes here... (or use 'Generate Prompt' button above)",
        help="Saved locally in session. Used later for calling.",
        height=160,
    )
    st.session_state["vapi_prompt"] = vapi_prompt
    
    # Candidate fitness list (cards by default for readability)
    fitness_text = (st.session_state.get("candidate_fitness_list") or "").strip()
    if fitness_text:
        st.markdown("**Candidate fitness (inferred role fit):**")
        lines = [ln for ln in fitness_text.splitlines() if ln.strip()]
        # Always render as Cards by default
        items = []
        for line in lines:
            txt = line.strip()
            if txt.startswith("- "):
                txt = txt[2:]
            parts = [p.strip() for p in txt.split('—')]
            if len(parts) >= 3:
                items.append({"name": parts[0], "fit": parts[1], "note": '—'.join(parts[2:]).strip()})
            else:
                items.append({"name": txt, "fit": "", "note": ""})

        if items:
            for item in items:
                emoji = ""
                fval = item.get("fit", "").lower()
                if "high" in fval:
                    emoji = "🟢"
                elif "medium" in fval:
                    emoji = "🟡"
                elif "low" in fval:
                    emoji = "🔴"
                name = item.get("name", "")
                note = item.get("note", "")
                cols = st.columns([3, 1, 6])
                with cols[0]:
                    st.markdown(f"**{name}**")
                with cols[1]:
                    st.markdown(f"**{emoji} {item.get('fit','')}**" if item.get('fit') else "")
                with cols[2]:
                    if isinstance(note, str) and note.strip():
                        st.markdown(note)
    
    # First message input
    st.write("**First Message (Optional)**")
    default_first_message = st.session_state.get("vapi_first_message", "")
    vapi_first_message = st.text_input(
        label="What should the agent say first when the call connects?",
        value=default_first_message,
        placeholder="e.g., 'Hi, is this [Name]? I'm calling from [Company]...'",
        help="This will be the opening line when the call connects. Leave empty to use default.",
        label_visibility="collapsed"
    )
    st.session_state["vapi_first_message"] = vapi_first_message
    
    return vapi_prompt

    # Note: Additional sections are rendered below after returning the prompt

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
    st.subheader("Extract & Enrich Lead Data")
    csv_module = _load_csv_client_module()
    
    if csv_module is None:
        st.error("CSV client not available")
        return [], []

    # Check if we have existing enriched data
    existing_phone_numbers = st.session_state.get("phone_numbers", [])
    existing_lead_data = st.session_state.get("complete_lead_data", [])
    
    if existing_phone_numbers and existing_lead_data:
        st.info(f"Using {len(existing_phone_numbers)} previously processed leads with enriched data")
        # Always keep the preview visible even when subsequent actions (like VAPI calls) run
        st.write("**Sample enriched leads:**")
        sample_to_show = existing_lead_data[:min(len(existing_lead_data), 3)]
        for i, lead in enumerate(sample_to_show, 1):
            with st.expander(f"Lead {i}: {lead.get('phone_number', 'No phone')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Data:**")
                    st.json(lead.get("original_lead", {}))
                with col2:
                    st.write("**Enriched Data:**")
                    if lead.get("structured_data"):
                        st.json(lead.get("structured_data", {}))
                    else:
                        st.write("No enrichment available")
        return existing_phone_numbers, existing_lead_data

    if st.button("Extract & Enrich Lead Data", type="primary"):
        if dataframe is None or dataframe.empty:
            st.error("No data to process")
            return [], []
        
        # Step 1: Extract basic lead data from CSV
        with st.spinner("Step 1/2: Extracting lead data from CSV..."):
            # Convert dataframe to dynamic objects first
            dynamic_objects = dataframe_to_dynamic_objects_safe(dataframe)
            
            if not dynamic_objects:
                st.error("Failed to process CSV data")
                return [], []
            
            # Extract phone numbers directly from CSV data
            phone_numbers = csv_module.extract_phone_numbers_from_leads(dynamic_objects)
            
            if not phone_numbers:
                st.warning("No phone numbers found in the CSV data")
                return [], []
            
            st.success(f"✅ Extracted {len(phone_numbers)} leads with phone numbers")
        
        # Step 2: Enrich leads with SixtyFour API
        with st.spinner("Step 2/2: Enriching leads with SixtyFour AI..."):
            try:
                successful_enrichments, failed_enrichments = enrich_leads_and_transform(dataframe)
                
                if not successful_enrichments and not failed_enrichments:
                    st.warning("No enrichment results received")
                    return [], []
                
                # Process successful enrichments into complete lead data
                complete_lead_data = []
                processed_phones = []
                
                for enriched_lead in successful_enrichments:
                    # Debug: Print the enriched lead structure
                    print(f"DEBUG: Processing enriched lead: {enriched_lead}")
                    
                    # Get enriched data but use phone number ONLY from original CSV
                    structured_data = enriched_lead.get("structured_data", {})
                    # Correct key: enrichment result stores the original CSV row under 'lead'
                    original_lead = enriched_lead.get("lead", {})
                    
                    print(f"DEBUG: Structured data: {structured_data}")
                    print(f"DEBUG: Original lead before processing: {original_lead}")
                    
                    # Handle nested lead structure from SixtyFour API
                    if "lead" in original_lead and isinstance(original_lead["lead"], dict):
                        original_lead = original_lead["lead"]
                        print(f"DEBUG: Original lead after unwrapping: {original_lead}")
                    
                    # Only use phone from original CSV data (ignore SixtyFour phone data)
                    phone = (original_lead.get("phone") or original_lead.get("Phone") or
                            original_lead.get("phone_number") or original_lead.get("Phone Number"))
                    
                    if phone and str(phone).strip():
                        normalized_phone = csv_module.normalize_phone_number(str(phone).strip())
                        if normalized_phone:
                            lead_data_item = {
                                "phone_number": normalized_phone,
                                "original_lead": original_lead,
                                "structured_data": structured_data,
                                "enriched_data": structured_data,  # For backward compatibility
                                "notes": enriched_lead.get("notes", ""),
                                "findings": enriched_lead.get("findings", []),
                                "references": enriched_lead.get("references", {}),
                                "confidence_score": enriched_lead.get("confidence_score")
                            }
                            complete_lead_data.append(lead_data_item)
                            processed_phones.append(normalized_phone)
                
                # Add any leads that weren't enriched but have phone numbers
                for obj in dynamic_objects:
                    if not isinstance(obj, dict):
                        continue
                    
                    lead = obj.get("lead", {})
                    if not isinstance(lead, dict):
                        continue
                    
                    phone = (lead.get("phone") or lead.get("Phone") or 
                            lead.get("phone_number") or lead.get("Phone Number"))
                    
                    if phone and str(phone).strip():
                        normalized_phone = csv_module.normalize_phone_number(str(phone).strip())
                        if normalized_phone and normalized_phone not in processed_phones:
                            lead_data_item = {
                                "phone_number": normalized_phone,
                                "original_lead": lead,
                                "structured_data": {},
                                "enriched_data": {},
                                "notes": "Not enriched by SixtyFour API",
                                "findings": [],
                                "references": {},
                                "confidence_score": None
                            }
                            complete_lead_data.append(lead_data_item)
                            processed_phones.append(normalized_phone)
                
                # Save to session state
                st.session_state["phone_numbers"] = processed_phones
                st.session_state["complete_lead_data"] = complete_lead_data
                
                # Show results
                st.success(f"✅ Successfully processed {len(complete_lead_data)} leads")
                if successful_enrichments:
                    st.info(f"🚀 {len(successful_enrichments)} leads enriched with SixtyFour AI")
                if failed_enrichments:
                    st.warning(f"⚠️ {len(failed_enrichments)} leads could not be enriched")
                
                # Show preview of enriched data
                st.write("**Sample enriched leads:**")
                sample_to_show = complete_lead_data[:min(len(complete_lead_data), 3)]
                for i, lead in enumerate(sample_to_show, 1):
                    with st.expander(f"Lead {i}: {lead.get('phone_number', 'No phone')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original Data:**")
                            st.json(lead.get("original_lead", {}))
                        with col2:
                            st.write("**Enriched Data:**")
                            if lead.get("structured_data"):
                                st.json(lead.get("structured_data", {}))
                            else:
                                st.write("No enrichment available")
                
                return processed_phones, complete_lead_data
                
            except Exception as e:
                st.error(f"Error during enrichment: {str(e)}")
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
    lead_data: List[Dict[str, Any]], 
    prompt: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Make VAPI calls to leads with phone numbers from CSV
    
    Args:
        lead_data: List of lead data objects with phone numbers
        prompt: User-provided prompt for the AI agent
        
    Returns:
        Tuple of (successful_calls, failed_calls)
    """
    if not lead_data or not prompt.strip():
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
    
    # Get first message from session state
    first_message = st.session_state.get("vapi_first_message", "").strip()
    
    # Prepare call requests with enriched data
    call_requests = []
    for lead in lead_data:
        phone_number = lead.get("phone_number")
        if phone_number and isinstance(phone_number, str):
            # Prepare enriched data for the VAPI prompt
            enriched_data = {
                "original_lead": lead.get("original_lead", {}),
                "enriched_data": lead.get("enriched_data", {}),
                "structured_data": lead.get("structured_data", {}),
                "notes": lead.get("notes", ""),
                "findings": lead.get("findings", []),
                "references": lead.get("references", {}),
                "confidence_score": lead.get("confidence_score")
            }
            
            call_request = {
                "phone_number": phone_number,
                "prompt": prompt,
                "enriched_data": enriched_data
            }
            
            # Add first message if provided
            if first_message:
                call_request["first_message"] = first_message
            
            call_requests.append(call_request)
    
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
    phone_numbers = st.session_state.get("phone_numbers", [])
    complete_lead_data = st.session_state.get("complete_lead_data", [])
    vapi_prompt = st.session_state.get("vapi_prompt", "")
    
    if not phone_numbers or not complete_lead_data:
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
    
    # Check VAPI API key configuration using the client's built-in function
    try:
        vapi_api_ready = bool(vapi_client.has_api_key_configured())
    except Exception:
        vapi_api_ready = False
    
    if not vapi_api_ready:
        st.info("VAPI_API_TOKEN is not configured in the environment. Please set the VAPI_API_TOKEN environment variable.")
        return
    
    if st.button("Start VAPI Calls"):
        if not phone_numbers or not complete_lead_data:
            st.error("No phone numbers available for calling")
            return
        
        with st.spinner("Making VAPI calls..."):
            successful_calls, failed_calls = make_vapi_calls(complete_lead_data, vapi_prompt)
        
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
