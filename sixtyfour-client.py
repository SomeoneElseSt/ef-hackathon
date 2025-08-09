"""
SixtyFour AI API Client
Handles simple synchronous phone number enrichment with rate limiting and error handling.
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


BASE_URL = "https://api.sixtyfour.ai"
ENRICH_LEAD_ENDPOINT = "/enrich-lead"
ENRICH_LEAD_ASYNC_ENDPOINT = "/enrich-lead-async"
JOB_STATUS_ENDPOINT = "/job-status"
DEFAULT_RATE_LIMIT_DELAY = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 2.0
POLLING_INTERVAL = 10.0

@dataclass
class EnrichmentResult:
    """Result object for lead enrichment"""
    success: bool
    structured_data: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    findings: Optional[List[str]] = None
    references: Optional[Dict[str, str]] = None
    confidence_score: Optional[float] = None
    original_lead: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None

class SixtyFourClient:
    """Simple synchronous client for SixtyFour AI API with async polling support"""
    
    def __init__(
        self, 
        api_key: str, 
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        use_async: bool = True
    ):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.base_url = BASE_URL
        self.use_async = use_async
        self.default_struct = self._get_default_struct()
        
    def enrich_leads(self, leads: List[Dict[str, Any]]) -> List[EnrichmentResult]:
        """
        Enriches multiple leads with comprehensive information
        
        Args:
            leads: List of lead dictionaries from CSV rows
            
        Returns:
            List of EnrichmentResult objects
        """
        if not leads:
            return []
        
        results = []
        for lead in leads:
            result = self._enrich_single_lead(lead)
            results.append(result)
            time.sleep(self.rate_limit_delay)
            
        return results
    
    def _enrich_single_lead(self, lead: Dict[str, Any]) -> EnrichmentResult:
        """
        Enriches a single lead with comprehensive information
        
        Args:
            lead: Lead dictionary from CSV row
            
        Returns:
            EnrichmentResult object
        """
        cleaned_lead = self._clean_lead_data(lead)
        
        if not cleaned_lead:
            return EnrichmentResult(
                success=False,
                original_lead=lead,
                error="No valid lead data after cleaning"
            )
        
        payload = {
            "lead_info": cleaned_lead,
            "struct": self.default_struct
        }
        
        for attempt in range(MAX_RETRIES):
            if self.use_async:
                result = self._make_async_api_request(payload, lead)
            else:
                result = self._make_sync_api_request(payload, lead)
            
            if result.success:
                return result
                
            if "rate limit" in (result.error or "").lower():
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
                
            return result
            
        return EnrichmentResult(
            success=False,
            original_lead=lead,
            error="Max retries exceeded"
        )
    
    def _get_default_struct(self) -> Dict[str, str]:
        """
        Returns default struct for sales voicebot lead enrichment
        """
        return {
            "name": "The individual's full name",
            "email": "The individual's email address",
            "phone": "The individual's phone number",
            "company": "The company the individual is associated with",
            "title": "The individual's job title",
            "linkedin": "LinkedIn URL for the person",
            "website": "Company website URL",
            "location": "The individual's location and/or company location",
            "industry": "Industry the person operates in"
        }
    
    def _make_sync_api_request(
        self, 
        payload: Dict[str, Any], 
        original_lead: Dict[str, Any]
    ) -> EnrichmentResult:
        """
        Makes synchronous API request to SixtyFour AI
        
        Args:
            payload: Request payload
            original_lead: Original lead data
            
        Returns:
            EnrichmentResult object
        """
        url = f"{self.base_url}{ENRICH_LEAD_ENDPOINT}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "ngrok-skip-browser-warning": "true"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            return self._parse_enrichment_response(response, original_lead)
                
        except requests.exceptions.RequestException as e:
            return EnrichmentResult(
                success=False,
                original_lead=original_lead,
                error=f"Request error: {str(e)}"
            )
        except Exception as e:
            return EnrichmentResult(
                success=False,
                original_lead=original_lead,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _make_async_api_request(
        self, 
        payload: Dict[str, Any], 
        original_lead: Dict[str, Any]
    ) -> EnrichmentResult:
        """
        Makes async API request to SixtyFour AI with polling
        
        Args:
            payload: Request payload
            original_lead: Original lead data
            
        Returns:
            EnrichmentResult object
        """
        # Start async job
        url = f"{self.base_url}{ENRICH_LEAD_ASYNC_ENDPOINT}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "ngrok-skip-browser-warning": "true"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            raw_text = response.text
            
            if response.status_code != 200:
                return self._handle_error_response(response.status_code, raw_text, original_lead)
            
            try:
                task_data = response.json()
            except Exception as parse_error:
                return EnrichmentResult(
                    success=False,
                    original_lead=original_lead,
                    error=f"Invalid JSON response: {str(parse_error)} | raw={raw_text}",
                    raw_response=raw_text
                )
            
            task_id = task_data.get("task_id")
            if not task_id:
                return EnrichmentResult(
                    success=False,
                    original_lead=original_lead,
                    error="No task_id returned from async endpoint",
                    raw_response=raw_text
                )
            
            # Poll for results
            return self._poll_for_results(task_id, original_lead)
                
        except requests.exceptions.RequestException as e:
            return EnrichmentResult(
                success=False,
                original_lead=original_lead,
                error=f"Request error: {str(e)}"
            )
        except Exception as e:
            return EnrichmentResult(
                success=False,
                original_lead=original_lead,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _poll_for_results(self, task_id: str, original_lead: Dict[str, Any]) -> EnrichmentResult:
        """
        Polls for async job results
        
        Args:
            task_id: Task ID from async job
            original_lead: Original lead data
            
        Returns:
            EnrichmentResult object
        """
        url = f"{self.base_url}{JOB_STATUS_ENDPOINT}/{task_id}"
        headers = {
            "x-api-key": self.api_key,
            "ngrok-skip-browser-warning": "true"
        }
        
        max_polls = 30  # Max 5 minutes of polling
        polls = 0
        
        while polls < max_polls:
            try:
                response = requests.get(url, headers=headers)
                raw_text = response.text
                
                if response.status_code != 200:
                    return self._handle_error_response(response.status_code, raw_text, original_lead)
                
                try:
                    status_data = response.json()
                except Exception as parse_error:
                    return EnrichmentResult(
                        success=False,
                        original_lead=original_lead,
                        error=f"Invalid JSON response: {str(parse_error)} | raw={raw_text}",
                        raw_response=raw_text
                    )
                
                status = status_data.get("status")
                
                if status == "completed":
                    result_data = status_data.get("result", {})
                    return EnrichmentResult(
                        success=True,
                        structured_data=result_data.get("structured_data"),
                        notes=result_data.get("notes"),
                        findings=result_data.get("findings"),
                        references=result_data.get("references"),
                        confidence_score=result_data.get("confidence_score"),
                        original_lead=original_lead,
                        raw_response=raw_text
                    )
                elif status == "failed":
                    return EnrichmentResult(
                        success=False,
                        original_lead=original_lead,
                        error=f"Job failed: {status_data.get('error', 'Unknown error')}",
                        raw_response=raw_text
                    )
                elif status in ["pending", "processing"]:
                    time.sleep(POLLING_INTERVAL)
                    polls += 1
                    continue
                else:
                    return EnrichmentResult(
                        success=False,
                        original_lead=original_lead,
                        error=f"Unknown status: {status}",
                        raw_response=raw_text
                    )
                    
            except requests.exceptions.RequestException as e:
                return EnrichmentResult(
                    success=False,
                    original_lead=original_lead,
                    error=f"Polling error: {str(e)}"
                )
            except Exception as e:
                return EnrichmentResult(
                    success=False,
                    original_lead=original_lead,
                    error=f"Unexpected polling error: {str(e)}"
                )
        
        return EnrichmentResult(
            success=False,
            original_lead=original_lead,
            error="Polling timeout - job did not complete within expected time"
        )
    
    def _parse_enrichment_response(self, response, original_lead: Dict[str, Any]) -> EnrichmentResult:
        """
        Parses enrichment response from sync endpoint
        
        Args:
            response: HTTP response object
            original_lead: Original lead data
            
        Returns:
            EnrichmentResult object
        """
        raw_text = response.text
        
        if response.status_code != 200:
            return self._handle_error_response(response.status_code, raw_text, original_lead)
        
        try:
            data = response.json()
        except Exception as parse_error:
            return EnrichmentResult(
                success=False,
                original_lead=original_lead,
                error=f"Invalid JSON response: {str(parse_error)} | raw={raw_text}",
                raw_response=raw_text
            )
        
        return EnrichmentResult(
            success=True,
            structured_data=data.get("structured_data"),
            notes=data.get("notes"),
            findings=data.get("findings"),
            references=data.get("references"),
            confidence_score=data.get("confidence_score"),
            original_lead=original_lead,
            raw_response=raw_text
        )
    
    def _handle_error_response(self, status_code: int, raw_text: str, original_lead: Dict[str, Any]) -> EnrichmentResult:
        """
        Handles error responses from API
        
        Args:
            status_code: HTTP status code
            raw_text: Raw response text
            original_lead: Original lead data
            
        Returns:
            EnrichmentResult object with error
        """
        if status_code == 429:
            return EnrichmentResult(
                success=False,
                original_lead=original_lead,
                error="Rate limit exceeded",
                raw_response=raw_text
            )
        
        if status_code == 401:
            return EnrichmentResult(
                success=False,
                original_lead=original_lead,
                error="Invalid API key",
                raw_response=raw_text
            )
        
        return EnrichmentResult(
            success=False,
            original_lead=original_lead,
            error=f"HTTP {status_code}: {raw_text}",
            raw_response=raw_text
        )
    
    def _clean_lead_data(self, lead: Dict[str, Any]) -> Dict[str, str]:
        """
        Removes null, empty, and whitespace-only values from lead data
        
        Args:
            lead: Raw lead dictionary from CSV
            
        Returns:
            Cleaned lead dictionary
        """
        if lead and isinstance(lead, dict) and "lead" in lead and isinstance(lead["lead"], dict):
            lead = lead["lead"]
        
        if not lead:
            return {}
            
        cleaned = {}
        
        for key, value in lead.items():
            if not self._is_valid_value(value):
                continue
                
            cleaned_key = self._clean_key(key)
            cleaned_value = str(value).strip()
            
            if cleaned_key and cleaned_value:
                cleaned[cleaned_key] = cleaned_value
                
        return cleaned
    
    def _is_valid_value(self, value: Any) -> bool:
        """Checks if value is valid (not null, empty, or whitespace)"""
        if value is None:
            return False
            
        if isinstance(value, str):
            return bool(value.strip())
            
        if isinstance(value, (int, float)):
            return True
            
        return bool(value)
    
    def _clean_key(self, key: str) -> str:
        """Cleans and standardizes dictionary keys"""
        if not key:
            return ""
            
        cleaned = str(key).strip().lower()
        

        key_mapping = {
            "full_name": "name",
            "person_name": "name",
            "contact_name": "name",
            "company_name": "company",
            "organization": "company",
            "linkedin": "linkedin_url",
            "linkedin_profile": "linkedin_url",
            "website": "domain",
            "company_domain": "domain",
            "email_address": "email",
            "work_email": "email",
            "business_email": "email"
        }
        
        return key_mapping.get(cleaned, cleaned)
    
    def _process_results(self, results: List[Union[EnrichmentResult, Exception]]) -> List[EnrichmentResult]:
        """Processes gathered results and handles exceptions"""
        processed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(EnrichmentResult(
                    success=False,
                    error=f"Task exception: {str(result)}"
                ))
                continue
            processed_results.append(result)
            
        return processed_results


def enrich_leads(
    api_key: str, 
    leads: List[Dict[str, Any]],
    use_async: bool = True
) -> List[EnrichmentResult]:
    """
    Convenience function to enrich leads with comprehensive information
    
    Args:
        api_key: SixtyFour AI API key
        leads: List of lead dictionaries
        use_async: Whether to use async endpoint with polling
        
    Returns:
        List of EnrichmentResult objects
    """
    client = SixtyFourClient(api_key, use_async=use_async)
    return client.enrich_leads(leads)

def _get_api_key_from_env() -> Optional[str]:
    api_key = os.getenv("SIXTYFOUR_API_KEY", "").strip()

    if not api_key:
        return None
    return api_key

def has_api_key_configured() -> bool:
    return _get_api_key_from_env() is not None

def enrich_leads_with_env(
    leads: List[Dict[str, Any]],
    use_async: bool = True
) -> List[EnrichmentResult]:
    api_key = _get_api_key_from_env()
    if not api_key:
        return []
    return enrich_leads(api_key, leads, use_async=use_async)

def extract_successful_phones(results: List[EnrichmentResult]) -> List[Dict[str, Any]]:
    """
    Extracts successful enrichment results with phone numbers
    
    Args:
        results: List of EnrichmentResult objects
        
    Returns:
        List of dictionaries with successful enrichments (backward compatibility)
    """
    successful = []
    
    for result in results:
        if not result.success or not result.structured_data:
            continue
        
        # Create enriched lead with phone number for backward compatibility
        enriched = result.original_lead.copy() if result.original_lead else {}
        
        # Extract phone from structured_data
        phone = result.structured_data.get("phone")
        if phone:
            enriched["phone"] = phone
            
        # Add other enriched data
        if result.structured_data:
            enriched.update(result.structured_data)
            
        if result.raw_response is not None:
            enriched["raw_response"] = result.raw_response
            
        successful.append(enriched)
        
    return successful

def extract_failed_leads(results: List[EnrichmentResult]) -> List[Dict[str, Any]]:
    """
    Extracts failed enrichment attempts
    
    Args:
        results: List of EnrichmentResult objects
        
    Returns:
        List of dictionaries with failed enrichments and error messages
    """
    failed = []
    
    for result in results:
        if result.success:
            continue
            
        failed_lead = result.original_lead.copy() if result.original_lead else {}
        failed_lead["error"] = result.error
        failed.append(failed_lead)
        
    return failed
