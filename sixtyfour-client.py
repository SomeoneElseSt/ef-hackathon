"""
SixtyFour AI API Client
Handles async phone number enrichment with rate limiting and error handling.
"""

import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Constants
BASE_URL = "https://api.sixtyfour.ai"
FIND_PHONE_ENDPOINT = "/find-phone"
DEFAULT_MAX_CONCURRENT_REQUESTS = 10
DEFAULT_RATE_LIMIT_DELAY = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 2.0

@dataclass
class PhoneResult:
    """Result object for phone enrichment"""
    success: bool
    phone: Optional[Union[str, List[Dict[str, str]]]] = None
    original_lead: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SixtyFourClient:
    """Async client for SixtyFour AI API"""
    
    def __init__(
        self, 
        api_key: str, 
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY
    ):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.base_url = BASE_URL
        
    async def enrich_leads_async(self, leads: List[Dict[str, Any]]) -> List[PhoneResult]:
        """
        Enriches multiple leads asynchronously with phone numbers
        
        Args:
            leads: List of lead dictionaries from CSV rows
            
        Returns:
            List of PhoneResult objects
        """
        if not leads:
            return []
            
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._enrich_single_lead_with_semaphore(session, lead, semaphore)
                for lead in leads
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        return self._process_results(results)
    
    async def _enrich_single_lead_with_semaphore(
        self, 
        session: aiohttp.ClientSession, 
        lead: Dict[str, Any], 
        semaphore: asyncio.Semaphore
    ) -> PhoneResult:
        """Enriches single lead with semaphore control"""
        async with semaphore:
            result = await self._enrich_single_lead(session, lead)
            await asyncio.sleep(self.rate_limit_delay)
            return result
    
    async def _enrich_single_lead(
        self, 
        session: aiohttp.ClientSession, 
        lead: Dict[str, Any]
    ) -> PhoneResult:
        """
        Enriches a single lead with phone number
        
        Args:
            session: aiohttp session
            lead: Lead dictionary from CSV row
            
        Returns:
            PhoneResult object
        """
        cleaned_lead = self._clean_lead_data(lead)
        
        if not cleaned_lead:
            return PhoneResult(
                success=False,
                original_lead=lead,
                error="No valid lead data after cleaning"
            )
        
        payload = {"lead": cleaned_lead}
        
        for attempt in range(MAX_RETRIES):
            result = await self._make_api_request(session, payload, lead)
            
            if result.success:
                return result
                
            if "rate limit" in (result.error or "").lower():
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
                
            return result
            
        return PhoneResult(
            success=False,
            original_lead=lead,
            error="Max retries exceeded"
        )
    
    async def _make_api_request(
        self, 
        session: aiohttp.ClientSession, 
        payload: Dict[str, Any], 
        original_lead: Dict[str, Any]
    ) -> PhoneResult:
        """
        Makes API request to SixtyFour AI
        
        Args:
            session: aiohttp session
            payload: Request payload
            original_lead: Original lead data
            
        Returns:
            PhoneResult object
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        url = f"{self.base_url}{FIND_PHONE_ENDPOINT}"
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 429:
                    return PhoneResult(
                        success=False,
                        original_lead=original_lead,
                        error="Rate limit exceeded"
                    )
                
                if response.status == 401:
                    return PhoneResult(
                        success=False,
                        original_lead=original_lead,
                        error="Invalid API key"
                    )
                
                if response.status != 200:
                    error_text = await response.text()
                    return PhoneResult(
                        success=False,
                        original_lead=original_lead,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                
                data = await response.json()
                phone = data.get("phone")
                
                return PhoneResult(
                    success=True,
                    phone=phone,
                    original_lead=original_lead
                )
                
        except aiohttp.ClientError as e:
            return PhoneResult(
                success=False,
                original_lead=original_lead,
                error=f"Client error: {str(e)}"
            )
        except Exception as e:
            return PhoneResult(
                success=False,
                original_lead=original_lead,
                error=f"Unexpected error: {str(e)}"
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
        
        # Map common CSV column names to API field names
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
            "email_address": "email"
        }
        
        return key_mapping.get(cleaned, cleaned)
    
    def _process_results(self, results: List[Union[PhoneResult, Exception]]) -> List[PhoneResult]:
        """Processes gathered results and handles exceptions"""
        processed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(PhoneResult(
                    success=False,
                    error=f"Task exception: {str(result)}"
                ))
                continue
                
            processed_results.append(result)
            
        return processed_results

# Convenience functions
async def enrich_leads(
    api_key: str, 
    leads: List[Dict[str, Any]], 
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_REQUESTS
) -> List[PhoneResult]:
    """
    Convenience function to enrich leads with phone numbers
    
    Args:
        api_key: SixtyFour AI API key
        leads: List of lead dictionaries
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of PhoneResult objects
    """
    client = SixtyFourClient(api_key, max_concurrent)
    return await client.enrich_leads_async(leads)

def _get_api_key_from_env() -> Optional[str]:
    api_key = os.getenv("SIXTYFOUR_API_KEY", "").strip()
    if not api_key:
        return None
    return api_key

def has_api_key_configured() -> bool:
    return _get_api_key_from_env() is not None

async def enrich_leads_with_env(
    leads: List[Dict[str, Any]], 
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_REQUESTS
) -> List[PhoneResult]:
    api_key = _get_api_key_from_env()
    if not api_key:
        return []
    return await enrich_leads(api_key, leads, max_concurrent)

def extract_successful_phones(results: List[PhoneResult]) -> List[Dict[str, Any]]:
    """
    Extracts successful phone results
    
    Args:
        results: List of PhoneResult objects
        
    Returns:
        List of dictionaries with successful phone enrichments
    """
    successful = []
    
    for result in results:
        if not result.success:
            continue
            
        enriched = result.original_lead.copy() if result.original_lead else {}
        enriched["phone"] = result.phone
        successful.append(enriched)
        
    return successful

def extract_failed_leads(results: List[PhoneResult]) -> List[Dict[str, Any]]:
    """
    Extracts failed enrichment attempts
    
    Args:
        results: List of PhoneResult objects
        
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
