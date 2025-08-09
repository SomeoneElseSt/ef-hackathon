"""
VAPI AI Outbound Call Client
Handles outbound phone calls with customizable AI assistants using ElevenLabs and OpenAI.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Constants
BASE_URL = "https://api.vapi.ai"
OUTBOUND_CALL_ENDPOINT = "/call/phone"
DEFAULT_VOICE_PROVIDER = "11labs"
DEFAULT_MODEL_PROVIDER = "openai"
DEFAULT_TRANSCRIBER_PROVIDER = "deepgram"
DEFAULT_MAX_CONCURRENT_CALLS = 5
DEFAULT_CALL_DELAY = 2.0
MAX_RETRIES = 3
RETRY_DELAY = 5.0

class CallStatus(Enum):
    """Call status enumeration"""
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    INVALID_PHONE = "invalid_phone"
    INSUFFICIENT_FUNDS = "insufficient_funds"

@dataclass
class CallResult:
    """Result object for outbound call attempts"""
    success: bool
    call_id: Optional[str] = None
    phone_number: Optional[str] = None
    status: Optional[CallStatus] = None
    error: Optional[str] = None

class VAPIClient:
    """Async client for VAPI outbound calling"""
    
    def __init__(
        self, 
        api_token: str,
        phone_number_id: str,
        assistant_id: Optional[str] = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_CALLS,
        call_delay: float = DEFAULT_CALL_DELAY
    ):
        self.api_token = api_token
        self.phone_number_id = phone_number_id
        self.assistant_id = assistant_id
        self.max_concurrent = max_concurrent
        self.call_delay = call_delay
        self.base_url = BASE_URL
    
    async def make_outbound_calls(
        self, 
        call_requests: List[Dict[str, str]]
    ) -> List[CallResult]:
        """
        Makes multiple outbound calls asynchronously
        
        Args:
            call_requests: List of dicts with 'phone_number' and 'prompt' keys
            
        Returns:
            List of CallResult objects
        """
        if not call_requests:
            return []
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._make_single_call_with_semaphore(session, request, semaphore)
                for request in call_requests
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return self._process_call_results(results)
    
    async def make_single_call(
        self, 
        phone_number: str, 
        prompt: str,
        assistant_overrides: Optional[Dict[str, Any]] = None
    ) -> CallResult:
        """
        Makes a single outbound call
        
        Args:
            phone_number: Target phone number (e.g., "+1234567890")
            prompt: System prompt for the AI assistant
            assistant_overrides: Optional assistant configuration overrides
            
        Returns:
            CallResult object
        """
        async with aiohttp.ClientSession() as session:
            request_data = {
                "phone_number": phone_number,
                "prompt": prompt
            }
            
            if assistant_overrides:
                request_data.update(assistant_overrides)
            
            return await self._execute_outbound_call(session, request_data)
    
    async def _make_single_call_with_semaphore(
        self, 
        session: aiohttp.ClientSession,
        call_request: Dict[str, str],
        semaphore: asyncio.Semaphore
    ) -> CallResult:
        """Makes single call with semaphore control"""
        async with semaphore:
            result = await self._execute_outbound_call(session, call_request)
            await asyncio.sleep(self.call_delay)
            return result
    
    async def _execute_outbound_call(
        self, 
        session: aiohttp.ClientSession,
        call_request: Dict[str, str]
    ) -> CallResult:
        """
        Executes outbound call with retry logic
        
        Args:
            session: aiohttp session
            call_request: Dict with phone_number and prompt
            
        Returns:
            CallResult object
        """
        phone_number = call_request.get("phone_number", "")
        prompt = call_request.get("prompt", "")
        
        if not self._is_valid_phone_number(phone_number):
            return CallResult(
                success=False,
                phone_number=phone_number,
                status=CallStatus.INVALID_PHONE,
                error="Invalid phone number format"
            )
        
        if not prompt:
            return CallResult(
                success=False,
                phone_number=phone_number,
                status=CallStatus.FAILED,
                error="Prompt is required"
            )
        
        payload = self._build_call_payload(phone_number, prompt, call_request)
        
        for attempt in range(MAX_RETRIES):
            result = await self._make_api_call(session, payload, phone_number)
            
            if result.success:
                return result
            
            if result.status == CallStatus.RATE_LIMITED:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            
            return result
        
        return CallResult(
            success=False,
            phone_number=phone_number,
            status=CallStatus.FAILED,
            error="Max retries exceeded"
        )
    
    async def _make_api_call(
        self, 
        session: aiohttp.ClientSession,
        payload: Dict[str, Any],
        phone_number: str
    ) -> CallResult:
        """
        Makes API call to VAPI outbound endpoint
        
        Args:
            session: aiohttp session
            payload: Call payload
            phone_number: Target phone number
            
        Returns:
            CallResult object
        """
        headers = self._get_request_headers()
        url = f"{self.base_url}{OUTBOUND_CALL_ENDPOINT}"
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 429:
                    return CallResult(
                        success=False,
                        phone_number=phone_number,
                        status=CallStatus.RATE_LIMITED,
                        error="Rate limit exceeded"
                    )
                
                if response.status == 401:
                    return CallResult(
                        success=False,
                        phone_number=phone_number,
                        status=CallStatus.FAILED,
                        error="Invalid API token"
                    )
                
                if response.status == 402:
                    return CallResult(
                        success=False,
                        phone_number=phone_number,
                        status=CallStatus.INSUFFICIENT_FUNDS,
                        error="Insufficient account balance"
                    )
                
                response_data = await response.json()
                
                if response.status == 201:
                    call_id = response_data.get("id")
                    return CallResult(
                        success=True,
                        call_id=call_id,
                        phone_number=phone_number,
                        status=CallStatus.SUCCESS
                    )
                
                error_message = response_data.get("message", f"HTTP {response.status}")
                return CallResult(
                    success=False,
                    phone_number=phone_number,
                    status=CallStatus.FAILED,
                    error=error_message
                )
                
        except aiohttp.ClientError as e:
            return CallResult(
                success=False,
                phone_number=phone_number,
                status=CallStatus.FAILED,
                error=f"Client error: {str(e)}"
            )
        except Exception as e:
            return CallResult(
                success=False,
                phone_number=phone_number,
                status=CallStatus.FAILED,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _build_call_payload(
        self, 
        phone_number: str, 
        prompt: str, 
        additional_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds the API call payload
        
        Args:
            phone_number: Target phone number
            prompt: System prompt
            additional_config: Additional configuration options
            
        Returns:
            API payload dictionary
        """
        # Create transient assistant if no assistant_id provided
        if self.assistant_id:
            payload = {
                "assistantId": self.assistant_id,
                "phoneNumberId": self.phone_number_id,
                "customer": {
                    "number": phone_number
                },
                "assistantOverrides": {
                    "model": {
                        "messages": [
                            {
                                "role": "system",
                                "content": prompt
                            }
                        ]
                    }
                }
            }
        else:
            # Create transient assistant with full configuration
            payload = {
                "phoneNumberId": self.phone_number_id,
                "customer": {
                    "number": phone_number
                },
                "assistant": self._create_transient_assistant(prompt, additional_config)
            }
        
        # Add optional first message override
        first_message = additional_config.get("first_message")
        if first_message:
            if "assistantOverrides" not in payload:
                payload["assistantOverrides"] = {}
            payload["assistantOverrides"]["firstMessage"] = first_message
        
        return payload
    
    def _create_transient_assistant(
        self, 
        prompt: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates transient assistant configuration
        
        Args:
            prompt: System prompt
            config: Additional configuration
            
        Returns:
            Assistant configuration dictionary
        """
        # Extract configuration values with defaults
        model = config.get("model", "gpt-3.5-turbo")
        voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default ElevenLabs voice
        first_message = config.get("first_message", "Hello! How can I help you today?")
        
        assistant = {
            "model": {
                "provider": DEFAULT_MODEL_PROVIDER,
                "model": model,
                "messages": [
                    {
                        "role": "system", 
                        "content": prompt
                    }
                ]
            },
            "voice": {
                "provider": DEFAULT_VOICE_PROVIDER,
                "voiceId": voice_id
            },
            "transcriber": {
                "provider": DEFAULT_TRANSCRIBER_PROVIDER,
                "model": "nova-2"
            },
            "firstMessage": first_message
        }
        
        # Add optional configuration overrides
        if "temperature" in config:
            assistant["model"]["temperature"] = config["temperature"]
        
        if "max_tokens" in config:
            assistant["model"]["maxTokens"] = config["max_tokens"]
        
        return assistant
    
    def _get_request_headers(self) -> Dict[str, str]:
        """Returns request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    def _is_valid_phone_number(self, phone_number: str) -> bool:
        """
        Validates phone number format
        
        Args:
            phone_number: Phone number string
            
        Returns:
            True if valid format
        """
        if not phone_number:
            return False
        
        # Remove spaces and special characters for validation
        cleaned = phone_number.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        
        # Must start with + and have 10-15 digits
        if not cleaned.startswith("+"):
            return False
        
        digits = cleaned[1:]
        if not digits.isdigit():
            return False
        
        return 10 <= len(digits) <= 15
    
    def _process_call_results(
        self, 
        results: List[Union[CallResult, Exception]]
    ) -> List[CallResult]:
        """Processes gathered results and handles exceptions"""
        processed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(CallResult(
                    success=False,
                    status=CallStatus.FAILED,
                    error=f"Task exception: {str(result)}"
                ))
                continue
            
            processed_results.append(result)
        
        return processed_results

# Convenience functions
async def make_outbound_calls(
    api_token: str,
    phone_number_id: str,
    call_requests: List[Dict[str, str]],
    assistant_id: Optional[str] = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_CALLS
) -> List[CallResult]:
    """
    Convenience function to make multiple outbound calls
    
    Args:
        api_token: VAPI API token
        phone_number_id: VAPI phone number ID
        call_requests: List of call requests with phone_number and prompt
        assistant_id: Optional pre-configured assistant ID
        max_concurrent: Maximum concurrent calls
        
    Returns:
        List of CallResult objects
    """
    client = VAPIClient(
        api_token=api_token,
        phone_number_id=phone_number_id,
        assistant_id=assistant_id,
        max_concurrent=max_concurrent
    )
    return await client.make_outbound_calls(call_requests)

async def make_single_outbound_call(
    api_token: str,
    phone_number_id: str,
    phone_number: str,
    prompt: str,
    assistant_id: Optional[str] = None,
    **kwargs
) -> CallResult:
    """
    Convenience function to make a single outbound call
    
    Args:
        api_token: VAPI API token
        phone_number_id: VAPI phone number ID
        phone_number: Target phone number
        prompt: System prompt for AI assistant
        assistant_id: Optional pre-configured assistant ID
        **kwargs: Additional configuration options
        
    Returns:
        CallResult object
    """
    client = VAPIClient(
        api_token=api_token,
        phone_number_id=phone_number_id,
        assistant_id=assistant_id
    )
    return await client.make_single_call(phone_number, prompt, kwargs)

def extract_successful_calls(results: List[CallResult]) -> List[Dict[str, Any]]:
    """
    Extracts successful call results
    
    Args:
        results: List of CallResult objects
        
    Returns:
        List of successful call data
    """
    successful = []
    
    for result in results:
        if not result.success:
            continue
        
        successful.append({
            "call_id": result.call_id,
            "phone_number": result.phone_number,
            "status": result.status.value if result.status else None
        })
    
    return successful

def extract_failed_calls(results: List[CallResult]) -> List[Dict[str, Any]]:
    """
    Extracts failed call attempts
    
    Args:
        results: List of CallResult objects
        
    Returns:
        List of failed call data with error information
    """
    failed = []
    
    for result in results:
        if result.success:
            continue
        
        failed.append({
            "phone_number": result.phone_number,
            "status": result.status.value if result.status else None,
            "error": result.error
        })
    
    return failed
