"""
ChatGPT API Client for Prompt Generation
Generates recruitment call prompts based on CSV dataframe analysis.
"""

import os
import json
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Constants
OPENAI_BASE_URL = "https://api.openai.com/v1"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
DEFAULT_MODEL = "gpt-4"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7

@dataclass
class PromptGenerationResult:
    """Result object for prompt generation"""
    success: bool
    generated_prompt: Optional[str] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None

class ChatGPTClient:
    """Client for OpenAI ChatGPT API to generate recruitment call prompts"""
    
    def __init__(
        self, 
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = OPENAI_BASE_URL
    
    def generate_recruitment_prompt(self, dataframe: pd.DataFrame) -> PromptGenerationResult:
        """
        Generate a recruitment call prompt based on CSV dataframe analysis
        
        Args:
            dataframe: Pandas DataFrame containing lead data
            
        Returns:
            PromptGenerationResult object
        """
        if dataframe is None or dataframe.empty:
            return PromptGenerationResult(
                success=False,
                error="DataFrame is empty or None"
            )
        
        # Analyze the dataframe to understand the data structure
        dataframe_analysis = self._analyze_dataframe(dataframe)
        
        # Create system prompt for ChatGPT
        system_prompt = self._get_system_prompt()
        
        # Create user prompt with dataframe analysis
        user_prompt = self._create_user_prompt(dataframe_analysis)
        
        # Make API request
        return self._make_chat_completion_request(system_prompt, user_prompt)
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the dataframe to extract key information for prompt generation
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "sample_data": {},
            "data_types": {},
            "non_null_counts": {},
            "unique_values": {}
        }
        
        # Get sample data for first few rows
        sample_size = min(3, len(df))
        analysis["sample_data"] = df.head(sample_size).to_dict('records')
        
        # Analyze each column
        for col in df.columns:
            analysis["data_types"][col] = str(df[col].dtype)
            analysis["non_null_counts"][col] = int(df[col].notna().sum())
            
            # Get unique values for categorical-looking columns
            unique_count = df[col].nunique()
            if unique_count <= 20 and unique_count > 1:  # Reasonable number of unique values
                unique_vals = df[col].dropna().unique().tolist()
                analysis["unique_values"][col] = unique_vals[:10]  # Limit to first 10
        
        return analysis
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for ChatGPT to generate sales call prompts
        
        Returns:
            System prompt string
        """
        return """You are an expert recruitment prompt engineer specializing in creating effective AI agent scripts for outbound recruitment calls. Your task is to analyze CSV data containing candidate information and generate a professional, personalized recruitment call prompt that AI agents can use when calling these candidates.

Key requirements for the generated prompt:
1. Be professional, friendly, and conversational
2. Focus on building rapport and understanding candidate interests
3. Include placeholders for personalized information that will be filled in from enriched lead data
4. Have a clear call-to-action (usually scheduling an interview or follow-up discussion)
5. Be concise but comprehensive (aim for 200-400 words)
6. Include instructions for handling common candidate concerns or objections
7. Adapt the tone and approach based on the type of candidates in the data
8. Focus on opportunity presentation rather than hard selling

The prompt should work well with enriched candidate data that includes:
- Basic info (name, company, title, email, phone)
- Company details (industry, size, revenue, location)
- Professional details (years of experience, seniority level, main roles)
- Key insights and findings about the candidate

Generate a prompt that will help AI agents have successful, engaging recruitment conversations with these candidates, focusing on understanding their career goals and presenting relevant opportunities."""

    def _create_user_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        Create the user prompt with dataframe analysis
        
        Args:
            analysis: Dataframe analysis results
            
        Returns:
            User prompt string
        """
        prompt = f"""Please analyze this CSV data and generate an effective sales call prompt for AI agents:

**Dataset Overview:**
- Total leads: {analysis['total_rows']}
- Columns: {analysis['column_count']}
- Available data fields: {', '.join(analysis['columns'])}

**Sample Lead Data:**
```json
{json.dumps(analysis['sample_data'], indent=2, default=str)}
```

**Data Analysis:**
"""
        
        # Add column analysis
        for col in analysis['columns']:
            non_null = analysis['non_null_counts'].get(col, 0)
            data_type = analysis['data_types'].get(col, 'unknown')
            prompt += f"- {col}: {non_null}/{analysis['total_rows']} filled ({data_type})"
            
            if col in analysis['unique_values']:
                unique_vals = analysis['unique_values'][col]
                prompt += f" | Examples: {', '.join(map(str, unique_vals[:5]))}"
            prompt += "\n"
        
        prompt += """
Based on this data, please generate a recruitment call prompt that:
1. Is tailored to the type of candidates in this dataset
2. References the available data fields appropriately
3. Includes instructions for personalizing the call using enriched data
4. Has a clear opportunity presentation and call-to-action
5. Includes guidance for handling common candidate concerns
6. Focuses on understanding candidate career goals and interests

Return only the generated prompt, formatted as a clear script for AI agents to follow."""
        
        return prompt
    
    def _make_chat_completion_request(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> PromptGenerationResult:
        """
        Make a chat completion request to OpenAI API
        
        Args:
            system_prompt: System message for ChatGPT
            user_prompt: User message with dataframe analysis
            
        Returns:
            PromptGenerationResult object
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        url = f"{self.base_url}{CHAT_COMPLETIONS_ENDPOINT}"
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 401:
                return PromptGenerationResult(
                    success=False,
                    error="Invalid OpenAI API key"
                )
            
            if response.status_code == 429:
                return PromptGenerationResult(
                    success=False,
                    error="Rate limit exceeded. Please try again later."
                )
            
            if response.status_code != 200:
                return PromptGenerationResult(
                    success=False,
                    error=f"API request failed with status {response.status_code}: {response.text}"
                )
            
            response_data = response.json()
            
            # Extract the generated prompt
            if "choices" not in response_data or not response_data["choices"]:
                return PromptGenerationResult(
                    success=False,
                    error="No response choices returned from API"
                )
            
            generated_prompt = response_data["choices"][0]["message"]["content"].strip()
            tokens_used = response_data.get("usage", {}).get("total_tokens")
            
            return PromptGenerationResult(
                success=True,
                generated_prompt=generated_prompt,
                tokens_used=tokens_used,
                model_used=self.model
            )
            
        except requests.exceptions.Timeout:
            return PromptGenerationResult(
                success=False,
                error="Request timed out"
            )
        except requests.exceptions.RequestException as e:
            return PromptGenerationResult(
                success=False,
                error=f"Request failed: {str(e)}"
            )
        except Exception as e:
            return PromptGenerationResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )

# Convenience functions
def generate_prompt_with_env(dataframe: pd.DataFrame) -> PromptGenerationResult:
    """
    Generate recruitment prompt using environment-configured OpenAI API key
    
    Args:
        dataframe: Pandas DataFrame containing candidate data
        
    Returns:
        PromptGenerationResult object
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return PromptGenerationResult(
            success=False,
            error="OPENAI_API_KEY not configured in environment"
        )
    
    client = ChatGPTClient(api_key=api_key)
    return client.generate_recruitment_prompt(dataframe)

def has_openai_key_configured() -> bool:
    """Check if OpenAI API key is configured in environment"""
    return bool(os.getenv("OPENAI_API_KEY"))
