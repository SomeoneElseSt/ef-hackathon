#!/usr/bin/env python3
"""
Test script to demonstrate enriched prompts with SixtyFour API data
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import the VAPI client
exec(open('vapi-client.py').read())

def test_enriched_prompts():
    """Test the enriched prompt functionality"""
    
    # Create a VAPI client instance
    client = VAPIClient(
        api_token='test_token',
        phone_number_id='test_phone_id'
    )
    
    # Base prompt from user
    base_prompt = """You are a professional sales representative calling potential leads. 
Your goal is to introduce our software solution and schedule a demo. 
Be friendly, professional, and focus on their specific business needs."""
    
    # Sample enriched data from SixtyFour API
    enriched_data = {
        'original_lead': {
            'name': 'Sarah Johnson',
            'company': 'TechStart Inc',
            'phone': '555-123-4567',
            'email': 'sarah@techstart.com',
            'title': 'CTO'
        },
        'structured_data': {
            'industry': 'Software Development',
            'company_size': '25-50 employees',
            'revenue': '$2M-$5M',
            'location': 'San Francisco, CA',
            'founded': '2020',
            'technologies': 'React, Node.js, AWS'
        },
        'findings': [
            'Recently raised Series A funding',
            'Actively hiring developers',
            'Has expressed interest in automation tools',
            'Currently using outdated project management system'
        ],
        'notes': 'High-potential prospect. Company is in growth phase and likely has budget for new tools. CTO is the decision maker.',
        'confidence_score': 0.92,
        'references': {
            'linkedin': 'https://linkedin.com/company/techstart-inc',
            'website': 'https://techstart.com'
        }
    }
    
    # Build enriched prompt
    enriched_prompt = client._build_enriched_prompt(base_prompt, enriched_data)
    
    # Test phone number normalization
    test_phone = '555-123-4567'
    normalized_phone = normalize_phone_number(test_phone)
    
    print("=" * 80)
    print("ENRICHED VAPI PROMPT TEST")
    print("=" * 80)
    
    print("\n📞 PHONE NUMBER NORMALIZATION:")
    print(f"Original: {test_phone}")
    print(f"Normalized: {normalized_phone}")
    
    print(f"\n📝 BASE PROMPT ({len(base_prompt)} chars):")
    print("-" * 40)
    print(base_prompt)
    
    print(f"\n🚀 ENRICHED PROMPT ({len(enriched_prompt)} chars):")
    print("-" * 40)
    print(enriched_prompt)
    
    print(f"\n📊 STATS:")
    print(f"- Base prompt: {len(base_prompt)} characters")
    print(f"- Enriched prompt: {len(enriched_prompt)} characters")
    print(f"- Added context: {len(enriched_prompt) - len(base_prompt)} characters")
    print(f"- Confidence score: {enriched_data['confidence_score']}")
    print(f"- Key findings: {len(enriched_data['findings'])} items")
    
    print("\n✅ Test completed successfully!")
    print("The AI agent will now have rich context about the lead when making the call.")

if __name__ == "__main__":
    test_enriched_prompts()
