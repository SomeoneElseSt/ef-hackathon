#!/usr/bin/env python3
"""
Mock server that returns a specific phone number
Run with: python3 mock_server.py
Then use ngrok: ngrok http 5001
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.after_request
def after_request(response):
    """Add ngrok-skip-browser-warning header to all responses"""
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

MOCK_PHONE = "+14155836520"

@app.route('/find-phone', methods=['POST'])
def find_phone():
    """Mock endpoint that always returns the same phone number"""
    try:
        data = request.get_json()
        if not data or 'lead' not in data:
            return jsonify({"error": "Invalid request"}), 400
        
        lead = data['lead']
        
        # Return the lead data with the mock phone number
        response = {
            "name": lead.get("name", ""),
            "company": lead.get("company", ""),
            "title": lead.get("title", ""),
            "linkedin_url": lead.get("linkedin_url", ""),
            "email": lead.get("email", ""),
            "domain": lead.get("domain", ""),
            "phone": MOCK_PHONE
        }
        
        # Remove empty fields
        response = {k: v for k, v in response.items() if v}
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "mock_phone": MOCK_PHONE})

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Mock SixtyFour API Server",
        "endpoints": {
            "find-phone": "POST /find-phone",
            "health": "GET /health"
        },
        "mock_phone": MOCK_PHONE
    })

if __name__ == '__main__':
    print("🚀 Mock SixtyFour API Server starting...")
    print(f"📞 Will return phone: {MOCK_PHONE}")
    print("🌐 Run 'ngrok http 5001' to expose publicly")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=True)
