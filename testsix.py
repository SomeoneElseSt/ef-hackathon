import requests

url = "https://api.sixtyfour.ai/find-phone"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "api_wGE9awshV7gPteFUKv6bm3vUXSz9LAqu",
    "ngrok-skip-browser-warning": "true"
}
data = {
    "lead": {
        "name": "Stiven Triana",
        "company": "Domu (YC S24)", 
        "linkedin_url": "https://www.linkedin.com/in/stiven-triana-876061255",
        "email": "stiven@domu.ai",
        "domain": "domu.ai",
        "personal_website": "stiven.me",
        "phone_number_url": "https://09d50013bd1a.ngrok-free.app"
    }
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(f"Phone found: {result.get('phone', 'Not found')}")