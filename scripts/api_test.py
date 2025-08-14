# Not useful. It's just for me to test the model

import requests

payload = {
    "users": [
        {"id": "A0015610VMNR0JC9XVL1"}
    ]
}

# Envia o POST
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
)

print(f'Status code: {response.status_code}')
print(f'Response: {response.content}')
