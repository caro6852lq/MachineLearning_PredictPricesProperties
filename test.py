import requests

url = "http://localhost:9696/predict"

house= {
    "property_type": "apartment",
    "lat": -34.56,
    "lon": -58.45,
    "surface_total": 71,
    "surface_covered": 68,
    "rooms": 3
    "barrio": "SAN CRISTOBAL"
    "comuna": 3

}

response = requests.post(url, json=house)
print(response.json())


