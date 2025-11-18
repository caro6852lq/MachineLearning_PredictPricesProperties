import requests

url = "http://localhost:9696/predict"

house= {
    "property_type": "apartment",
     "lat": -34.5438853785,
     "lon": -58.4779876511999,
     "surface_total": 40,
     "surface_covered": 35,
     "rooms": 4,
     "barrio": "SAAVEDRA",
     "comuna": 12

}

response = requests.post(url, json=house)
print(response.json())


