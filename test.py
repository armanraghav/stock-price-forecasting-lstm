import requests
import numpy as np

url = "http://localhost:5000/predict"

# Generate 60 time steps, each with 12 features (random floats for testing)
dummy_input = np.random.rand(60, 12).tolist()

payload = {
  "features_sequence": dummy_input
}

response = requests.post(url, json=payload)
print(response.json())
