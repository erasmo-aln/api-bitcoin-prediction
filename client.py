import requests
import pandas as pd

from utils.store_results import store_prediction


payload = {
    'Model': 'Machine Learning'
}

api_response = requests.post('http://localhost:3000/predict', json=payload).json()

print('Accessing API...')
store_prediction(api_response)
print('Prediction stored succesfully.')
