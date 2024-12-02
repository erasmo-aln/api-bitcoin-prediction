import os
import pandas as pd


def store_prediction(api_response):

    api_data = {
        'Model': [api_response['Model']],
        'Last Price': [api_response['Last Price']],
        'Prediction': [api_response['Prediction']],
    }
    prediction_data = pd.DataFrame(api_data)

    is_data_populated = len(os.listdir('data/')) > 0
    print('Saving data to "data/prediction_history.csv" directory...')
    if is_data_populated:
        prediction_data.to_csv('data/prediction_history.csv', sep=';', mode='a', header=False, index=False)

    else:
        prediction_data.to_csv('data/prediction_history.csv', sep=';', index=False)
