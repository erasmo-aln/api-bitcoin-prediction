import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from utils.indicators import calculate_indicators
import yfinance as yf


tags_metadata = [{'name': 'Bitcoin-Prediction', 'description': 'Predict the bitcoin price with ML'}]

app = FastAPI(
    title = 'Bitcoin Price API',
    description = 'Predicting Bitcoin price with Machine Learning',
    version = '1.0',
    contact = {'name': 'Erasmo Neto', 'e-mail': 'erasmo.aln@gmail.com'},
    openapi_tags = tags_metadata
)

class Features(BaseModel):
    Model: str


@app.get('/')
def message():
    text = 'This is a bitcoin price prediction API. Use the right method.'
    return text


@app.post('/predict', tags=['Bitcoin-Prediction'])
async def predict(Features: Features):
    btc_ticker = yf.Ticker('BTC-USD')

    history_btc = btc_ticker.history(period='200d', actions=False)
    history_btc = history_btc.tz_localize(None)
    history_btc = calculate_indicators(history_btc)
    history_btc = history_btc.sort_index(ascending=False)

    input_data = history_btc.iloc[0, :]
    input_data = input_data.fillna(0)
    input_data = input_data.array
    input_data = input_data.reshape(1, -1)

    scaler = load('model/scaler.bin')

    input_data = scaler.transform(input_data)

    Model = Features.Model

    if Model == 'Machine Learning':
        model_path = 'model/model.joblib'
        model = load(model_path)

    prediction = model.predict(input_data)

    last_price = history_btc.iloc[0, 3]

    api_response = {
        'Model': Model,
        'Last Price': round(last_price, 2),
        'Prediction': round(prediction.tolist()[0], 2)
    }

    return api_response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3000)
