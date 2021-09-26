import uvicorn
from fastapi import FastAPI
from Banknotes import BankNote
import numpy as np
import pickle
import pandas as pd
app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))


@app.get('/')
def index():
    return {'Namaste': ['Hello', 'Welcome']}


@app.get('/welcome')
def get_name(name: str):
    return {'Welcome to fastapi': f'{name}'}


@app.post('/predict')
def predict_note(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    if(prediction[0] > 0.5):
        prediction = 'Fake Note'
    else:
        prediction = 'Real Note'
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn app:app --reload
# web: gunicorn app:app
