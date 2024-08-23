import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
dill.settings['recurse'] = True
with open('model/sber_podpiska_predict_model.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: object
    client_id: object
    visit_date: object
    visit_time: object
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_model: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object
    hit_date: object
    hit_number: float


class Prediction(BaseModel):
    utm_medium: object
    result: int


@app.get('/status')
def status():
    return "I'm OKAY"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'utm_medium': form.utm_medium,
        'result': y[0]
    }


