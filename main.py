import os
import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.training.ml.data import cat_features, process_data
from src.training.ml.model import inference

# load model
with open(os.path.join(os.getcwd(), "model/encoder.pkl"), 'rb') as f:
    encoder = pickle.load(f)

with open(os.path.join(os.getcwd(), "model/dct_model.pkl"), 'rb') as f:
    model = pickle.load(f)
# from model/model_score.txt
_lb = ['<=50K', '>50K']


class InputData(BaseModel):
    # Example field with a hyphen in the name, use alias
    age: int
    workclass: Literal['Private', 'Self-emp-not-inc', 'State-gov',
                       'Self-emp-inc', 'Federal-gov', 'Without-pay',
                       'Never-worked', 'Local-gov', '?']
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: Literal['Married-civ-spouse', 'Never-married',
                            'Divorced', 'Separated', 'Widowed',
                            'Married-spouse-absent', 'Married-AF-spouse'
                            ] = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: Literal['White', 'Black', 'Asian-Pac-Islander',
                  'Amer-Indian-Eskimo', 'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int = Field(alias='capital-gain')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    # Include an example
    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "white",
                "sex": "Male",
                "capital-gain": 2174,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


app = FastAPI()


@app.get("/", response_class=ORJSONResponse)
async def welcome_page():
    return ORJSONResponse(content={"message": "Welcome to the Udacity MLops L4 model inference API!"})  # NOQA:E501


@app.post("/predict", response_class=ORJSONResponse)
async def predict(input: InputData):
    df = pd.DataFrame(input.model_dump(by_alias=True), index=[0])
    # preprocess data
    data, _, _, _ = process_data(X=df, categorical_features=cat_features,   # type: ignore  # NOQA:E501
                                 training=False, encoder=encoder)
    pred = inference(model, data)[0]
    pred = _lb[pred]
    return ORJSONResponse(content={"pred": pred})  # type: ignore  # NOQA:E501
