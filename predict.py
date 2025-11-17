import pickle

from typing import Dict, Any
import uvicorn
from fastapi import FastAPI

app = FastAPI(tittle="converted-prediction")

with open ('model_xgb.bin', 'rb') as f_in:
       pipeline = pickle.load(f_in)

def predict_single(house):
    result = pipeline.predict(house)
    return float(result)


@app.post("/predict")
def predict(house: Dict[str, Any]):
    price = predict_single(house)

    result = {
        'converted_probability': price,
    }

    return result


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9696)