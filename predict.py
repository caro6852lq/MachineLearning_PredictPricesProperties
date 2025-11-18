import pickle

from typing import Dict, Any
import uvicorn
from fastapi import FastAPI
import xgboost as xgb

app = FastAPI(tittle="price-prediction")

with open ('model_xgb.bin', 'rb') as f_in:
       (dv, model) = pickle.load(f_in)

def predict_single(house: Dict[str, Any]):
    X = dv.transform([house])
    feature_names = list(dv.get_feature_names_out())
    d = xgb.DMatrix(X, feature_names=feature_names)
    pred = model.predict(d)
    return float(pred[0])

@app.post("/predict")
def predict(house: Dict[str, Any]):
    price = predict_single(house)

    result = {
        'suggested_price': price,
    }

    return result


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9696)