import pickle
import xgboost as xgb

with open ('model_xgb.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

house = {
    'property_type': 'apartment',
     'lat': -34.5438853785,
     'lon': -58.4779876511999,
     'surface_total': 40,
     'surface_covered': 35,
     'rooms': 4,
     'barrio': 'SAAVEDRA',
     'comuna': 12
}

X = dv.transform([house])

feature_names = list(dv.get_feature_names_out())

d = xgb.DMatrix(X, feature_names=feature_names)

suggested_price = model.predict(d)
print(f"Precio sugerido para la propiedad: USD {suggested_price[0]:,.0f}")