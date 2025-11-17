
# Importación de librerías

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle


# Lectura del dataset

url = "https://github.com/caro6852lq/MachineLearning_PredictPricesProperties/raw/refs/heads/main/Data/Dataset_Inmuebles.xlsx"

get_ipython().system('pip install openpyxl')

df1 = pd.read_excel(url, sheet_name=0)   # primera hoja
df2 = pd.read_excel(url, sheet_name=1)   # segunda hoja
df3 = pd.read_excel(url, sheet_name=2)   # tercera hoja

df1df2 = df1.merge(df2, how = 'left', on='ID') # hago merge de las dos primeras hojas a través del ID
df = df1df2.merge(df3, how = 'left', on='ID') # agrego al merge la 3° hoja


# Limpieza de Datos

## Ajusto la columna de precio
df["price_usd"] = (
    df["price_usd"]
      .str.replace("k", "", regex=True)   # quita k/K finales
      )
df["price_usd"] = df["price_usd"].astype("float64")
df["price_usd"] = df["price_usd"]*100


#Transformar Latitud a float64
df['lat'] = df['lat'].str.replace('.', '', regex=False)  
df['lat'] = df['lat'].apply(lambda x: x[:3] + '.' + x[3:]) 
df['lat'] = df['lat'].astype('float64') 


#Transformar lonitud a float64
df['lon'] = df['lon'].str.replace('.', '', regex=False)  
df['lon'] = df['lon'].apply(lambda x: x[:3] + '.' + x[3:]) 
df['lon'] = df['lon'].astype('float64') 

#Reemplazo los nulos 
df.fillna({'property_type': 'Sin Dato'}, inplace=True)
df = df.fillna(0)

#Filtro valores atípicos
media = df["price_usd"].mean()
desv_std = df["price_usd"].std()
LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
df = df[(df["price_usd"] >= LI_DS) & (df["price_usd"] <= LS_DS)]


media = df["surface_total"].mean()
desv_std = df["surface_total"].std()
LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
df = df[(df["surface_total"] >= LI_DS) & (df["surface_total"] <= LS_DS)]
df = df[(df["surface_total"] >= 10)]
df = df[(df["rooms"] < 17) ]

df = df[df["property_type"] !='Sin Dato']

df=df[["ID", "property_type",'lat', 'lon','price_usd', 'surface_total', 'surface_covered','rooms',
       'barrio', 'comuna']]

# Defino las variables numéricas
numerical = ['lat', 'lon', 'surface_total','surface_covered', 'rooms']

# Defino las variables categóricas
categorical = ['property_type','barrio', 'comuna']


# Divido el Dataset

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.price_usd.values
del df_full_train['price_usd']


def train(df_train, y_train):
    dicts_full_train = df_full_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts_full_train)

    feature_names = list(dv.get_feature_names_out())

    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                        feature_names=feature_names)

    xgb_params = {
        'eta': 0.3, 
        'max_depth': 5, 
        'min_child_weight': 1, 

        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params,dfulltrain, num_boost_round=100)

    return dv, model

dv, model = train(df_full_train, y_full_train)


output_file = "model_xgb.bin"

f_out = open(output_file, 'wb')
pickle.dump((dv,model),f_out)
f_out.close


with open (output_file, 'wb') as f_out:
    pickle.dump((dv,model),f_out)

