#!/usr/bin/env python
# coding: utf-8

# ## Importación de librerías

# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle


# ## Lectura del dataset

# In[2]:


url = "https://github.com/caro6852lq/MachineLearning_PredictPricesProperties/raw/refs/heads/main/Data/Dataset_Inmuebles.xlsx"


# In[3]:


get_ipython().system('pip install openpyxl')


# In[4]:


df1 = pd.read_excel(url, sheet_name=0)   # primera hoja
df2 = pd.read_excel(url, sheet_name=1)   # segunda hoja
df3 = pd.read_excel(url, sheet_name=2)   # tercera hoja


# In[5]:


df1df2 = df1.merge(df2, how = 'left', on='ID') # hago merge de las dos primeras hojas a través del ID


# In[6]:


df = df1df2.merge(df3, how = 'left', on='ID') # agrego al merge la 3° hoja


# ## Limpieza de Datos

# In[7]:


## Ajusto la columna de precio
df["price_usd"] = (
    df["price_usd"]
      .str.replace("k", "", regex=True)   # quita k/K finales
      )
df["price_usd"] = df["price_usd"].astype("float64")
df["price_usd"] = df["price_usd"]*100


# In[8]:


#Transformar Latitud a float64
# Primero, debes reemplazar los puntos incorrectos. Usaremos regex para transformar el formato.
df['lat'] = df['lat'].str.replace('.', '', regex=False)  # Eliminar todos los puntos
df['lat'] = df['lat'].apply(lambda x: x[:3] + '.' + x[3:]) #sumo el punto dp de los tres primeros valores
df['lat'] = df['lat'].astype('float64') # paso a float


# In[9]:


#Transformar lonitud a float64
# Primero, debes reemplazar los puntos incorrectos. Usaremos regex para transformar el formato.
df['lon'] = df['lon'].str.replace('.', '', regex=False)  # Eliminar todos los puntos
df['lon'] = df['lon'].apply(lambda x: x[:3] + '.' + x[3:]) #sumo el punto dp de los tres primeros valores
df['lon'] = df['lon'].astype('float64') # paso a float


# In[10]:


#Reemplazo los nulos por "sin dato"
df.fillna({'property_type': 'Sin Dato'}, inplace=True)


# In[11]:


#Completar nulos
df = df.fillna(0)


# In[12]:


#Filtro valores atípicos
media = df["price_usd"].mean()
desv_std = df["price_usd"].std()

LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std

df = df[(df["price_usd"] >= LI_DS) & (df["price_usd"] <= LS_DS)]


# In[13]:


## Superficie Total por DS

media = df["surface_total"].mean()
desv_std = df["surface_total"].std()

LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std

df = df[(df["surface_total"] >= LI_DS) & (df["surface_total"] <= LS_DS)]
df = df[(df["surface_total"] >= 10)]


# In[14]:


## En este caso tomo un modo arbitrario
# Filtramos los valores dentro del rango para la superficie total
df = df[(df["rooms"] < 17) ]


# In[15]:


## Borro los registros "Sin Dato" para tipo de propiedad
df = df[df["property_type"] !='Sin Dato']


# In[16]:


## Saco: description, title, floor (x cantidad de nulos), columnas calculadas, expensas porque la mayoría tiene valor $0
df=df[["ID", "property_type",'lat', 'lon','price_usd', 'surface_total', 'surface_covered','rooms',
       'barrio', 'comuna']]


# In[17]:


# Defino las variables numéricas
numerical = ['lat', 'lon', 'surface_total','surface_covered', 'rooms']


# In[18]:


# Defino las variables categóricas
categorical = ['property_type','barrio', 'comuna']


# ### Divido el Dataset

# In[19]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[20]:


df_full_train = df_full_train.reset_index(drop=True)


# In[21]:


y_full_train = df_full_train.price_usd.values


# In[22]:


del df_full_train['price_usd']


# In[23]:


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


# In[24]:


dv, model = train(df_full_train, y_full_train)


# In[25]:


output_file = "model_xgb.bin"


# In[26]:


f_out = open(output_file, 'wb')
pickle.dump((dv,model),f_out)
f_out.close


# In[27]:


with open (output_file, 'wb') as f_out:
    pickle.dump((dv,model),f_out)

