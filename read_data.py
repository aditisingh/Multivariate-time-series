import pandas
import tensorflow as tf
from tensorflow import keras

def read_file(filename):
    df = pandas.read_excel(filename)
    FORMAT = ['file','bx','by','bz','bl','bm','bn','bmag','vx','vy','vz','vmag','np','tpar','tper','goal']
    df_selected = df[FORMAT]
    return df_selected

filename='challenge_dataset.xlsx'
df=read_file(filename)

start_idx=0
step=89

df[start_idx:start_idx+step]

#normalize all features

for col in df.columns :
    if col not in ['file','goal']:
        std=df[col].std()
        mean=df[col].mean()
        df[col]=(df[col]-mean)/std

series_size=89

model = keras.Sequential()
model.add(keras.layers.Embedding(series_size, 14))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(14, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.sigmoid))

model.summary()
