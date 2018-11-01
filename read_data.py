import pandas
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import metrics, cross_validation

import numpy as np

def read_file(filename):
    df = pandas.read_excel(filename)
    FORMAT = ['file','bx','by','bz','bl','bm','bn','bmag','vx','vy','vz','vmag','np','tpar','tper','goal']
    df_selected = df[FORMAT]
    return df_selected

filename='challenge_dataset.xlsx'
df=read_file(filename)

#normalize all features
for col in df.columns :
    if col not in ['file','goal']:
        std=df[col].std()
        mean=df[col].mean()
        df[col]=(df[col]-mean)/std

start_idx=0
step=89

df_data=[]
df_label=[]

for start_idx in range(0,len(df),step):
    df_data.append(df[start_idx:start_idx+step][['bx','by','bz','bl','bm','bn','bmag','vx','vy','vz','vmag','np','tpar','tper']].values)
    df_label.append(df['goal'][start_idx])

X=np.array(np.reshape(df_data,(205,89*14)))
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
predicted = cross_validation.cross_val_predict(clf, X,df_label, cv=10)
# k_means = KMeans(n_clusters=len(np.unique(df_label)))
# labels = k_means.labels_

print metrics.accuracy_score(df_label, predicted)
print metrics.classification_report(df_label, predicted)

from statsmodels.tsa.seasonal import seasonal_decompose
series = df1
result = seasonal_decompose(series, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)

series_size=89
n_features=14

model = keras.Sequential()
model.add(keras.layers.Embedding(series_size, n_features))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(n_features, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


df_data=df[['bx','by','bz','bl','bm','bn','bmag','vx','vy','vz','vmag','np','tpar','tper']]
df_label=df['goal']

model.fit(df_data,df_label)
