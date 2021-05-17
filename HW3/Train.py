import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import numpy as np
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

file_list = os.listdir('training_data')
model_con = Sequential([
  Dense(168, activation='relu'),
  Dropout(0.2),
  Dense(336, activation='relu'),
  Dropout(0.2),
  Dense(48, activation='relu'),
  Dropout(0.2),
  Dense(24, activation='relu')
])
model_con.compile(optimizer='adam', loss='mse', metrics=['mae'])

model_gen = Sequential([
  Dense(168, activation='relu'),
  Dropout(0.2),
  Dense(336, activation='relu'),
  Dropout(0.2),
  Dense(48, activation='relu'),
  Dropout(0.2),
  Dense(24, activation='relu')
])
model_gen.compile(optimizer='adam', loss='mse', metrics=['mae'])


train_X_con = []
train_Y_con = []

train_X_gen = []
train_Y_gen = []

for fname in file_list:
    df = pd.read_csv('training_data/' + fname)
    for i in range(int(df.shape[0] / 24 - 7)):
        train_X_con.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'consumption'].tolist())
        train_Y_con.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'consumption'].tolist())
        
        train_X_gen.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'generation'].tolist())
        train_Y_gen.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'generation'].tolist())

train_X_con = np.array(train_X_con)
train_Y_con = np.array(train_Y_con)
train_X_gen = np.array(train_X_gen)
train_Y_gen = np.array(train_Y_gen)

history_con = model_con.fit(
    train_X_con,
    train_Y_con,
    batch_size=64,
    epochs=1000,
    validation_split=0.2
)
        
history_gen = model_gen.fit(
    train_X_gen,
    train_Y_gen,
    batch_size=64,
    epochs=1000,
    validation_split=0.2
)
        
model_con.save('con.h5')
model_gen.save('gen.h5')