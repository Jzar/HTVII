
# coding: utf-8

# In[64]:

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# In[65]:

eth_df = pd.read_json('ethereum_prices.json')
eth_df


# In[66]:

eth_df = eth_df.iloc[1:,:]
eth_df.reset_index(inplace=True,drop=True)
eth_df


# In[67]:

eth_df.drop('date',axis=1,inplace=True)
eth_df.drop('weightedAverage',axis=1,inplace=True)
eth_df.head()


# In[68]:

eth_np = eth_df.as_matrix()


# In[69]:

from sklearn.preprocessing import scale


# In[70]:

X = eth_np[:-1,:]
y = eth_np[1:,0]


# In[71]:

X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)


# In[72]:

train = eth_np[:4469,:]
test = eth_np[4469:, :]


# In[73]:

eth_np.shape


# In[74]:

X_train = X[:4469]
y_train = y[:4469]
X_test = X[4469:]
y_test = y[4469:]


# In[75]:

X_train.shape


# In[76]:

y_test


# In[77]:

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[78]:

from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# In[79]:

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return abs(x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})



# In[80]:

model = Sequential()

model.add(LSTM(50, input_shape = (X_train.shape[1], X_train.shape[2]),
              return_sequences = True))

model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100, return_sequences = True))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(50, activation='linear'))
model.add(Dense(25, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(5, activation='linear'))
model.add(Dense(1, activation='linear'))
model.add(Activation(custom_activation))


# In[81]:

model.compile(optimizer='Adam',loss='mae')


# In[82]:

hist = model.fit(
    X_train,
    y_train,
    validation_split = 0.1)


# In[83]:

model.evaluate(X_test, y_test)


# In[84]:

from matplotlib import pyplot as plt

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()
plt.show()


# In[85]:

predict = model.predict(X_test)


# In[86]:

plt.plot(predict, label='prediction')
plt.plot(y_test, label='actual')
plt.legend()
plt.show()


# In[28]:

from keras.callbacks import EarlyStopping


# In[23]:

model2 = Sequential()

model2.add(LSTM(100, input_shape = (X_train.shape[1], X_train.shape[2]),return_sequences=True))

model2.add(LSTM(100, return_sequences=True))
model2.add(Dense(200))
model2.add(Dropout(0.2))
model2.add(Dense(300))
model2.add(Dropout(0.2))
model2.add(Dense(200))
model2.add(Dropout(0.2))
model2.add(Dense(100))
model2.add(Dropout(0.2))

model2.add(Flatten())

model2.add(Dense(50, activation='linear'))
model2.add(Dense(25, activation='linear'))
model2.add(Dense(10, activation='linear'))
model2.add(Dense(5, activation='linear'))
model2.add(Dense(1, activation='linear'))


# In[24]:

model2.compile(optimizer='Adam',loss='mae')


# In[35]:

early = EarlyStopping(monitor='val_loss', mode='auto', patience = 5)


# In[36]:

hist2 = model2.fit(
    X_train,
    y_train,
    validation_split = 0.1,
    callbacks=[early])
plt.plot(hist2.history['loss'], label='train')
plt.plot(hist2.history['val_loss'], label='val')
plt.legend()
plt.show()


# In[37]:

model2.evaluate(X_test, y_test)


# In[62]:

predict2 = model2.predict(X_train)


# In[63]:

#plt.plot(predict, label='prediction1')
plt.plot(predict2[:20], label='prediction2')
plt.plot(y_train[:20], label='actual')
plt.legend()
plt.show()


# In[40]:

eth_df.shape


# In[87]:

model.save('eth_predictor_v1.h5')
model2.save('eth_predictor_v2.h5')


# In[42]:

model.predict(X_train).shape


# In[88]:

all_prediction = np.concatenate((np.zeros((1,1),dtype=float),model.predict(X_train), model.predict(X_test)),axis=0)


# In[89]:

all_prediction.shape


# In[90]:

final_df = eth_df.copy()


# In[91]:

final_df.shape


# In[92]:

final_df['prediction'] = all_prediction


# In[93]:

final_df


# In[94]:

final_df.to_csv('predictions.csv')


# In[ ]:
