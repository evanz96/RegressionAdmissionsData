import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

dataset = pd.read_csv('admissions_data.csv')
features = dataset.iloc[:, 1:7]
labels = dataset.iloc[:, -1]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 1)
numerical_features = features.select_dtypes(include = ['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([('scale', StandardScaler(), numerical_columns)], remainder = 'passthrough') # all cols are numerical
features_train_set = ct.fit_transform(features_train)
features_test_set = ct.transform(features_test)
# features_train_set = pd.DataFrame(features_train)
# features_test_set = pd.DataFrame(features_test) # necessary?
model = Sequential()
model.add(InputLayer(input_shape = (features.shape[1], )))
model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.1))
model.add(Dense(8, activation = 'softmax'))
model.add(Dense(1))
opt = Adam(learning_rate = 0.001)
stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)
model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)
history = model.fit(features_train_set, labels_train, epochs = 100, batch_size = 3, verbose = 1, validation_split = 0.12, callbacks = [stop])
res_mse, res_mae = model.evaluate(features_test_set, labels_test, verbose = 0)
print("final loss = {0}, final mae = {1}".format(res_mse, res_mae))
predicted_values = model.predict(features_test_set) 
print(r2_score(labels_test, predicted_values))

# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
  # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()
