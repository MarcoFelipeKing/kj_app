# This a LSTM neural network for predicting the air quality in KJ's data for sensor 1149

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import data from csv file in data directory
df = pd.read_csv('data/1149_test.csv', sep=",")


#delete timestamp column and create one with index
del df['timestamp']
df.index = pd.to_datetime(df.index)

# drop all rows with NA values
df = df.dropna()

# split the training and testing data
train = df[:int(len(df)*0.8)]
test = df[int(len(df)*0.8):]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

scaled_train[:10]

#import keras libraries and packages
from keras.preprocessing.sequence import TimeseriesGenerator

# define generator
n_input = 12
n_features=3
genarator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# write length of generator to a file called generator_length.txt
with open('generator_length.txt', 'w') as f:
    f.write(str(len(genarator)))

X,y=genarator[0]

#import LSTM model from keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# write model.summary() to a file called model.txt
with open('model.txt', 'w') as f:
    f.write(str(model.summary()))

# fit model
model.fit(genarator, epochs=50, verbose=1)

# plot the loss_per_epoch graph and save to a file called loss_per_epoch.png
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.savefig('loss_per_epoch.png')


last_train_batch = scaled_train[-n_input:]

last_train_batch = last_train_batch.reshape((1, n_input, n_features))

model.predict(last_train_batch)

scaled_test[0]

test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# write the test_predictions array to a file called test_predictions.txt
with open('test_predictions.txt', 'w') as f:
    f.write(str(test_predictions))

true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions

test.plot(figsize=(14,5))

plt.savefig('test_predictions.png')


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(test['Production'],test['Predictions']))
print(rmse)

#save trained model to a file called model.h5
model.save('model.h5')


