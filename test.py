import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import yfinance as yf
yf.pdr_override()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

crypto_currency = 'BTC'
against_currency = 'USD'

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

data = pdr.get_data_yahoo(f'{crypto_currency}-{against_currency}', start=start, end=end)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60
now = dt.datetime.now() - start
now = now.days


x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#Create XGBoost Model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

model.fit(x_train, y_train)


# Testing Model

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
test_data = pdr.get_data_yahoo(f'{crypto_currency}-{against_currency}', start= test_start, end= test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])
    
x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices.reshape(-1, 1))


plt.plot(actual_prices, color='blue', label='Actual Price')
plt.plot(prediction_prices, color='green', label='Predicted Price')
plt.axvline(now , color='red', label = 'Current Date')
plt.title(f'{crypto_currency} Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()