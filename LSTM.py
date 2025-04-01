import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the data
df = pd.read_csv('d:\CODE\Projects\Sea_Level_Rise\Main\global_mean_sea_level_1993-2024.csv')
# Plot the data
data = df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1)

df.head(15)

#Plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'])
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.show()

#Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1))

#sequences for LSTM
time_steps = 15
def create_sequences(data, time_steps = 15):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)
X, y = create_sequences(scaled_data, time_steps=15)

# Split into train/test
train_size = int(0.8*len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# simply do scatterplot(columnName) if you want to review the many other data
def scatterplot(col):
    plt.scatter(df['YearPlusFraction'], df[col], alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel(col)
    plt.show()
scatterplot('SmoothedGMSLWithGIASigremoved')

#LSTM model
model = Sequential()
model.add(LSTM(units = 50, activation = 'tanh', input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences = True))
model.add(LSTM(units = 30, activation = 'tanh', return_sequences = True ))
model.add(Dropout(0.1))
model.add(LSTM(units = 25, activation = 'tanh', return_sequences = False))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss='mse')

#Train
history = model.fit(
    X_train, y_train,
    epochs = 100,
    batch_size = 16,
    validation_data = (X_test, y_test),
    verbose = 1
)

#predict on test data
y_pred = model.predict(X_test)

#inverse scaling
y_test_actual = scaler.inverse_transform(y_test)
y_pred_actual = scaler.inverse_transform(y_pred)

#plot predictions vs actual
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label = 'Actual', color = 'blue')
plt.plot(y_pred_actual, label = 'Predicted', color = 'red', linestyle='--')
plt.xlabel('Time Step (Test Data Index)')
plt.ylabel('Sea Level (mm)')
plt.title('LSTM Prediction vs Actual')
plt.legend()
plt.show()
# Add labels and title
df.describe()
df.shape
df.info()
plt.figure(figsize=(12,6))
plt.plot(df['YearPlusFraction'][time_steps+train_size:], y_test_actual, label = 'Actual', color = 'blue')
plt.plot(df['YearPlusFraction'][time_steps+train_size:], y_pred_actual, label = 'Predicted', color = 'red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Sea LeVeL (mm)')
plt.title('LSTM Prediction vs Actual')
plt.legend()
plt.show()

#start with the last time steps data points
current_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
future_predictions = []
for i in range(50):
    next_pred = model.predict(current_sequence)
    future_predictions.append(next_pred[0, 0])
    #update
    current_sequence = np.append(current_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis = 1)
#inverse scaling
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

#generate future years
last_year = df['YearPlusFraction'].max()
future_years = np.arange(last_year + 1, last_year + 51)

plt.figure(figsize=(12,6))
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'], label = 'Historical Data', color = 'blue')
plt.plot(future_years, future_predictions, label='LSTM Future Predictions', color = 'green', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.title('LSTM Future Predictions: Next 50 years')
plt.legend()
plt.show()
