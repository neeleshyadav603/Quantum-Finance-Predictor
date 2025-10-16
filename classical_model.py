import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# --- Imports for the powerful LSTM model ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# This is your old function. It's good to keep for reference.
def train_classical_model(data: pd.DataFrame, n_past=60, n_future=30):
    """
    Trains a classical Linear Regression model and predicts future stock prices.
    """
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    X, y = [], []
    for i in range(n_past, len(scaled_prices)):
        X.append(scaled_prices[i-n-past:i, 0])
        y.append(scaled_prices[i, 0])
    X, y = np.array(X), np.array(y)
    model = LinearRegression()
    model.fit(X, y)
    y_pred_train = model.predict(X)
    mse = mean_squared_error(y, y_pred_train)
    last_n_days_scaled = scaled_prices[-n_past:]
    current_batch = last_n_days_scaled.reshape(1, n_past)
    future_predictions_scaled = []
    for _ in range(n_future):
        next_prediction_scaled = model.predict(current_batch)
        future_predictions_scaled.append(next_prediction_scaled[0])
        current_batch = np.append(current_batch[:, 1:], [[next_prediction_scaled[0]]], axis=1)
    predicted_prices = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    return mse, predicted_prices.flatten().tolist()


# --- This is the NEW LSTM model function that your API needs ---
def train_lstm_model(data: pd.DataFrame, n_past=60, n_future=30):
    """
    Trains a powerful LSTM model for time-series forecasting.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(n_past, len(scaled_prices)):
        X.append(scaled_prices[i-n_past:i, 0])
        y.append(scaled_prices[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape the data for the LSTM model [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model architecture
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Training LSTM model... (This may take a few moments)")
    model.fit(X, y, batch_size=1, epochs=5, verbose=1)

    # --- Prediction Logic ---
    last_n_days = scaled_prices[-n_past:]
    current_batch = last_n_days.reshape((1, n_past, 1))
    future_predictions_scaled = []

    for _ in range(n_future):
        next_pred_scaled = model.predict(current_batch, verbose=0)[0]
        future_predictions_scaled.append(next_pred_scaled)
        new_batch_entry = next_pred_scaled.reshape((1, 1, 1))
        current_batch = np.append(current_batch[:, 1:, :], new_batch_entry, axis=1)

    predicted_prices = scaler.inverse_transform(np.array(future_predictions_scaled))
    
    y_pred_train_scaled = model.predict(X, verbose=0)
    mse = mean_squared_error(y, y_pred_train_scaled)
    
    print("LSTM training and prediction complete.")
    return mse, predicted_prices.flatten().tolist()