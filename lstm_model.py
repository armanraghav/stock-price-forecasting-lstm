import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import math
import joblib


def download_data(ticker='AAPL', start_date='2015-01-01', end_date='2023-01-01'):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_csv(f'data/{ticker}_data.csv')
    print(df.columns)
    return df


def add_technical_indicators(df, ticker='AAPL'):
    close = df['Close'][ticker]

    df[('SMA_20', ticker)] = close.rolling(window=20).mean()
    df[('SMA_50', ticker)] = close.rolling(window=50).mean()
    df[('EMA_20', ticker)] = close.ewm(span=20, adjust=False).mean()
    df[('EMA_50', ticker)] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df[('RSI_14', ticker)] = rsi

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df[('MACD', ticker)] = macd
    df[('MACD_signal', ticker)] = signal

    df = df.dropna()
    return df


def preprocess_data(df, ticker='AAPL'):
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50',
                'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 'MACD_signal']
    feature_cols = [(feat, ticker) for feat in features]

    data = df[feature_cols]
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Save scaler to file
    joblib.dump(scaler, 'scaler.save')
    print("Scaler saved to scaler.save")

    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - 60:]

    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i - time_step:i, :])
            Y.append(data[i, 0])  # Close price
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    return X_train, y_train, X_test, y_test, scaler, train_len


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def plot_predictions(scaler, train_len, df, predictions, y_test, ticker='AAPL'):
    close_col = ('Close', ticker)
    train = df[close_col][:train_len]
    valid = df[close_col][train_len:]
    valid = valid.to_frame()
    valid.columns = ['Close']
    valid['Predictions'] = scaler.inverse_transform(
        np.hstack((predictions, np.zeros((len(predictions), scaler.n_features_in_-1)))))[:, 0]

    plt.figure(figsize=(16, 8))
    plt.title('Stock Price Prediction Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train.index, train, label='Train Actual')
    plt.plot(valid.index, valid['Close'], label='Test Actual')
    plt.plot(valid.index, valid['Predictions'], label='Test Predictions')
    plt.legend(loc='lower right')
    plt.show()


def predict_future_prices(model, last_sequence, future_days=30, scaler=None):
    current_seq = last_sequence
    predictions = []

    for _ in range(future_days):
        pred = model.predict(current_seq[np.newaxis, :, :])[0, 0]
        predictions.append(pred)
        new_step = np.zeros((current_seq.shape[1],))
        new_step[0] = pred
        current_seq = np.vstack([current_seq[1:], new_step])

    if scaler is not None:
        predictions = scaler.inverse_transform(
            np.hstack((np.array(predictions).reshape(-1, 1), np.zeros((future_days, scaler.n_features_in_-1)))))
        return predictions[:, 0]
    else:
        return np.array(predictions)


def main():
    ticker = 'AAPL'
    df = download_data(ticker)
    df = add_technical_indicators(df, ticker)

    X_train, y_train, X_test, y_test, scaler, train_len = preprocess_data(df, ticker)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=20,
              batch_size=64,
              verbose=1,
              callbacks=[early_stop, checkpoint])

    model = load_model('best_model.h5')

    predictions = model.predict(X_test)

    y_test_unscaled = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaler.n_features_in_-1)))))
    predictions_unscaled = scaler.inverse_transform(
        np.hstack((predictions, np.zeros((len(predictions), scaler.n_features_in_-1)))))
    rmse = math.sqrt(mean_squared_error(y_test_unscaled[:, 0], predictions_unscaled[:, 0]))
    mae = mean_absolute_error(y_test_unscaled[:, 0], predictions_unscaled[:, 0])
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    plot_predictions(scaler, train_len, df, predictions_unscaled[:, 0].reshape(-1, 1), y_test, ticker)

    last_sequence = X_test[-1]
    future_preds = predict_future_prices(model, last_sequence, future_days=30, scaler=scaler)

    print("Future 30 days prices prediction:")
    print(future_preds)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), future_preds, marker='o')
    plt.title('Future 30 Days Stock Price Predictions')
    plt.xlabel('Days Ahead')
    plt.ylabel('Predicted Close Price USD ($)')
    plt.show()


if __name__ == "__main__":
    main()