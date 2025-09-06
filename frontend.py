import streamlit as st
import yfinance as yf
import numpy as np
import requests
import pandas as pd
import plotly.graph_objs as go


# Set page config and custom styles
st.set_page_config(page_title="📈 Stock Price Forecasting", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
            color: #0f111b;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton > button {
            background-color: #0072C6;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 45px;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #005a99;
            color: #d9e2ec;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1.5px solid #0072C6;
            padding-left: 12px;
            height: 40px;
            font-size: 16px;
        }
        .stNumberInput>div>div>input {
            border-radius: 10px;
            border: 1.5px solid #0072C6;
            height: 40px;
            font-size: 16px;
            text-align: center;
        }
        .title {
            font-size: 48px;
            font-weight: 700;
            color: #0072C6;
            margin-bottom: -10px;
        }
        .subtitle {
            font-size: 20px;
            color: #42526e;
            margin-bottom: 40px;
        }
    </style>
""", unsafe_allow_html=True)


# Title and subtitle section
st.markdown('<div class="title">📈 Stock Price Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict future stock prices with LSTM-based deep learning model</div>', unsafe_allow_html=True)


# Input form in three columns
col1, col2, col3 = st.columns([3, 2, 3])


with col1:
    ticker_input = st.text_input("Enter Stock Ticker Symbol", placeholder="e.g. AAPL, MSFT, TSLA")


with col2:
    days_to_predict = st.number_input("Days to Forecast", min_value=1, max_value=60, value=30, step=1)


with col3:
    api_key_input = st.text_input("Enter API Key", type="password", help="Your API key for authentication")


# API URL configurable with smaller input below
api_url = st.text_input("API URL", "http://localhost:5000/predict", help="Backend API Endpoint URL")


st.markdown("---")


# Prediction button centered
predict_button = st.button("Predict Next Days Price")


def fetch_features(ticker):
    df = yf.download(ticker, period='150d')  # Increased period to 150 days
    if df.empty:
        st.error("No data found for ticker symbol.")
        return None, None

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df = df.dropna()

    if len(df) < 60:
        st.error(f"Not enough data after feature calculation for ticker {ticker}. Required 60 time steps, got {len(df)}.")
        return None, None

    df_inputs = df[['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50',
                    'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 'MACD_signal']].tail(60)
    return df, df_inputs.values.tolist()


def call_api(features_seq, days, api_url, api_key):
    headers = {
        'x-api-key': api_key
    }
    payload = {
        "features_sequence": features_seq,
        "days": days
    }
    response = requests.post(api_url, json=payload, headers=headers)
    return response


def plot_interactive_line_chart(data, title, yaxis_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index if hasattr(data, 'index') else list(range(1, len(data) + 1)),
        y=data.values if hasattr(data, 'values') else data,
        mode='lines+markers',
        line=dict(color='#0072C6'),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date' if hasattr(data, 'index') else 'Days Ahead',
        yaxis_title=yaxis_title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0f111b'),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    st.plotly_chart(fig, use_container_width=True)


if predict_button:
    if not ticker_input:
        st.warning("Please enter a stock ticker symbol.")
    elif not api_key_input:
        st.warning("Please enter your API key.")
    else:
        with st.spinner("Fetching data and predicting..."):
            data_df, features_seq = fetch_features(ticker_input.strip().upper())

            # Debug prints for troubleshooting
            st.write("Features sequence length:", len(features_seq) if features_seq else 'None')
            st.write("Example first feature vector:", features_seq[0] if features_seq else 'None')
            st.write("Days to predict:", days_to_predict)

            if features_seq is not None:
                try:
                    response = call_api(features_seq, days_to_predict, api_url, api_key_input)
                    if response.status_code == 401:
                        st.error("Unauthorized: Invalid API key")
                    elif response.status_code == 400:
                        st.error(f"Bad Request (400): {response.text}")
                    else:
                        response.raise_for_status()
                        prediction = response.json()

                        predicted_price = prediction.get('predicted_price')
                        future_prices = prediction.get('future_predictions')

                        if predicted_price is not None:
                            st.success(f"Predicted next day closing price for {ticker_input.upper()}: ${predicted_price:.2f}")

                        if future_prices:
                            st.subheader(f"📅 Future {days_to_predict} Days Forecast")
                            future_series = pd.Series(future_prices, index=range(1, days_to_predict + 1))
                            plot_interactive_line_chart(future_series, f"Future {days_to_predict} Days Stock Price Prediction", "Price (USD)")

                        st.subheader("📊 Historical Closing Prices")
                        plot_interactive_line_chart(data_df['Close'], "Historical Closing Prices", "Price (USD)")

                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
