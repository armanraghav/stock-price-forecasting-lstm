![](https://i.pinimg.com/originals/cb/07/60/cb07601b2d5a4335c20880ea71b82edd.gif)
# Stock Price Forecasting with LSTM and Streamlit

This project is a stock price forecasting web application using a deep learning LSTM model. It includes a Flask API backend that serves predictions and a Streamlit frontend for user interaction and visualization.

---

## Features

- **LSTM Time Series Forecasting:** Predicts future stock prices based on historical data and technical indicators.
- **Flask REST API:** Provides a secure endpoint to serve predictions using a token-based API key.
- **Streamlit Frontend:** User-friendly interactive UI to input stock ticker, forecast horizon, and display results with interactive charts.
- **Dockerized Deployment:** Both frontend and backend containerized for cloud deployments.
- **Authentication:** Simple API key authentication for securing the prediction API.

---

## Architecture

- **Backend:** Flask API loads trained LSTM model and scaler, processes input features, returns predicted prices.
- **Frontend:** Streamlit app fetches stock data from Yahoo Finance, computes features, calls backend API with authentication.
- **Communication:** Frontend sends JSON payload including `features_sequence` and forecast `days` with API key in headers.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional for containerized deployment)
- Google Cloud or Heroku account (optional for cloud deployment)

### Installation

1. Clone the repo:

```
git clone https://github.com/yourusername/stock-price-forecast.git
```
```
cd stock-price-forecast
```

2. Create and activate virtual environment:
```
python -m venv venv
```
```
source venv/bin/activate # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Set up environment variables:
```
export API_KEY="your-secure-api-key"
```
5. Run Flask API:
```
python app.py
```
6. In another terminal, run Streamlit frontend:
```
streamlit run frontend.py
```
---

## Usage

- Enter a valid stock ticker (e.g. `AAPL`).
- Enter number of days to forecast (1 to 60).
- Provide the API URL (default: `http://localhost:5000/predict`).
- Enter your API key.
- Click **Predict Next Days Price** to see predicted and historical prices displayed with interactive charts.

---

## Deployment

- Both backend and frontend are Dockerized.
- You can deploy to Google Cloud Run, Heroku, or any container-friendly cloud service.
- Remember to set the `API_KEY` environment variable in your deployment environment.
- Refer to the project wiki or documentation for detailed deployment steps.

---

## API Details

### Endpoint

`POST /predict`

### Headers

- `x-api-key`: your API key

### Request JSON body
```
{
"features_sequence": [ [num, num, ..., num], ..., [num, num, ..., num] ],
"days": 30
}
```
- `features_sequence`: array of 60 time steps, each an array of 12 numerical features.
- `days`: integer (1 to 60) specifying number of days to predict.

### Response JSON
```
{
"predicted_price": float,
"future_predictions": [float]
}
```
---

## Security

- API is secured with an API key passed in the request header.
- Keep your API key confidential.
- For production deployments, consider using HTTPS and enhanced authentication mechanisms.

---

## License

MIT License

---

## Acknowledgments

- TensorFlow and Keras for LSTM modeling.
- Streamlit for rapid web app development.
- Yahoo Finance (yfinance) for stock data.
- Inspiration from open-source forecasting projects.

---

