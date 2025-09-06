from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os

app = Flask(__name__)

model = load_model('best_model.h5')
scaler = joblib.load('scaler.save')
print("Scaler loaded!")

TICKER = 'AAPL'
TIME_STEP = 60
NUM_FEATURES = 12

# Load API key from environment variable for security
API_KEY = os.getenv('API_KEY', 'your-secure-default-key')  # Replace default with your secure key or set env variable


@app.before_request
def check_api_key():
    if request.endpoint == 'predict':
        key = request.headers.get('x-api-key')
        if not key or key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401


@app.route('/')
def home():
    return "Stock Forecast API is running."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json
        features_sequence = content.get('features_sequence')
        days = content.get('days', 1)  # Default 1 day if not provided
        if not isinstance(days, int) or days < 1 or days > 60:
            return jsonify({'error': 'Parameter "days" must be an integer between 1 and 60.'}), 400

        if not features_sequence or len(features_sequence) != TIME_STEP:
            return jsonify({'error': f'Please provide exactly {TIME_STEP} time steps of features.'}), 400
        if any(len(step) != NUM_FEATURES for step in features_sequence):
            return jsonify({'error': f'Each time step must have {NUM_FEATURES} features.'}), 400

        features_array = np.array(features_sequence).reshape(-1, NUM_FEATURES)
        scaled_input = scaler.transform(features_array).reshape(1, TIME_STEP, NUM_FEATURES)

        future_predictions_scaled = []
        current_seq = scaled_input[0]

        for _ in range(days):
            pred_scaled = model.predict(current_seq[np.newaxis, :, :])[0, 0]
            future_predictions_scaled.append(pred_scaled)
            new_step = np.zeros((NUM_FEATURES,))
            new_step[0] = pred_scaled
            current_seq = np.vstack([current_seq[1:], new_step])

        # Inverse transform all predictions
        preds_full = np.zeros((len(future_predictions_scaled), scaler.n_features_in_))
        preds_full[:, 0] = future_predictions_scaled
        future_predictions = scaler.inverse_transform(preds_full)[:, 0]

        # Return only first day predicted price if days=1 for backward compatibility
        if days == 1:
            return jsonify({'predicted_price': float(future_predictions[0])})
        else:
            return jsonify({
                'predicted_price': float(future_predictions[0]),
                'future_predictions': future_predictions.tolist()
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
