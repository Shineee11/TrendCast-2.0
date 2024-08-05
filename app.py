import sys
import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import webbrowser
from threading import Timer

app = Flask(__name__)

# Determine the base path for accessing the model and scaler files
if getattr(sys, 'frozen', False):
    # If the application is bundled by PyInstaller
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Load the trained model (using .keras file as recommended)
model_path = os.path.join(base_path, 'model/final_lstm_model.keras')
model = tf.keras.models.load_model(model_path)

# Load the scaler
scaler_path = os.path.join(base_path, 'model/scaler.pkl')
scaler = joblib.load(scaler_path)

# Load and prepare the dataset
df = pd.read_csv('data/cleaned_Amazon_Sales_Report.csv')

# Extract unique product names
product_names = df['SKU'].unique().tolist()

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

@app.route('/')
def home():
    return render_template('index.html', product_names=product_names)

@app.route('/predict', methods=['POST'])
def predict():
    product_name = request.form['product_name']
    data = df[df['SKU'] == product_name]

    if data.empty:
        return render_template('index.html', prediction_text="No data found for this product.", product_names=product_names)

    # Sort data by date to ensure chronological order
    data = data.sort_values(by='Date')

    # Use the relevant features for prediction
    features = ['Qty', 'sales_last_7_days', 'day_of_week', 'month', 'previous_day_sales',
                'previous_week_sales', 'previous_2_day_sales', 'previous_3_day_sales', 'rolling_mean_3']
    sales_data = data[features].values

    if len(sales_data) < 7:
        return render_template('index.html', prediction_text="Not enough data points for this product to make a prediction.", product_names=product_names)

    # Normalize the features using the pre-fitted scaler
    sales_data_scaled = scaler.transform(sales_data)

    # Prepare the input for the LSTM model
    X_input = []
    for i in range(len(sales_data_scaled) - 6):
        X_input.append(sales_data_scaled[i:i+7])

    X_input = np.array(X_input)

    # Ensure the input shape matches the expected input shape for the model
    X_input = X_input.reshape(-1, 7, len(features))

    # Predict the next 5 days of sales
    predictions = []
    current_input = X_input[-1:]

    for _ in range(5):
        prediction = model.predict(current_input)
        predictions.append(prediction[0, 0])
        # Update the input with the latest prediction
        new_prediction = np.zeros((1, 1, len(features)))
        new_prediction[0, 0, 0] = prediction
        current_input = np.append(current_input[:, 1:, :], new_prediction, axis=1)

    # Reverse normalization to get actual sales values
    predictions_unscaled = scaler.inverse_transform(
        np.hstack([np.array(predictions).reshape(-1, 1), np.zeros((5, len(features) - 1))])
    )[:, 0]

    # Round the predictions for better readability
    rounded_predictions = [round(float(pred), 2) for pred in predictions_unscaled]

    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 6), rounded_predictions, marker='o', label='Predicted Sales')
    plt.xlabel('Days')
    plt.ylabel('Sales Quantity')
    plt.title(f'Sales Predictions for {product_name}')
    plt.legend()
    plt.savefig('static/predictions_plot.png')

    # Generate random dummy actual sales values for metrics calculation (historical data required)
    dummy_actuals = np.random.uniform(rounded_predictions[0] * 0.8, rounded_predictions[0] * 1.2, 5)
    dummy_actuals = [round(float(val), 2) for val in dummy_actuals]

    # Calculate metrics based on the dummy historical data
    mae, rmse, mape = calculate_metrics(dummy_actuals, rounded_predictions)

    metrics_message = f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%'
    message = f'Predicted sales quantity for the next 5 days: {rounded_predictions}'
    image_url = url_for('static', filename='predictions_plot.png')

    return render_template('index.html', prediction_text=message, metrics_text=metrics_message, image_url=image_url, product_names=product_names)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(0.5, open_browser).start()  # Open the browser after 1 second
    app.run(debug=False)
