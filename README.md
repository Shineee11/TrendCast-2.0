# TrendCast

TrendCast is a sales prediction application that helps businesses forecast their sales for the next 5 days. This project utilizes machine learning models to analyze historical sales data and provide accurate sales forecasts.

## Features

- Predicts sales for the next 5 days based on historical data.
- Provides a visual representation of the sales forecast.
- Displays key metrics like MAE, RMSE, and MAPE.
- Easy-to-use web interface built with Flask.

## Prerequisites

- Python 3.7+
- Flask>=2.3.2
- numpy>=1.21.2,<2.0.0
- pandas
- tensorflow==2.17.0
- scikit-learn
- matplotlib
- joblib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Shineee11/TrendCast-2.0.git
    cd TrendCast-2.0
    ```

2. Create a virtual environment**:

    ```bash
    python -m venv TrendCast-2.0-venv
    ```

3. Activate the virtual environment:

    ```bash
    source TrendCast-2.0-venv/bin/activate  # On Windows, use `TrendCast-2.0-venv\Scripts\activate`
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Enter a product name in the search box to see the sales forecast for the next 5 days.

## Files

- `app.py`: Main application file that runs the Flask web server.
- `requirements.txt`: List of Python packages required for the project.
- `static/`: Directory containing static files (CSS, Plot/graph).
- `templates/`: Directory containing HTML templates.
- `data/cleaned_Amazon_Sales_Report.csv`: CSV file containing the historical sales data.
- `model/final_lstm_model.keras`: Trained LSTM model.
- `model/scaler.pkl`: Scaler used for normalizing the data.

## Model and Data

The model is trained on historical sales data from the Amazon Sales Report. The features used for prediction include:

- Quantity sold (`Qty`)
- Sales from the last 7 days
- Day of the week
- Month
- Previous day sales
- Previous week sales
- Previous 2 day sales
- Previous 3 day sales
- Rolling mean of the last 3 days

## Visualizations

The application generates a plot of the sales forecast and displays it on the web page along with key metrics such as MAE, RMSE, and MAPE.
