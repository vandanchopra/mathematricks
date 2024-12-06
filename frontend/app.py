from flask import Flask, render_template, jsonify
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)

# Global variables to store the data
backtest_data = pd.DataFrame()
current_performance_data = pd.DataFrame()

# Function to load backtest data
def load_backtest_data():
    global backtest_data
    # Load your backtest data here
    # Example:
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    performance = np.random.randn(100).cumsum()
    backtest_data = pd.DataFrame({'date': dates, 'performance': performance})

# Function to update current performance data
def update_current_performance_data():
    global current_performance_data
    while True:
        # Update your current performance data here
        # Example:
        now = datetime.now()
        performance = np.random.randn(1)[0]
        new_data = pd.DataFrame({'date': [now], 'performance': [performance]})
        current_performance_data = pd.concat([current_performance_data, new_data])
        time.sleep(900)  # Update every 15 minutes

# Start the background task to update current performance data
threading.Thread(target=update_current_performance_data, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    global backtest_data, current_performance_data
    combined_data = pd.concat([backtest_data, current_performance_data])
    return combined_data.to_json(orient='records')

if __name__ == '__main__':
    load_backtest_data()
    app.run(debug=True)