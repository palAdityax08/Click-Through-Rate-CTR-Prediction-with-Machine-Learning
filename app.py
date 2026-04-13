from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

# Load the model
try:
    model_pipeline = joblib.load('model_pipeline.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# SQLite Database initialization
DB_PATH = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_age REAL,
            user_gender TEXT,
            user_income TEXT,
            device_type TEXT,
            time_of_day TEXT,
            day_of_week REAL,
            ad_category TEXT,
            ad_placement TEXT,
            predicted_probability REAL,
            prediction INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

def log_prediction(data, prob, pred):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Ensure default None for missing keys
        row = (
            datetime.now().isoformat(),
            data.get('user_age'),
            data.get('user_gender'),
            data.get('user_income'),
            data.get('device_type'),
            data.get('time_of_day'),
            data.get('day_of_week'),
            data.get('ad_category'),
            data.get('ad_placement'),
            prob,
            pred
        )
        
        cursor.execute('''
            INSERT INTO api_logs (
                timestamp, user_age, user_gender, user_income, device_type,
                time_of_day, day_of_week, ad_category, ad_placement,
                predicted_probability, prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', row)
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log prediction to DB: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline:
        return jsonify({"error": "Model failed to load on server startup."}), 500

    try:
        # Parse JSON data
        data = request.get_json(force=True)
        
        # Validate required fields
        required_fields = ['user_age', 'user_gender', 'user_income', 'device_type', 'time_of_day', 'day_of_week', 'ad_category', 'ad_placement']
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400
        
        # Convert to DataFrame (model pipeline expects a DataFrame)
        df = pd.DataFrame([data])
        
        # Make prediction
        probability = float(model_pipeline.predict_proba(df)[0][1])
        prediction = int(model_pipeline.predict(df)[0])
        
        # Log to SQLite
        log_prediction(data, probability, prediction)
        
        return jsonify({
            "predicted_ctr_probability": probability,
            "prediction_class": prediction,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Run the server on all interfaces and port 5000
    app.run(host='0.0.0.0', port=5001)
