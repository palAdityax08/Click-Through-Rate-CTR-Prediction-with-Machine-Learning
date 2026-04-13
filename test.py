import requests
import sqlite3
import json

url = "http://localhost:5000/predict"
payload = {
    "user_age": 28,
    "user_gender": "Female",
    "user_income": "Medium",
    "device_type": "Mobile",
    "time_of_day": "Evening",
    "day_of_week": 4,
    "ad_category": "Fashion",
    "ad_placement": "In-feed"
}

print("1. Sending POST request to the API...")
try:
    response = requests.post(url, json=payload)
    print("API Status Code:", response.status_code)
    print("API Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error connecting to API: {e}")

print("\n2. Checking the SQLite Database representation of MySQL/MongoDB...")
try:
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM api_logs ORDER BY id DESC LIMIT 1;")
    row = cursor.fetchone()
    
    # Get column names
    col_names = [description[0] for description in cursor.description]
    
    if row:
        print("Successfully retrieved the latest logged request from the DB:")
        for col, val in zip(col_names, row):
            print(f"  - {col}: {val}")
    else:
        print("Database is empty or could not be read.")
    conn.close()
except Exception as e:
    print(f"Error reading database: {e}")
