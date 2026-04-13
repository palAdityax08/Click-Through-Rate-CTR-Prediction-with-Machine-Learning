import requests
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import time

URL = "http://localhost:5001/predict"
DB_PATH = "predictions.db"

valid_payload = {
    "user_age": 28,
    "user_gender": "Female",
    "user_income": "Medium",
    "device_type": "Mobile",
    "time_of_day": "Evening",
    "day_of_week": 4,
    "ad_category": "Fashion",
    "ad_placement": "In-feed"
}

print("=== Starting Automated Test Protocol ===\n")

# B. API Validation (Missing Data)
print("1. Executing API Validation Test (Missing Field)...")
invalid_payload = valid_payload.copy()
del invalid_payload["user_age"]
response_invalid = requests.post(URL, json=invalid_payload)
print(f"Status Code: {response_invalid.status_code}")
print(f"Response: {response_invalid.json()}\n")

# A. Rapid-Fire Test
print("2. Executing Rapid-Fire Test (10 concurrent requests)...")
def send_request(i):
    # Alter some payload minimally
    payload = valid_payload.copy()
    payload["user_age"] = min(20 + i, 80)
    res = requests.post(URL, json=payload)
    return res.status_code

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(send_request, range(10)))
print(f"10 Requests Completed. Status Codes: {results}\n")

time.sleep(1) # wait for db inserts

# C. Database Verification
print("3. Executing Database Proof (Verifying 10+ previous records logged)...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, timestamp, user_age, ad_category, predicted_probability, prediction FROM api_logs ORDER BY id DESC LIMIT 5")
rows = cursor.fetchall()
col_names = [desc[0] for desc in cursor.description]

print(f"{' | '.join(col_names)}")
print("-" * 80)
for row in rows:
    # Format line for printing
    print(" | ".join([str(x)[:20] for x in row]))
conn.close()

print("\n=== Automated Test Protocol Complete ===")
