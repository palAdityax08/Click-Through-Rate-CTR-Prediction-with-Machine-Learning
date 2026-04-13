# Click-Through Rate (CTR) Prediction

This project demonstrates a complete end-to-end Machine Learning solution for Click-Through Rate (CTR) Prediction, suitable for digital advertising platforms. 

## Project Architecture

1. **Data Generation**: (`generate_data.py`) Creates a realistic synthetic dataset simulating users, digital properties, and context to demonstrate CTR prediction without compromising proprietary platform data.
2. **Preprocessing & Model Selection**: (`train_model.py`) Handles preprocessing tasks (missing values, categorical encoding, scaling) and trains multiple models including:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - PyTorch Neural Network
   It performs hyperparameter tuning and evaluates them based on common classification metrics before saving the best performing model.
3. **Model Deployment & API**: (`app.py`, `Dockerfile`) A RESTful Flask API capable of taking in new user and contextual features in JSON format to predict CTR probability in real-time. Logs requests directly to a local SQLite database representing an enterprise database schema.
4. **Interactive Dashboard**: (`frontend.py`) A high-fidelity, premium Streamlit UI offering a 'Prediction Engine' leveraging the Flask API and a 'Business Analytics' tab utilizing Plotly to render historical insights.
5. **Experimentation**: (`experimentation.ipynb`) A Jupyter notebook providing Exploratory Data Analysis (EDA) and visualizations to understand feature relationships and model performance (ROC-AUC curves).

## Setup & Execution

### 1. Requirements

Ensure you have Python 3.10+ installed. Install project requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
Create the synthetic dataset `ctr_data.csv`:
```bash
python generate_data.py
```

### 3. Train Model
Run the training script to preprocess data, train models, and output `model_pipeline.pkl`:
```bash
python train_model.py
```

### 4. Create Jupyter Notebook
To generate the interactive notebook for experimentation:
```bash
python create_notebook.py
```
You can then open `experimentation.ipynb` in your IDE or Jupyter environment.

### 5. Run the API (Local)
Start the prediction server:
```bash
python app.py
```
Send a test POST request to `http://localhost:5000/predict`:
```json
{
  "user_age": 28,
  "user_gender": "Female",
  "user_income": "Medium",
  "device_type": "Mobile",
  "time_of_day": "Evening",
  "day_of_week": 4,
  "ad_category": "Fashion",
  "ad_placement": "In-feed"
}
```

### 6. Run the Interactive Dashboard (Streamlit)
To visualize the project professionally, start the Streamlit UI. **Ensure `app.py` is running simultaneously in another terminal.**
```bash
streamlit run frontend.py
```
This premium UI provides interactive prediction inputs and Business Analytics charts.

### 6. Docker (Containerization)
Build the container:
```bash
docker build -t ctr-api .
```
Run the container:
```bash
docker run -p 5000:5000 ctr-api
```
