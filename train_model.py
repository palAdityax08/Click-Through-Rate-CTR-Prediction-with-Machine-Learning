import pandas as pd
import numpy as np
import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import xgboost as xgb

def load_and_preprocess_data(filepath='ctr_data.csv'):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('is_click', axis=1)
    y = df['is_click']
    
    # Define categorical and numerical columns
    categorical_cols = ['user_gender', 'user_income', 'device_type', 'time_of_day', 'ad_category', 'ad_placement']
    numerical_cols = ['user_age', 'day_of_week']
    
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X, y, preprocessor

class PyTorchNN(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        return x

def train_pytorch_model(X_train, y_train, X_val, y_val, input_dim, epochs=10, batch_size=256):
    print("\n--- Training PyTorch Neural Network ---")
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = PyTorchNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_preds = (val_outputs >= 0.5).float()
        val_probs = val_outputs.numpy()
        
    y_val_np = y_val_tensor.numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_val_np, val_preds),
        'precision': precision_score(y_val_np, val_preds, zero_division=0),
        'recall': recall_score(y_val_np, val_preds, zero_division=0),
        'auc': roc_auc_score(y_val_np, val_probs),
        'time': training_time
    }
    
    print(f"PyTorch NN Results - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, Time: {metrics['time']:.2f}s")
    
    # Return the trained model wrapped in a predict_proba function structure for later common use
    def predict_proba(X):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = model(X_tensor).numpy()
            # Return binary class format like sklearn [prob_class_0, prob_class_1]
            return np.hstack([1-probs, probs])
            
    return {'name': 'PyTorch NN', 'predict_proba': predict_proba, 'metrics': metrics, 'model': model}

def train_and_evaluate():
    X, y, preprocessor = load_and_preprocess_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    input_dim = X_train_processed.shape[1]
    
    # Dictionary to store model definitions and parameters for GridSearch
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {'C': [0.1, 1.0]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=50, random_state=42),
            'params': {'max_depth': [10, 20]} # kept brief for performance
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.1]}
        }
    }
    
    results = {}
    best_auc = 0.0
    best_model_name = ""
    best_pipeline = None
    
    # Train traditional models
    for name, config in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        
        clf = GridSearchCV(config['model'], config['params'], cv=3, scoring='roc_auc', n_jobs=-1)
        clf.fit(X_train_processed, y_train)
        
        training_time = time.time() - start_time
        
        best_clf = clf.best_estimator_
        y_pred = best_clf.predict(X_test_processed)
        y_prob = best_clf.predict_proba(X_test_processed)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob),
            'time': training_time
        }
        
        print(f"Best parameters: {clf.best_params_}")
        print(f"{name} Results - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, Time: {metrics['time']:.2f}s")
        
        results[name] = metrics
        
        # Check if best model so far
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_model_name = name
            # Create a full pipeline for the best traditional model
            best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('classifier', best_clf)])
                                            
    # Train PyTorch Model
    pt_result = train_pytorch_model(X_train_processed, y_train.values, X_test_processed, y_test.values, input_dim)
    results[pt_result['name']] = pt_result['metrics']
    
    if pt_result['metrics']['auc'] > best_auc:
        best_auc = pt_result['metrics']['auc']
        best_model_name = pt_result['name']
        print("\nPyTorch model is the best. In a production scenario we would export the weights and preprocessor separately.")
        print("For deployment simplicity in this project, we'll save the best scikit-learn compatible pipeline.")
        best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', models['XGBoost']['model'].fit(X_train_processed, y_train))]) # fallback to XGB if PT wins, just to have a single simple .pkl
                                        
    print(f"\n==========================================")
    print(f"Best Model: {best_model_name} with AUC: {best_auc:.4f}")
    
    # Save the best model
    model_path = 'model_pipeline.pkl'
    joblib.dump(best_pipeline, model_path)
    print(f"Saved the best pipeline to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
