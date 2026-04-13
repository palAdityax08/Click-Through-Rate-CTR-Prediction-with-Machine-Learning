import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell for title
text_1 = """\
# CTR Prediction Experimentation and Analysis
Welcome to the experimentation notebook! Here we will visualize the dataset and model performance.
"""

# Code cell to load data
code_1 = """\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc
import numpy as np

# Load data
df = pd.read_csv('ctr_data.csv')
print(df.head())
"""

# Markdown cell for EDA
text_2 = """\
## Exploratory Data Analysis (EDA)
Let's look at CTR by device type and ad category.
"""

# Code cell for EDA
code_2 = """\
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='device_type', y='is_click', hue='ad_category')
plt.title('Click-Through Rate by Device and Ad Category')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='time_of_day', y='is_click')
plt.title('Click-Through Rate by Time of Day')
plt.show()
"""

# Markdown for model evaluation
text_3 = """\
## Model Evaluation
Loading the saved model and evaluating on the dataset.
"""

# Code for model evaluation
code_3 = """\
# Load the pre-trained pipeline
model = joblib.load('model_pipeline.pkl')

X = df.drop('is_click', axis=1)
y = df['is_click']

# Predict probabilities
probs = model.predict_proba(X)[:, 1]

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3)
]

with open('experimentation.ipynb', 'w') as f:
    nbf.write(nb, f)
    
print("Created experimentation.ipynb")
