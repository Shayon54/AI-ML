import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset_path = "data.csv"  # Ensure this file is in the correct directory
df = pd.read_csv(dataset_path)

# Extract labels and features
labels = df.iloc[:, 0].values  # First column contains labels
features = df.iloc[:, 1:].values  # Remaining columns contain pixel values

# Normalize pixel values (scale between 0 and 1)
features = features / 255.0  

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Logistic Regression Model
model = LogisticRegression(max_iter=500)  # Increased iterations for better convergence
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Explainable AI: SHAP for Logistic Regression
explainer = shap.LinearExplainer(model, X_train)  # Use LinearExplainer for logistic regression
shap_values = explainer(X_test)  # Compute SHAP values

# SHAP Summary Plot (Feature Importance)
shap.summary_plot(shap_values, X_test, feature_names=[f"Pixel {i}" for i in range(X_test.shape[1])])
