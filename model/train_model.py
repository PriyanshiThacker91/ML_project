import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("data/cardio_train.csv", sep=";")

# Feature engineering
df["age_years"] = df["age"] / 365.25
df["BMI"] = df["weight"] / ((df["height"]/100)**2)

# Drop unwanted
df.drop(["id","age"], axis=1, inplace=True)

# Split features and target
X = df.drop("cardio", axis=1)
y = df["cardio"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
lr = LogisticRegression()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Save best model
joblib.dump(rf, "model/trained_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model Saved Successfully")
