# Stroke Prediction – Data Science Project.py

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# 2. Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# 3. Handle missing values
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# 4. Drop unnecessary column
df.drop("id", axis=1, inplace=True)

# 5. One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# 6. Feature-target split
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Handle imbalanced data using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# 9. Feature scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)

# 10. Train model (XGBoost)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_resampled, y_resampled)

# 11. Prediction and evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# 12. Save model and scaler
joblib.dump(model, "stroke_model.pkl")
joblib.dump(scaler, "scaler.pkl")