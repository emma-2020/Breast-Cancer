# ============================================================
#  Breast Cancer Diagnosis - Model Training & Saving
#  Dataset: Breast Cancer Wisconsin
#  Target: diagnosis (M = Malignant, B = Benign)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

# Drop 'id' (not a feature) and 'Unnamed: 32' (empty column)
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Encode target: M → 1, B → 0
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])  # M=1, B=0

# Split features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\nFeature matrix shape:", X.shape)
print("Target distribution:\n", y.value_counts().rename({1: "Malignant", 0: "Benign"}))

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing  samples: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. FEATURE SCALING
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)         # apply same scale to test

# ─────────────────────────────────────────────
# 5. TRAIN MODEL  (Random Forest)
# ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("\nModel training complete!")

# ─────────────────────────────────────────────
# 6. EVALUATE MODEL
# ─────────────────────────────────────────────
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

# ── Confusion Matrix Plot ──
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved → confusion_matrix.png")

# ── ROC Curve Plot ──
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="darkorange")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("ROC curve saved → roc_curve.png")

# ─────────────────────────────────────────────
# 7. SAVE MODEL & SCALER
# ─────────────────────────────────────────────
joblib.dump(model,  r"C:\Users\piusa\Desktop\AI\breast_cancer_model.pkl")
joblib.dump(scaler, r"C:\Users\piusa\Desktop\AI\scaler.pkl")
print("\nModel  saved → breast_cancer_model.pkl")
print("Scaler saved → scaler.pkl")

# ─────────────────────────────────────────────
# 8. HOW TO LOAD & USE LATER
# ─────────────────────────────────────────────
# loaded_model  = joblib.load("breast_cancer_model.pkl")
# loaded_scaler = joblib.load("scaler.pkl")
#
# new_data_scaled = loaded_scaler.transform(new_data)
# prediction      = loaded_model.predict(new_data_scaled)
# print("Prediction:", "Malignant" if prediction[0] == 1 else "Benign")