import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ========================
# 📂 Load Dataset
# ========================
dataset_path = r"C:\Users\Rishi\Downloads\dataset.csv.zip"
print("📂 Loading dataset...")
df = pd.read_csv(dataset_path, compression="zip")

# ========================
# 🛠 Feature Engineering
# ========================
df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

# Create models folder
os.makedirs("models", exist_ok=True)

# ========================
# Encode 'type'
# ========================
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# Save encoder (for app.py)
joblib.dump(le, "models/label_encoder.pkl")

# SMOTE with sampling_strategy=0.5
smote = SMOTE(random_state=42, sampling_strategy=0.5)

# ========================
# 📌 Helper Function: Train + Eval
# ========================
def train_model(X, y, model_name):
    print(f"\n🔄 Training {model_name} Model...")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Balance with SMOTE
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Drop constant columns
    X_train = X_train.loc[:, X_train.nunique() > 1]
    X_test = X_test[X_train.columns]

    # LightGBM fast params
    model = lgb.LGBMClassifier(
        device="cpu",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions (probabilities)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ========================
    # 🔍 Find Best Threshold (maximize F1)
    # ========================
    thresholds = np.linspace(0.1, 0.9, 50)
    best_thresh, best_f1 = 0.5, 0
    for t in thresholds:
        y_pred_temp = (y_proba > t).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"⭐ Best Threshold for {model_name}: {best_thresh:.2f} (F1={best_f1:.4f})")

    # Final predictions
    y_pred = (y_proba > best_thresh).astype(int)

    # Metrics
    print(f"\n✅ {model_name} Model Metrics:")
    print(classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Model (Threshold={best_thresh:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ========================
    # ✅ Save model + threshold dict
    # ========================
    joblib.dump({"model": model, "threshold": best_thresh},
                f"models/{model_name.lower()}.pkl")
    print(f"✅ {model_name} Model + threshold saved as models/{model_name.lower()}.pkl")


# ========================
# 1️⃣ Sender Model
# ========================
features_sender = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "type_encoded", "errorBalanceOrig"]
train_model(df[features_sender], df["isFraud"], "Fraud_Sender")

# ========================
# 2️⃣ Receiver Model
# ========================
features_receiver = ["step", "amount", "oldbalanceDest", "newbalanceDest", "type_encoded", "errorBalanceDest"]
train_model(df[features_receiver], df["isFraud"], "Fraud_Receiver")

print("\n🎉 All models trained, best thresholds applied, and saved successfully!")




