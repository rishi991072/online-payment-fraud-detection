import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"C:\Users\Rishi\Downloads\dataset.csv.zip"

try:
    df = pd.read_csv(file_path)
    print("✅ Dataset Loaded Successfully!")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
except FileNotFoundError:
    print("❌ File not found!")
    exit()

# ================================
# Step 2: Basic Info
# ================================
print("\n📌 First 5 rows:")
print(df.head())

print("\n📌 Dataset Info:")
print(df.info())

print("\n📌 Missing Values:")
print(df.isnull().sum())

print("\n📌 Descriptive Statistics:")
print(df.describe())

# ================================
# Step 3: Visualization
# ================================
# Transaction Type Count
plt.figure(figsize=(6, 4))
sns.countplot(x="type", data=df)
plt.title("Transaction Type Distribution")
plt.xticks(rotation=45)
plt.show()

# Amount Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.show()

# Fraud vs Non-Fraud Transactions (अगर dataset में 'isFraud' कॉलम है तो)
if "isFraud" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="isFraud", data=df)
    plt.title("Fraud vs Non-Fraud Transaction Count")
    plt.show()

print("\n✅ EDA Completed Successfully!")
