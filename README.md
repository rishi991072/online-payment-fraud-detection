<div align="center">

# 🛡️ FraudShield
### Online Payment Fraud Detection System



**FraudShield** is a real-time ML-powered fraud detection system for online financial transactions. It analyzes both the **sender** and **receiver** sides of a transaction independently using trained machine learning models and returns an instant fraud probability score — helping flag suspicious activity before damage is done.

[Features](#-features) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Project Structure](#-project-structure) · [How It Works](#-how-it-works) · [Tech Stack](#-tech-stack)

---

</div>

## ✨ Features

- 🔍 **Dual-sided Detection** — Independently analyses the Sender and Receiver of each transaction
- 🤖 **ML-Powered** — Separate trained models for sender and receiver fraud patterns
- ⚡ **Real-time Prediction** — Instant fraud probability score on every API call
- 📊 **Transaction History** — Stores last 4 predictions in memory for quick review
- 📥 **CSV Export** — Download full prediction history as a `.csv` file
- 🧪 **Debug & Test Endpoints** — Built-in routes to verify model health on startup
- 🔧 **Graceful Error Handling** — Meaningful error messages for missing fields, unknown types, or unloaded models
- 🧹 **History Management** — Clear prediction history via a dedicated API endpoint

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip
- Trained model files (`.pkl`) placed in the `models/` directory

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/FraudShield.git
cd FraudShield

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model files
#    models/fraud_sender.pkl
#    models/fraud_receiver.pkl
#    models/label_encoder.pkl

# 5. Run the Flask server
python app.py
```

The API will be live at `http://localhost:5000`

---

## 📁 Project Structure

```
FraudShield/
│
├── models/
│   ├── fraud_sender.pkl           ← trained fraud model for sender-side
│   ├── fraud_receiver.pkl         ← trained fraud model for receiver-side
│   └── label_encoder.pkl          ← label encoder for transaction types
│
├── templates/
│   └── index.html                 ← frontend UI (if applicable)
│
├── app.py                         ← Flask app · all routes & prediction logic
├── requirements.txt
└── README.md
```

---

## 🔍 How It Works

```
Incoming Transaction Request
           │
           ▼
┌─────────────────────────────┐
│  Role Check                  │  sender  /  receiver
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Label Encoder               │  Encodes transaction type
│  (TRANSFER, CASH_OUT, etc.)  │  into numeric feature
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────────┐
│  Sender  │  │  Receiver    │
│  Model   │  │  Model       │
│          │  │              │
│ Features:│  │ Features:    │
│ step     │  │ step         │
│ amount   │  │ amount       │
│ oldBal   │  │ oldBalDest   │
│ newBal   │  │ newBalDest   │
│ type_enc │  │ type_enc     │
│ errorBal │  │ errorBal     │
└────┬─────┘  └──────┬───────┘
     │               │
     ▼               ▼
┌─────────────────────────────┐
│  predict_proba()             │
│  → Fraud Probability Score   │
│  → Compare vs threshold      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  JSON Response               │
│  • prediction label          │
│  • probability %             │
│  • transaction details       │
│  • updated history (max 4)   │
└─────────────────────────────┘
```

### Feature Engineering

Both models use a derived **balance error** feature to detect discrepancies:

| Role | Error Feature Formula |
|---|---|
| Sender | `errorBalanceOrig = oldbalanceOrg − amount − newbalanceOrig` |
| Receiver | `errorBalanceDest = oldbalanceDest + amount − newbalanceDest` |

A non-zero balance error is a strong signal of a fraudulent or manipulated transaction.

---

## 📡 API Reference

### `GET /`
Returns the frontend UI (HTML page).

---

### `POST /predict`
Run fraud prediction on a transaction.

**Request Body (JSON):**

For **Sender** role:
```json
{
  "role": "sender",
  "type": "TRANSFER",
  "step": 1,
  "amount": 9000.00,
  "oldbalanceOrg": 10000.00,
  "newbalanceOrig": 1000.00
}
```

For **Receiver** role:
```json
{
  "role": "receiver",
  "type": "CASH_OUT",
  "step": 1,
  "amount": 9000.00,
  "oldbalanceDest": 500.00,
  "newbalanceDest": 9500.00
}
```

**Supported Transaction Types:**
`TRANSFER` · `CASH_OUT` · `PAYMENT` · `CASH_IN` · `DEBIT`

**Response (JSON):**
```json
{
  "result": {
    "prediction": "Fraudulent Transaction",
    "probability": 94.73,
    "details": { ...input fields... }
  },
  "history": [ ...last 4 results... ]
}
```

---

### `GET /debug`
Returns model load status — useful for verifying setup.

**Response:**
```json
{
  "sender_model_loaded": true,
  "receiver_model_loaded": true,
  "label_encoder_loaded": true,
  "sender_model_type": "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
  "available_classes": ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
}
```

---

### `GET /test_prediction`
Runs a quick prediction with built-in sample data to verify the models work end-to-end.

---

### `GET /download_csv`
Downloads prediction history as `transaction_history.csv`.

---

### `POST /clear_history`
Clears all stored prediction history.

**Response:**
```json
{
  "message": "History cleared",
  "history": []
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Web Framework** | [Flask](https://flask.palletsprojects.com) |
| **ML Models** | [scikit-learn](https://scikit-learn.org) |
| **Model Serialisation** | [joblib](https://joblib.readthedocs.io) |
| **Data Processing** | [Pandas](https://pandas.pydata.org) · [NumPy](https://numpy.org) |
| **Language** | Python 3.10+ |

---

## 📦 Dependencies

```
flask
numpy
pandas
scikit-learn
joblib
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Model Files

The following `.pkl` files must be present in the `models/` directory before running:

| File | Description |
|---|---|
| `fraud_sender.pkl` | Trained classifier for sender-side fraud detection |
| `fraud_receiver.pkl` | Trained classifier for receiver-side fraud detection |
| `label_encoder.pkl` | Encodes transaction type strings to numeric values |

Each model file can be either:
- A **direct model object** (e.g. `RandomForestClassifier`)
- A **dictionary** with keys `"model"` and `"threshold"` — the app handles both formats automatically

> If model files are missing or corrupt, the app will log errors on startup and return `500` on prediction requests. Use `GET /debug` to verify model status.

---

## 🗺️ Roadmap

- [ ] Real-time dashboard with transaction visualisations
- [ ] Database integration (PostgreSQL / MongoDB) for persistent history
- [ ] REST API authentication (API keys / JWT)
- [ ] Batch prediction endpoint (multiple transactions at once)
- [ ] Model retraining pipeline
- [ ] Docker containerisation
- [ ] Webhook alerts for flagged transactions

---
---



---

<div align="center">

Built with ❤️ using Python & Flask

**FraudShield** — *Detect fraud before it happens.*

</div>
