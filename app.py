

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd
import io
import os
import traceback

app = Flask(__name__)

# ==== Models Load ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Correct paths
sender_model_path = os.path.join(MODEL_DIR, "fraud_sender.pkl")
receiver_model_path = os.path.join(MODEL_DIR, "fraud_receiver.pkl")
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Load models with proper extraction
try:
    sender_data = joblib.load(sender_model_path)
    # Check if it's a dictionary or direct model
    if isinstance(sender_data, dict):
        sender_model = sender_data["model"]
        sender_thresh = sender_data.get("threshold", 0.5)
    else:
        sender_model = sender_data
        sender_thresh = 0.5
    print("✅ Sender model loaded successfully")
except Exception as e:
    print(f"❌ Error loading sender model: {e}")
    sender_model = None
    sender_thresh = 0.5

try:
    receiver_data = joblib.load(receiver_model_path)
    if isinstance(receiver_data, dict):
        receiver_model = receiver_data["model"]
        receiver_thresh = receiver_data.get("threshold", 0.5)
    else:
        receiver_model = receiver_data
        receiver_thresh = 0.5
    print("✅ Receiver model loaded successfully")
except Exception as e:
    print(f"❌ Error loading receiver model: {e}")
    receiver_model = None
    receiver_thresh = 0.5

try:
    label_encoder = joblib.load(label_encoder_path)
    print("✅ Label encoder loaded successfully")
    print(f"Available transaction types: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error loading label encoder: {e}")
    label_encoder = None

# Transaction history (max 4)
history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/debug", methods=["GET"])
def debug():
    """Debug endpoint to check model status"""
    debug_info = {
        "sender_model_loaded": sender_model is not None,
        "receiver_model_loaded": receiver_model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "sender_model_type": str(type(sender_model)) if sender_model else None,
        "receiver_model_type": str(type(receiver_model)) if receiver_model else None,
        "available_classes": list(label_encoder.classes_) if label_encoder else []
    }
    return jsonify(debug_info)

@app.route("/test_prediction", methods=["GET"])
def test_prediction():
    """Test prediction with sample data"""
    try:
        results = {}
        
        # Test sender prediction
        if sender_model is not None:
            sample_sender_data = {
                "step": 1, 
                "amount": 1000.0, 
                "oldbalanceOrg": 5000.0, 
                "newbalanceOrig": 4000.0,
                "type": "TRANSFER"
            }
            
            tx_type = sample_sender_data["type"]
            if tx_type in label_encoder.classes_:
                type_encoded = label_encoder.transform([tx_type])[0]
                step = sample_sender_data["step"]
                amount = sample_sender_data["amount"]
                oldbalanceOrg = sample_sender_data["oldbalanceOrg"]
                newbalanceOrig = sample_sender_data["newbalanceOrig"]
                errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig

                features = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, type_encoded, errorBalanceOrig]])
                proba = sender_model.predict_proba(features)[:, 1][0]
                pred = int(proba > sender_thresh)
                
                results["sender_test"] = {
                    "success": True,
                    "prediction": "Fraudulent" if pred == 1 else "Legitimate",
                    "probability": round(proba * 100, 2),
                    "features": features.tolist()
                }
            else:
                results["sender_test"] = {
                    "success": False,
                    "error": f"Transaction type '{tx_type}' not in label encoder classes"
                }
        
        # Test receiver prediction
        if receiver_model is not None:
            sample_receiver_data = {
                "step": 1, 
                "amount": 1000.0, 
                "oldbalanceDest": 2000.0, 
                "newbalanceDest": 3000.0,
                "type": "TRANSFER"
            }
            
            tx_type = sample_receiver_data["type"]
            if tx_type in label_encoder.classes_:
                type_encoded = label_encoder.transform([tx_type])[0]
                step = sample_receiver_data["step"]
                amount = sample_receiver_data["amount"]
                oldbalanceDest = sample_receiver_data["oldbalanceDest"]
                newbalanceDest = sample_receiver_data["newbalanceDest"]
                errorBalanceDest = oldbalanceDest + amount - newbalanceDest

                features = np.array([[step, amount, oldbalanceDest, newbalanceDest, type_encoded, errorBalanceDest]])
                proba = receiver_model.predict_proba(features)[:, 1][0]
                pred = int(proba > receiver_thresh)
                
                results["receiver_test"] = {
                    "success": True,
                    "prediction": "Fraudulent" if pred == 1 else "Legitimate",
                    "probability": round(proba * 100, 2),
                    "features": features.tolist()
                }
            else:
                results["receiver_test"] = {
                    "success": False,
                    "error": f"Transaction type '{tx_type}' not in label encoder classes"
                }
                
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Test prediction failed: {str(e)}", "traceback": traceback.format_exc()}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if models are loaded
        if sender_model is None or receiver_model is None or label_encoder is None:
            return jsonify({"error": "Models not loaded properly. Check server logs."}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        role = data.get("role", "").lower()
        
        if role not in ["sender", "receiver"]:
            return jsonify({"error": "Invalid role. Must be 'sender' or 'receiver'"}), 400
            
        # Check if all required fields are present
        required_fields = ["type", "step", "amount"]
        if role == "sender":
            required_fields.extend(["oldbalanceOrg", "newbalanceOrig"])
        else:
            required_fields.extend(["oldbalanceDest", "newbalanceDest"])
            
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400
            
        tx_type = data["type"]
        
        # Check if transaction type is valid
        if tx_type not in label_encoder.classes_:
            return jsonify({"error": f"Unknown transaction type: {tx_type}. Valid types: {list(label_encoder.classes_)}"}), 400
            
        type_encoded = label_encoder.transform([tx_type])[0]

        if role == "sender":
            step = int(data["step"])
            amount = float(data["amount"])
            oldbalanceOrg = float(data["oldbalanceOrg"])
            newbalanceOrig = float(data["newbalanceOrig"])
            errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig

            features = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, type_encoded, errorBalanceOrig]])
            proba = sender_model.predict_proba(features)[:, 1][0]
            pred = int(proba > sender_thresh)

        else:  # receiver
            step = int(data["step"])
            amount = float(data["amount"])
            oldbalanceDest = float(data["oldbalanceDest"])
            newbalanceDest = float(data["newbalanceDest"])
            errorBalanceDest = oldbalanceDest + amount - newbalanceDest

            features = np.array([[step, amount, oldbalanceDest, newbalanceDest, type_encoded, errorBalanceDest]])
            proba = receiver_model.predict_proba(features)[:, 1][0]
            pred = int(proba > receiver_thresh)

        result = {
            "prediction": "Fraudulent Transaction" if pred == 1 else "Legitimate Transaction",
            "probability": round(proba * 100, 2),
            "details": data
        }

        # Save history (max 4)
        history.insert(0, result)
        if len(history) > 4:
            history.pop()

        return jsonify({"result": result, "history": history})
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route("/download_csv", methods=["GET"])
def download_csv():
    if not history:
        return jsonify({"error": "No history available"}), 400

    try:
        df = pd.DataFrame(history)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="transaction_history.csv"
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate CSV: {str(e)}"}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear prediction history"""
    global history
    history = []
    return jsonify({"message": "History cleared", "history": history})

if __name__ == "__main__":
    # Test the models on startup
    print("🧪 Testing models on startup...")
    try:
        with app.test_client() as client:
            debug_response = client.get('/debug')
            print("Debug endpoint:", debug_response.get_json())
            
            test_response = client.get('/test_prediction')
            print("Test prediction:", test_response.get_json())
    except Exception as e:
        print(f"Startup test failed: {e}")
    
    print("🚀 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)