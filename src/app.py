# src/app.py

from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

# Paths
MODEL_LOGREG_PATH = os.path.join("models", "ticket_classifier_logreg.pkl")
MODEL_SVM_PATH = os.path.join("models", "ticket_classifier_svm.pkl")

# Load both models
logreg_model = joblib.load(MODEL_LOGREG_PATH)
svm_model = joblib.load(MODEL_SVM_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Auto Ticket Classifier API is running"})


@app.route("/classify", methods=["POST"])
def classify_ticket():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Request must contain 'text' field"}), 400

        text = data["text"]
        model_choice = data.get("model", None)  # optional field
        results = {}

        # --- Case 1: Only one model requested ---
        if model_choice:
            model_choice = model_choice.lower()

            if model_choice == "logreg":
                probs = logreg_model.predict_proba([text])[0]
                classes = logreg_model.classes_
                pred_index = probs.argmax()
                results["logreg"] = {
                    "category": classes[pred_index],
                    "confidence": round(probs[pred_index], 2)
                }

            elif model_choice == "svm":
                probs = svm_model.predict_proba([text])[0]
                classes = svm_model.classes_
                pred_index = probs.argmax()
                results["svm"] = {
                    "category": classes[pred_index],
                    "confidence": round(probs[pred_index], 2)
                }

            else:
                return jsonify({"error": "Invalid model. Use 'logreg' or 'svm'."}), 400

        # --- Case 2: No model specified → return both ---
        else:
            # Logistic Regression
            probs = logreg_model.predict_proba([text])[0]
            classes = logreg_model.classes_
            pred_index = probs.argmax()
            results["logreg"] = {
                "category": classes[pred_index],
                "confidence": round(probs[pred_index], 2)
            }

            # SVM
            probs = svm_model.predict_proba([text])[0]
            classes = svm_model.classes_
            pred_index = probs.argmax()
            results["svm"] = {
                "category": classes[pred_index],
                "confidence": round(probs[pred_index], 2)
            }

            # Pick best model
            if results["logreg"]["confidence"] >= results["svm"]["confidence"]:
                results["best_model"] = "logreg"
            else:
                results["best_model"] = "svm"

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


