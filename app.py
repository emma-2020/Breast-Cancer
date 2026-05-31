# ============================================================
#  OncoSense Diagnostic - Breast Cancer Prediction
#  Redesigned Flask Web App
#  Run: python app.py
#  Then open: http://127.0.0.1:5000
# ============================================================

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model  = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate all features are present
        missing = [f for f in FEATURES if f not in data or data[f] == ""]
        if missing:
            return jsonify({
                "error": f"Missing {len(missing)} field(s): {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}"
            }), 400

        values = [float(data[f]) for f in FEATURES]
        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        prediction   = model.predict(arr_scaled)[0]
        probability  = model.predict_proba(arr_scaled)[0]

        result = {
            "prediction":     "Malignant" if prediction == 1 else "Benign",
            "confidence":     round(float(max(probability)) * 100, 2),
            "malignant_prob": round(float(probability[1]) * 100, 2),
            "benign_prob":    round(float(probability[0]) * 100, 2),
        }
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Optional: model info endpoint used by the UI header ──
@app.route("/model-info")
def model_info():
    return jsonify({
        "model":   "Random Forest Classifier",
        "dataset": "Breast Cancer Wisconsin",
        "features": len(FEATURES),
        "classes": ["Benign", "Malignant"],
    })

if __name__ == "__main__":
    app.run(debug=True)
