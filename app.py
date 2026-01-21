from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & scaler from model folder
MODEL_PATH = os.path.join("model", "wine_cultivar_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["total_phenols"]),
            float(request.form["color_intensity"]),
            float(request.form["proline"])
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        result = model.predict(features_scaled)[0]
        prediction = f"Cultivar {result + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
