from flask import Flask, request, render_template
import numpy as np
import pickle

# Load trained model
with open("sonar_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form
        input_data = request.form["features"]
        input_data_np = np.asarray(input_data.split(","), dtype=float).reshape(1, -1)

        prediction = model.predict(input_data_np)

        if prediction[0] == "R":
            result = "🪨 This object is a Rock"
        else:
            result = "💣 This object is a Mine"

        return render_template("index.html", prediction_text=result)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
