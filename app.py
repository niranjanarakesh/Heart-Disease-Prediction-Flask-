from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = np.array([[
        float(request.form['age']),
        float(request.form['gender']),
        float(request.form['cholesterol']),
        float(request.form['blood_pressure']),
        float(request.form['heart_rate']),
        float(request.form['smoking']),
        float(request.form['alcohol']),
        float(request.form['exercise_hours']),
        float(request.form['family_history']),
        float(request.form['diabetes']),
        float(request.form['obesity']),
        float(request.form['stress_level']),
        float(request.form['blood_sugar']),
        float(request.form['angina']),
        float(request.form['chest_pain'])
    ]])

    # Scale the inputs
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "⚠️ High Risk of Heart Disease"
    else:
        result = "✅ Low Risk of Heart Disease"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)