# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ✅ Safe model loading
model = None
model_path = 'model/stroke_model.pkl'

if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print("✅ Model loaded successfully.")
else:
    print("❌ Model file is missing or empty!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Model not loaded. Please check your model file.")

    try:
        data = [
            int(request.form['gender']),
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            int(request.form['ever_married']),
            int(request.form['work_type']),
            int(request.form['Residence_type']),
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            int(request.form['smoking_status'])
        ]

        input_data = np.array([data])  # Make it 2D
        prediction = model.predict(input_data)

        result = "Stroke Risk" if prediction[0] == 1 else "No Stroke Risk"
        return render_template('index.html', prediction_text=f"Result: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True,port=9191)
