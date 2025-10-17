from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('svm_bitcoin_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    open_close = float(request.form['open_close'])
    low_high = float(request.form['low_high'])
    is_quarter_end = int(request.form['is_quarter_end'])

    # Scale and predict
    features = scaler.transform([[open_close, low_high, is_quarter_end]])
    prediction = model.predict(features)[0]

    result = "Increase" if prediction == 1 else "Decrease"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
