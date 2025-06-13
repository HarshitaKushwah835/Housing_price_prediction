from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
with open("house_price_prediction.pkl", "rb") as f:
    model=pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['crim']),
        float(request.form['zn']),
        float(request.form['indus']),
        float(request.form['chas']),
        float(request.form['nox']),
        float(request.form['rx']),
        float(request.form['age']),
        float(request.form['dis']),
        float(request.form['rad']),
        float(request.form['tax']),
        float(request.form['ptratio']),
        float(request.form['black']),
        float(request.form['lstat'])
    ]
    features_array = np.array([features]) 
    prediction  = model.predict(features_array)
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text=f"Predicted Price:{output}")

if __name__ ==  "__main__":
    app.run(debug=True)