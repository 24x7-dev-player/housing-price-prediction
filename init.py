import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import joblib

app = Flask(__name__)
# model = joblib.load("/Users/tushararora/Documents/house price prediction/housing_price_prediction.pkl")
model = pickle.load(open('housing_price_prediction.pkl','rb'))
sample = model.predict_proba([[[44,480,139,312,116,7]]])[0][0]
print(sample)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    value = np.array(input_features)
    output = model.predict_proba([[value]])[0][0]
    print(output)
    return render_template('index.html', Prediction_text = f"The house price is {output}")
if __name__ == "__main__":
    app.run(debug=True)
