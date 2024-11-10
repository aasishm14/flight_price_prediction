from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models
models = {
    'Linear Regression': joblib.load('linear_regression_model.joblib'),
    'Ridge Regression': joblib.load('ridge_regression_model.joblib'),
    'Lasso Regression': joblib.load('lasso_regression_model.joblib')
}

@app.route('/')
def home():
    return "Welcome to the ML Model Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame([data])
    results = {}
    for model_name, model in models.items():
        prediction = model.predict(data_df)
        results[model_name] = prediction[0]
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
