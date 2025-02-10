# breast_cancer_app.py
from flask import Flask, render_template, request
import pdfplumber
import joblib
import pandas as pd
import re
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('E:/healthcare-chatbot/breast_cancer_model.pkl')

# Feature names used in the model
feature_names = [ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                  'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 
                  'symmetry_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
                  'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
                  'symmetry_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
                  'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 
                  'symmetry_worst']

# Route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('bc.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['report_pdf']

        # Extract text from the PDF
        text = extract_data_from_pdf(file)

        # Parse the text to extract feature values
        features = parse_pdf_data(text)

        # Ensure that features match the model input
        if len(features) != len(feature_names):
            raise ValueError(f"Extracted features don't match expected number of features ({len(feature_names)})")

        # Convert to a DataFrame for prediction
        data = pd.DataFrame([features], columns=feature_names)

        # Make prediction and get prediction probability
        prediction = model.predict(data)
        prediction_proba = model.predict_proba(data)[0][1]  # Probability of being Malignant (class 1)

        # Diagnosis based on prediction
        diagnosis = 'Malignant' if prediction[0] == 1 else 'Benign'

        # Provide explanation based on feature values
        explanation = generate_explanation(features, prediction, prediction_proba)

    except Exception as e:
        diagnosis = "Error: " + str(e)
        explanation = ""

    return render_template('bc.html', prediction=diagnosis, explanation=explanation)

# Function to extract text from the uploaded PDF
def extract_data_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to parse extracted text and map to feature values
def parse_pdf_data(text):
    patterns = [
        r"radius_mean\s*[:\-]?\s*([\d\.]+)",
        r"texture_mean\s*[:\-]?\s*([\d\.]+)",
        r"perimeter_mean\s*[:\-]?\s*([\d\.]+)",
        r"area_mean\s*[:\-]?\s*([\d\.]+)",
        r"smoothness_mean\s*[:\-]?\s*([\d\.]+)",
        r"compactness_mean\s*[:\-]?\s*([\d\.]+)",
        r"concavity_mean\s*[:\-]?\s*([\d\.]+)",
        r"concave_points_mean\s*[:\-]?\s*([\d\.]+)",
        r"symmetry_mean\s*[:\-]?\s*([\d\.]+)",
        r"radius_se\s*[:\-]?\s*([\d\.]+)",
        r"texture_se\s*[:\-]?\s*([\d\.]+)",
        r"perimeter_se\s*[:\-]?\s*([\d\.]+)",
        r"area_se\s*[:\-]?\s*([\d\.]+)",
        r"smoothness_se\s*[:\-]?\s*([\d\.]+)",
        r"compactness_se\s*[:\-]?\s*([\d\.]+)",
        r"concavity_se\s*[:\-]?\s*([\d\.]+)",
        r"concave_points_se\s*[:\-]?\s*([\d\.]+)",
        r"symmetry_se\s*[:\-]?\s*([\d\.]+)",
        r"radius_worst\s*[:\-]?\s*([\d\.]+)",
        r"texture_worst\s*[:\-]?\s*([\d\.]+)",
        r"perimeter_worst\s*[:\-]?\s*([\d\.]+)",
        r"area_worst\s*[:\-]?\s*([\d\.]+)",
        r"smoothness_worst\s*[:\-]?\s*([\d\.]+)",
        r"compactness_worst\s*[:\-]?\s*([\d\.]+)",
        r"concavity_worst\s*[:\-]?\s*([\d\.]+)",
        r"concave_points_worst\s*[:\-]?\s*([\d\.]+)",
        r"symmetry_worst\s*[:\-]?\s*([\d\.]+)"
    ]
    
    features = {}
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            feature_name = feature_names[i]
            features[feature_name] = float(match.group(1))

    return features

# Function to generate explanation for the prediction
def generate_explanation(features, prediction, prediction_proba):
    explanation = f"The model predicted the result as {prediction[0]} (Malignant/Benign) with a probability of {prediction_proba * 100:.2f}%."

    # Add feature impact explanation
    explanation += "\nKey features contributing to the prediction:"
    for feature, value in features.items():
        explanation += f"\n- {feature}: {value}"

    # You could add more detailed feature-specific explanations if you have model-specific insights
    return explanation

if __name__ == '__main__':
    app.run(port=3000, debug=True)
