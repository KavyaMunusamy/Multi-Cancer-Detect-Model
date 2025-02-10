# cervical_cancer_app.py
from flask import Flask, render_template, request
import pdfplumber
import joblib
import pandas as pd
import re
import numpy as np

app = Flask(__name__)

# Load the trained cervical cancer model
cervical_model = joblib.load('E:/healthcare-chatbot/cervical_cancer_model.pkl')

# Feature names used in the cervical cancer model
cervical_feature_names = [  'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 
    'Smokes', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 
    'Hormonal Contraceptives (years)', 'STDs', 'STDs (number)', 
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 
    'STDs:pelvic inflammatory disease', 'STDs:molluscum contagiosum', 'STDs:AIDS', 
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 'Dx:HPV', 'Dx', 
    'Hinselmann', 'Schiller', 'Citology'
 ]

# Route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('cc.html')

# Route to handle prediction
@app.route('/predict_cervical', methods=['POST'])
def predict_cervical():
    try:
        # Get the uploaded file
        file = request.files['report_pdf']

        # Extract text from the PDF
        text = extract_data_from_pdf(file)

        # Parse the text to extract feature values for cervical cancer
        features = parse_pdf_data_cervical(text)

        # Ensure that features match the model input
        if len(features) != len(cervical_feature_names):
            raise ValueError(f"Extracted features don't match expected number of features ({len(cervical_feature_names)})")

        # Convert to a DataFrame for prediction
        data = pd.DataFrame([features], columns=cervical_feature_names)

        # Make prediction and get prediction probability
        prediction = cervical_model.predict(data)
        prediction_proba = cervical_model.predict_proba(data)[0][1]  # Probability of being Positive (class 1)

        # Diagnosis based on prediction
        diagnosis = 'Positive' if prediction[0] == 1 else 'Negative'

        # Provide explanation based on feature values
        explanation = generate_explanation_cervical(features, prediction, prediction_proba)

    except Exception as e:
        diagnosis = "Error: " + str(e)
        explanation = ""

    return render_template('cc.html', prediction=diagnosis, explanation=explanation)

# Function to extract text from the uploaded PDF
def extract_data_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to parse extracted text and map to feature values for cervical cancer
def parse_pdf_data_cervical(text):
    # Define patterns for the cervical cancer features
    patterns = [
        r"age\s*[:\-]?\s*(\d+)",
        r"Number of sexual partners\s*[:\-]?\s*(\d+)",
        r"First sexual intercourse\s*[:\-]?\s*(\d+)",
        r"Num of pregnancies\s*[:\-]?\s*(\d+)",
        r"Smokes\s*[:\-]?\s*(\d+)",  # Smokes: 0 for No, 1 for Yes
        r"Smokes(years)\s*[:\-]?\s*(\d+)",  # 0 or 1
        r"Smokes (packs/year)\s*[:\-]?\s*(\d+)",  # 0 or 1
        r"Hormonal Contraceptives\s*[:\-]?\s*(\d+)", # 0 or 1
        r"Hormonal Contraceptives (years)\s*[:\-]?\s*(\d+)",
        r"STDs\s*[:\-]?\s*(\d+)",
        r"STDs (number)*[:\-]?\s*(\d+)",
        r"STDs:cervical condylomatosis\s*[:\-]?\s*(\d+)",
        r"STDs:vaginal condylomatosis\s*[:\-]?\s*(\d+)",  # Smokes: 0 for No, 1 for Yes
        r"STDs:vulvo-perineal condylomatosis\s*[:\-]?\s*(\d+)",  # 0 or 1
        r"STDs:syphilis\s*[:\-]?\s*(\d+)",  # 0 or 1
        r"STDs:pelvic inflammatory disease\s*[:\-]?\s*(\d+)",  # 0 or 1
        r"STDs:molluscum contagiosum\s*[:\-]?\s*(\d+)",
        r"STDs:AIDS\s*[:\-]?\s*(\d+)",
        r"STDs:Hepatitis B\s*[:\-]?\s*(\d+)",
        r"STDs:HPV\s*[:\-]?\s*(\d+)",
        r"STDs: Number of diagnosis\s*[:\-]?\s*(\d+)",  # Smokes: 0 for No, 1 for Yes
        r"Dx:HPV\s*[:\-]?\s*(\d+)",  # 0 or 1
        r"Dx\s*[:\-]?\s*(\d+)",  # 0 or 1
        r"Hinselmann\s*[:\-]?\s*(\d+)", # 0 or 1
        r"Schiller\s*[:\-]?\s*(\d+)",
        r"Citology\s*[:\-]?\s*(\d+)"
    ] 
    
    features = {}
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            feature_name = cervical_feature_names[i]
            features[feature_name] = float(match.group(1))

    return features

# Function to generate explanation for cervical cancer prediction
def generate_explanation_cervical(features, prediction, prediction_proba):
    explanation = f"The model predicted the result as {prediction[0]} (Positive/Negative) with a probability of {prediction_proba * 100:.2f}%."

    # Add feature impact explanation
    explanation += "\nKey features contributing to the prediction:"
    for feature, value in features.items():
        explanation += f"\n- {feature}: {value}"

    # Additional logic can be added to explain specific features further if needed
    return explanation

if __name__ == '__main__':
    app.run(port=3000, debug=True)
