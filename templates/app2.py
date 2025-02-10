from flask import Flask, request, render_template, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
import docx

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded files

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to extract text from PDF and DOCX files
def analyze_report(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    return text

# Function to parse Boolean queries
def filter_data(data, query):
    # Basic implementation to filter based on Boolean keywords in query
    # Splitting by keywords to implement basic Boolean logic
    query = query.lower()  # Convert query to lowercase
    keywords = query.split(" ")
    
    filtered_data = data.copy()  # Start with full dataset
    
    # Apply filters based on Boolean logic
    for i, keyword in enumerate(keywords):
        if keyword == "and" and i + 1 < len(keywords):
            next_term = keywords[i + 1]
            filtered_data = filtered_data[filtered_data.apply(lambda row: row.astype(str).str.contains(next_term, case=False).any(), axis=1)]
        elif keyword == "or" and i + 1 < len(keywords):
            next_term = keywords[i + 1]
            filtered_data = pd.concat([filtered_data, data[data.apply(lambda row: row.astype(str).str.contains(next_term, case=False).any(), axis=1)]])
        elif keyword == "not" and i + 1 < len(keywords):
            next_term = keywords[i + 1]
            filtered_data = filtered_data[~filtered_data.apply(lambda row: row.astype(str).str.contains(next_term, case=False).any(), axis=1)]
        elif keyword not in ["and", "or", "not"]:
            filtered_data = filtered_data[filtered_data.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)]
            
    return filtered_data.drop_duplicates()

@app.route('/')
def upload_file():
    return render_template('ra.html')  # HTML file upload form

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract text from the uploaded file (PDF or DOCX)
    extracted_text = analyze_report(file_path)
    
    return jsonify({"message": "File uploaded and text extracted successfully", "extracted_text": extracted_text})

@app.route('/query', methods=['POST'])
def query():
    query = request.json.get('query')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.json.get('filename'))

    # Load the DataFrame from the uploaded file
    data = pd.read_csv(file_path)

    # Filter data based on the Boolean query
    filtered_data = filter_data(data, query)

    # Convert filtered data to JSON for the response
    return jsonify({"results": filtered_data.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
