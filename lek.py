from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import os
import joblib  # Import joblib for loading the model
from app import extract_data_from_pdf, parse_pdf_data, generate_explanation  # Import your functions from app.py

app = Flask(__name__)
bcrypt = Bcrypt(app)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/loginpage"
app.config["UPLOAD_FOLDER"] = "uploads"  # Folder for PDF uploads
mongo = PyMongo(app)

# Load the ML model
model = joblib.load('E:/healthcare-chatbot/breast_cancer_model.pkl')  # Load your actual model here

# MongoDB Collection
users_collection = mongo.db.LogInCollection

# Root Route
@app.route('/')
def index():
    return redirect(url_for('login'))  # Redirect to login page

# Sign Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        name = data.get('name')
        password = data.get('password')
        email = data.get('email')

        # Check if the user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return jsonify({"message": "User with this email already exists"}), 400

        # Hash password before storing it
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = {
            "name": name,
            "password": hashed_password,
            "email": email
        }

        # Insert the new user into MongoDB
        users_collection.insert_one(new_user)
        return jsonify({"message": "User registered successfully"}), 201

    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        name = data.get('name')
        password = data.get('password')

        # Find user by name
        user = users_collection.find_one({"name": name})

        # Check if user exists and password matches
        if user and bcrypt.check_password_hash(user['password'], password):
            return redirect(url_for('home'))  # Redirect to home if login is successful
        else:
            return jsonify({"message": "Incorrect username or password"}), 400

    return render_template('login.html')

# Home Route
@app.route('/home')
def home():
    return render_template('home.html')

# Breast Cancer Prediction Route
@app.route('/bc', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        file = request.files['report_pdf']
        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the PDF with your ML model
            pdf_data = extract_data_from_pdf(filepath)
            parsed_data = parse_pdf_data(pdf_data)
            prediction = model.predict([parsed_data])  # Call your ML model here
            explanation = generate_explanation(prediction)

            # Remove the file after prediction if needed
            os.remove(filepath)
            return render_template('bc.html', prediction=prediction, explanation=explanation)
    return render_template('bc.html')

# Cervical Cancer Detection Route
@app.route('/cc')
def cervical_cancer():
    return render_template('cc.html')

# Symptom Bot Route
@app.route('/symptom-bot')
def symptom_bot():
    return render_template('index.html')

# Start the server
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=3000)
