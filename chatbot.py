import random
from json import loads

import torch
from flask import Flask, jsonify, render_template, request

from chatmodel import NeuralNetwork
from nltk_stuffs import bow, tokenize_sentence

# Initialize Flask app
app = Flask(__name__)

# Set device to GPU if available
gpu_support = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json') as f:
    intents = loads(f.read())

# Load the trained model
FILE_NAME = "chatbot_data.pth"
model_data = torch.load(FILE_NAME)

model = NeuralNetwork(model_data["input_size"], model_data["hidden_size"], model_data["output_size"])
model = model.to(gpu_support)
model.load_state_dict(model_data['model_state'])
model.eval()  # Set model to evaluation mode

bot_name = "Health Care Chatbot"

# Serve the index.html file at the root
@app.route('/')
def home():
    return render_template('index.html')

# Define the chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    # Get the query from the request
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"response": "Please provide a query"}), 400

    # Process the user input
    user_query = tokenize_sentence(user_query)
    x = bow(user_query, model_data["all_words"])
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(gpu_support)
    
    # Get model output
    output = model(x)
    predicted = (torch.max(output, dim=1))[1]
    predicted_tag = model_data["tags"][predicted.item()]
    
    probabilities = torch.softmax(output, dim=1)
    prob = probabilities[0][predicted.item()]

    # Generate response
    if prob.item() < 0.75:
        response_text = "Apologies, I do not understand.."
    else:
        for intent in intents['intents']:
            if predicted_tag == intent['tag']:
                response_text = random.choice(intent['responses'])
                break

    return jsonify({"bot_name": bot_name, "response": response_text})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

