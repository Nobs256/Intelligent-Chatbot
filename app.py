from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
from datetime import datetime

import os
from main import ChatbotAssistant

app = Flask(__name__)

def load_chatbot():
    """
    Loads the chatbot assistant and the trained model.
    """
    assistant = ChatbotAssistant('intents.json')
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')
    return assistant

assistant = load_chatbot()

@app.route("/")
def home():
    """
    Renders the main chat page.
    """
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    """
    Gets a response from the chatbot for the user's message.
    """
    user_text = request.args.get('msg')
    bot_data = assistant.process_message(user_text)
    return jsonify(bot_data)

@app.route("/feedback", methods=['POST'])
def handle_feedback():
    """
    Saves user feedback to a file for later review.
    """
    data = request.get_json()
    
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_message": data.get("user_message"),
        "bot_response": data.get("bot_response"),
        "predicted_intent": data.get("predicted_intent"),
        "feedback": data.get("feedback_type")
    }
    
    # Append feedback to a file in JSON Lines format for easy parsing later
    with open("feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
        
    return jsonify({"status": "success", "message": "Feedback received"})

@app.route("/review")
def review_feedback():
    """
    Displays the collected feedback for review.
    """
    feedback_data = []
    feedback_file = "feedback.jsonl"
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            for line in f:
                feedback_data.append(json.loads(line))
    
    # Show newest feedback first
    feedback_data.reverse()

    # Pass all intent tags to the template for the dropdown
    all_intents = assistant.intents

    return render_template("review.html", feedback=feedback_data, all_intents=all_intents)

@app.route("/correct_intent", methods=['POST'])
def correct_intent():
    """
    Adds a user message as a new pattern to the correct intent in intents.json.
    """
    user_message = request.form.get('user_message')
    correct_intent_tag = request.form.get('correct_intent')

    if not user_message or not correct_intent_tag:
        return redirect(url_for('review_feedback'))

    intents_file_path = 'intents.json'
    with open(intents_file_path, 'r') as f:
        intents_data = json.load(f)

    for intent in intents_data['intents']:
        if intent['tag'] == correct_intent_tag:
            if user_message not in intent['patterns']:
                intent['patterns'].append(user_message)
            break
    
    with open(intents_file_path, 'w') as f:
        json.dump(intents_data, f, indent=4)

    return redirect(url_for('review_feedback'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)