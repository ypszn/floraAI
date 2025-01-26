from flask import Blueprint, request, jsonify
from app.chatbot import generate_response
from app.utils import sanitize_input, log_error, log_info

api = Blueprint('api', __name__)

@api.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint for the therapy chatbot.
    """
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        log_error("No message provided by the user.")
        return jsonify({"error": "No message provided"}), 400

    # Sanitize input
    sanitized_input = sanitize_input(user_input)
    log_info(f"Received sanitized input: {sanitized_input}")

    # Generate response
    try:
        response = generate_response(sanitized_input)
        log_info(f"Generated response: {response}")
        return jsonify({"response": response})
    except Exception as e:
        log_error(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500
