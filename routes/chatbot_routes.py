"""
Routes for fashion chatbot functionality
"""
from flask import render_template, request, jsonify
import os
import sys
import time

# Support local imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import chatbot service
try:
    from services.chatbot_service import get_chatbot_response
except ImportError as e:
    print(f"Error importing chatbot service: {e}")
    # Fallback function if chatbot service isn't available
    def get_chatbot_response(query, context=None):
        return f"I'd be happy to help with '{query}', but my fashion AI service is currently initializing. Please try again in a moment."

def register_chatbot_routes(app):
    """Register all chatbot-related routes"""
    
    @app.route("/chatbot")
    def chatbot_page():
        """Render the chatbot interface page"""
        return render_template("chatbot.html")

    @app.route("/api/chat", methods=["POST"])
    def chatbot_api():
        """API endpoint to get responses from the fashion chatbot"""
        start_time = time.time()
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
            
        user_message = data['message']
        chat_history = data.get('history', [])
        
        print(f"Chat request: '{user_message}' (with {len(chat_history)} history items)")
        
        try:
            # Get response from chatbot service
            response = get_chatbot_response(user_message, chat_history)
            
            # Log response time
            elapsed = time.time() - start_time
            print(f"Generated response in {elapsed:.2f}s: '{response[:50]}...'")
            
            return jsonify({
                "response": response,
                "sources": []  # Could include fashion knowledge sources in the future
            })
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Error in chat response: {str(e)}")
            print(f"Traceback: {error_traceback}")
            
            # Return a user-friendly error message
            return jsonify({
                "response": "I'm having trouble accessing my fashion knowledge right now. Could you please try asking your question again in a slightly different way?",
                "error": str(e)
            }), 200  # Return 200 instead of 500 to prevent browser console errors
