"""
Main application entry point
"""
import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from flask import Flask, g

# Import configuration
from config import UPLOAD_FOLDER, DEFAULT_IMG_DIR

# Import route handlers
from routes.main_routes import register_main_routes
from routes.match_routes import register_match_routes
from routes.analysis_routes import register_analysis_routes
from routes.media_routes import register_media_routes

# Import utilities
from utils.progress import show_progress

def create_app():
    """Create and configure the Flask app"""
    app = Flask(__name__)
    
    # Configure app
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    # Show initialization progress
    show_progress("Starting server", 25, "Initializing routes")
    
    # Register routes
    register_main_routes(app)
    register_match_routes(app)
    register_analysis_routes(app)
    register_media_routes(app)
    
    # Add template context processor for common URLs
    @app.context_processor
    def inject_common_urls():
        """Make common URLs available to all templates"""
        return {
            'url_home': '/',
            'url_match': '/match',
            'url_pose_rigging': '/pose_rigging',
        }
    
    show_progress("Starting server", 100, "Routes initialized", final=True)
    
    return app

# Create the Flask app
app = create_app()

if __name__ == '__main__':
    print("Fashion application running with full features at: http://127.0.0.1:8080/")
    print("Access pose rigging visualization at: http://127.0.0.1:8080/pose_rigging")
    app.run(debug=True, threaded=True, port=8080)
