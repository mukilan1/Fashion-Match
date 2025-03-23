"""
Main application entry point
"""
import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from flask import Flask, g, render_template, request, jsonify, redirect, url_for, Response
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
from config import UPLOAD_FOLDER, DEFAULT_IMG_DIR

# Import route handlers
from routes.main_routes import register_main_routes
from routes.match_routes import register_match_routes
from routes.analysis_routes import register_analysis_routes
from routes.media_routes import register_media_routes
from routes.chatbot_routes import register_chatbot_routes

# Import dress search service
from services.dress_search import search_dresses

# Import utilities
from utils.progress import show_progress
from utils.image_data import get_all_images, classify_wearable, is_top_wearable, is_bottom_wearable

def create_app():
    """Create and configure the Flask app"""
    app = Flask(__name__)
    
    # Configure app
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max size
    
    # Show initialization progress
    show_progress("Starting server", 25, "Initializing routes")
    
    # Register routes - ensure these are working
    try:
        register_main_routes(app)
        print("✓ Main routes registered")
        
        register_match_routes(app)
        print("✓ Match routes registered")
        
        register_analysis_routes(app)
        print("✓ Analysis routes registered")
        
        register_media_routes(app)
        print("✓ Media routes registered")
        
        register_chatbot_routes(app)
        print("✓ Chatbot routes registered")
    except Exception as e:
        print(f"ERROR registering routes: {str(e)}")
        raise
    
    # Add CORS headers to all routes
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    # Add error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error.html', error="Page not found", status_code=404), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('error.html', error="Server error", status_code=500), 500
    
    show_progress("Starting server", 100, "Routes initialized", final=True)
    
    return app

# Create the Flask app
app = create_app()

# Define the direct routes here (outside of route modules)
@app.route('/dress_search', methods=['GET', 'POST'])
def dress_search():
    """Handle dress search page and API requests"""
    try:
        if request.method == 'POST':
            prompt = request.form.get('prompt', '').strip()
            if prompt:
                # Use the imported search_dresses function
                logger.info(f"Received dress search request for: {prompt}")
                results = search_dresses(prompt)
                logger.info(f"Search complete, found {len(results.get('results', []))} results")
                return jsonify(results)
            else:
                logger.warning("Empty search prompt received")
                return jsonify({"error": "Search term cannot be empty", "results": []})
        
        # For GET requests, show the search page
        return render_template('dress_search.html')
    except Exception as e:
        logger.error(f"Error in dress search: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "results": []})

@app.route('/dress_match')
def dress_match():
    """Display dress matching page with tops and bottoms"""
    try:
        # Get all images from the existing labels.json
        all_images = get_all_images()
        
        # Log how many items were found
        logger.info(f"Found {len(all_images)} clothing items in total")
        
        # Filter tops and bottoms using the wearable field pattern from labels.json
        tops = [item for item in all_images if is_top_wearable(item.get('wearable', ''))]
        bottoms = [item for item in all_images if is_bottom_wearable(item.get('wearable', ''))]
        
        logger.info(f"Classified {len(tops)} tops and {len(bottoms)} bottoms for dress matcher")
        
        return render_template('dress_match.html', tops=tops, bottoms=bottoms)
    except Exception as e:
        logger.error(f"Error in dress_match route: {str(e)}")
        return render_template('error.html', error="Failed to load clothing data", status_code=500)

@app.route('/download-image')
def download_image():
    """Proxy for downloading images from external sources"""
    url = request.args.get('url')
    filename = request.args.get('filename', 'image.jpg')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
        
    try:
        # Request the image from the source
        response = requests.get(url, stream=True, timeout=10)
        
        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch image: {response.status_code}"}), 400
            
        # Get content type
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        
        # Return the image with appropriate headers for download
        return Response(
            response.raw.read(),
            headers={
                'Content-Type': content_type,
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return jsonify({"error": f"Failed to download: {str(e)}"}), 500

# Ensure there's a named route for the index page
# Remove the conflicting index route definition
# @app.route('/')
# def index():
#     """Display the main inventory page"""
#     # Get images from routes/main_routes.py
#     # This is likely already defined in your main_routes.py file
#     # We just need to make sure it has the proper name 'index'
#     return render_template('index.html', images=get_images())

# Remove this helper function as well since it's not needed
# def get_images():
#     """Get all images for display"""
#     try:
#         from routes.main_routes import get_all_images_for_display
#         return get_all_images_for_display()
#     except:
#         # Fallback if the function isn't available
#         return []

if __name__ == '__main__':
    print("Fashion application running with full features at: http://127.0.0.1:8080/")
    print("Access pose rigging visualization at: http://127.0.0.1:8080/pose_rigging")
    print("Access dress search tool at: http://127.0.0.1:8080/dress_search")
    app.run(debug=True, threaded=True, port=8080)
