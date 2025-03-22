"""
Entry point for running the Fashion Matching application
"""
import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the app and run it
from app import app

if __name__ == '__main__':
    print("Fashion application running with full features at: http://127.0.0.1:8080/")
    print("Access pose rigging visualization at: http://127.0.0.1:8080/pose_rigging")  # Keep this line
    app.run(debug=True, threaded=True, port=8080)
