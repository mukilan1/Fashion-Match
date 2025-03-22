"""
Configuration settings for the Fashion Matching application
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Uploads folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Default images directory
DEFAULT_IMG_DIR = os.path.join(BASE_DIR, "static/defaults")
if not os.path.exists(DEFAULT_IMG_DIR):
    os.makedirs(DEFAULT_IMG_DIR, exist_ok=True)
    # Create a simple default image if it doesn't exist
    default_img_path = os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")
    if not os.path.exists(default_img_path):
        try:
            from PIL import Image
            # Create simple colored image as fallback
            img = Image.new('RGB', (300, 300), color=(240, 240, 240))
            img.save(default_img_path)
        except Exception as e:
            print(f"Could not create default image: {e}")

# JSON data files
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
MATCHES_FILE = os.path.join(BASE_DIR, "matches.json")

# Model configuration
SBERT_MODEL_NAME = 'all-mpnet-base-v2'
OLLAMA_MODEL_NAME = 'deepseek-r1:1.5b'

# Color compatibility pairs for fashion matching
COLOR_PAIRS = {
    "black": ["white", "gray", "red", "blue", "green", "yellow", "purple", "pink"],
    "white": ["black", "navy", "red", "blue", "green", "purple", "pink"],
    "blue": ["white", "gray", "navy", "khaki", "brown", "pink", "black"],
    "navy": ["white", "khaki", "gray", "brown", "red"],
    "red": ["black", "white", "navy", "gray", "khaki"],
    "gray": ["black", "white", "navy", "blue", "red", "purple"],
    "green": ["black", "khaki", "white", "brown"],
    "brown": ["khaki", "blue", "white", "green"],
    "khaki": ["navy", "brown", "green", "red"],
    "pink": ["navy", "white", "gray", "black"],
    "purple": ["white", "black", "gray"]
}

# Minimum match threshold
MIN_MATCH_THRESHOLD = 0.4

# Dress item keywords
DRESS_ITEMS = ['dress', 'gown', 'sarong', 'frock', 'skirt', 'lehenga', 'saree', 'sari']
