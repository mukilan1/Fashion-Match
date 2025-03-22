"""
Image utility functions
"""
import os
import re
from PIL import Image
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, DEFAULT_IMG_DIR, DRESS_ITEMS

def validate_image(file_path):
    """
    Validate if a file is a valid image.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify it's an image
            return True, None
    except Exception as e:
        return False, str(e)

def get_default_image_path():
    """Get path to default image"""
    return os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")

def generate_label_from_filename(filename):
    """
    Generate a better label from the filename.
    Converts names like "blue_shirt_01.jpg" to "Blue Shirt"
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Remove numbers and special characters
    name = re.sub(r'[0-9()[\]{}]', '', name)
    
    # Title case and trim
    name = name.strip().title()
    
    # If empty after processing, use "Clothing Item"
    if not name:
        return "Clothing Item"
        
    return name

def guess_wearable_type(filename, label=""):
    """
    Make an educated guess about the wearable type from filename or label.
    """
    search_text = (label + " " + filename).lower()
    
    # Top wearable keywords
    top_items = ['shirt', 'top', 'blouse', 'jacket', 'sweater', 'hoodie', 'tshirt', 't-shirt',
                't shirt', 'pullover', 'sweatshirt', 'cardigan', 'vest', 'tank']
                
    # Bottom wearable keywords
    bottom_items = ['pants', 'jeans', 'shorts', 'skirt', 'trousers', 'slacks', 'leggings', 
                   'joggers', 'chinos', 'khakis']
                   
    # Dress items
    dress_items = ['dress', 'gown', 'frock', 'robe', 'jumpsuit', 'romper']
    
    # Check for each type
    for item in top_items:
        if item in search_text:
            return "top wearable"
            
    for item in bottom_items:
        if item in search_text:
            return "bottom wearable"
            
    for item in dress_items:
        if item in search_text:
            return "dress"
            
    # Default to top if we can't determine
    return "top wearable"

def guess_clothing_gender(filename, label=""):
    """
    Make an educated guess about the gender from filename or label.
    """
    search_text = (label + " " + filename).lower()
    
    # Check for explicit gender indicators
    if any(word in search_text for word in ['men', 'man', 'male', 'boy', 'gentleman']):
        return "Men's"
    
    if any(word in search_text for word in ['women', 'woman', 'female', 'girl', 'lady']):
        return "Women's"
        
    # Default to unisex
    return "unisex"

def get_all_images(labels_dict=None):
    """
    Retrieve all images from the database or storage.
    
    Args:
        labels_dict: Dictionary of labels, will load from file if None
    
    Returns:
        list: List of image dictionaries with metadata
    """
    # If no labels dictionary is provided, load it
    if labels_dict is None:
        from utils.file_utils import load_labels
        labels_dict = load_labels()
    
    # Get the list of actual files in the uploads directory
    files = os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else []
    
    images = []
    for filename in files:
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            continue
            
        # Get metadata from labels.json if available
        entry = labels_dict.get(filename, {})
        
        # Check if file exists and is a valid image
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        is_valid_image = os.path.exists(file_path)
        
        if is_valid_image:
            is_valid_image, _ = validate_image(file_path)
        
        # Generate better defaults for missing values
        label = entry.get("label")
        if not label or label == "unknown":
            label = generate_label_from_filename(filename)
        
        # Get or determine wearable type with better defaults
        wearable = entry.get("wearable")
        if not wearable or wearable == "Unknown" or wearable == "unknown":
            wearable = guess_wearable_type(filename, label)
        
        # Check for dress-like items
        label_lower = label.lower()
        entry_dress_match = any(item in label_lower for item in DRESS_ITEMS)
        if entry_dress_match:
            wearable = "dress"
        
        # Get costume information with confidence
        costume = entry.get("costume", "casual")  # Default to casual instead of unknown
        costume_display = entry.get("costume_display", "Casual")
        costume_confidence = entry.get("costume_confidence", 0)
        costume_description = entry.get("costume_description", "")
        
        # Better defaults for gender/sex
        sex = entry.get("sex")
        if not sex or sex == "Unknown" or sex == "unknown":
            sex = guess_clothing_gender(filename, label)
        
        # Better defaults for pattern
        pattern = entry.get("pattern", "solid")  # Default to solid instead of unknown
        
        # Better defaults for sleeve/hand
        hand = entry.get("hand")
        if not hand or hand == "Unknown" or hand == "unknown":
            if wearable == "top wearable":
                hand = "full hand"  # Default for tops
            elif wearable == "bottom wearable":
                hand = "N/A (Bottom Wear)"
            else:
                hand = "full hand"  # Default for other items
        
        # Add the image data with improved defaults
        images.append({
            "filename": filename,
            "label": label,
            "wearable": wearable,
            "costume": costume,
            "costume_display": costume_display,
            "costume_confidence": costume_confidence,
            "costume_description": costume_description,
            "color": entry.get("color", "#3b82f6"),  # Default to a nice blue instead of gray
            "pattern": pattern,
            "sex": sex,
            "gender_confidence": entry.get("gender_confidence", 0),
            "hand": hand,
            "is_valid": is_valid_image,
            "is_dress_item": entry_dress_match
        })
    
    return images
