"""
File operation utilities for the Fashion Matching application
"""
import json
import os
import re
from werkzeug.utils import secure_filename

from config import LABELS_FILE, MATCHES_FILE, UPLOAD_FOLDER

def load_labels():
    """Load the labels from the JSON file"""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_labels(labels):
    """Save the labels to the JSON file"""
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)

def load_matches():
    """Load the matches from the JSON file"""
    if os.path.exists(MATCHES_FILE):
        with open(MATCHES_FILE, "r") as f:
            return json.load(f)
    return []

def save_matches(matches):
    """Save the matches to the JSON file"""
    with open(MATCHES_FILE, "w") as f:
        json.dump(matches, f)

def generate_metadata_from_filename(filename):
    """Generate initial metadata from the filename"""
    name = os.path.splitext(filename)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'[0-9()[\]{}]', '', name).strip().title()
    
    # Defaults for a new item
    return {
        "label": name if name else "Clothing Item",
        "wearable": "top wearable",  # Default to top
        "costume": "casual",  # Default to casual
        "costume_display": "Casual",
        "costume_confidence": 0.5,
        "costume_description": "",
        "color": "#3b82f6",  # Blue
        "color_detail": "blue",
        "pattern": "solid",  # Default to solid
        "pattern_confidence": 0.5,
        "sex": "unisex",  # Default to unisex
        "gender_confidence": 0.5,
        "hand": "full hand"  # Default to full sleeve
    }

def synchronize_labels():
    """Ensure labels.json only contains entries for files that actually exist"""
    labels = load_labels()
    files = set(os.listdir(UPLOAD_FOLDER)) if os.path.exists(UPLOAD_FOLDER) else set()
    
    # Remove entries from labels that don't exist as files
    removed_entries = []
    for filename in list(labels.keys()):
        if filename not in files:
            removed_entries.append(filename)
            del labels[filename]
    
    # Add default entries for files that don't have metadata
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) and filename not in labels:
            # Generate default metadata based on filename
            labels[filename] = generate_metadata_from_filename(filename)
    
    # Save the updated labels
    if removed_entries or any(f for f in files if f not in labels):
        print(f"Removed {len(removed_entries)} stale entries and added metadata for new files")
        save_labels(labels)
    
    return removed_entries

def synchronize_matches():
    """Remove matches with missing image files"""
    labels = load_labels()
    files = set(os.listdir(UPLOAD_FOLDER)) if os.path.exists(UPLOAD_FOLDER) else set()
    matches = load_matches()
    
    valid_matches = []
    for match in matches:
        new_item_filename = match.get("new_item", {}).get("filename", "")
        best_match_filename = match.get("best_match", {}).get("filename", "")
        
        # Only keep matches where both files exist
        if new_item_filename in files and best_match_filename in files:
            valid_matches.append(match)
    
    # Save if any were removed
    if len(valid_matches) != len(matches):
        print(f"Removed {len(matches) - len(valid_matches)} stale matches")
        save_matches(valid_matches)
    
    return valid_matches
