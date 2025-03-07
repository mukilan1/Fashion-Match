#!/usr/bin/env python3
"""
Database synchronization utility for F_M project.

This script ensures that the labels.json and matches.json files only
contain references to image files that actually exist in the uploads folder.
It also fixes any corrupted image files.
"""

import os
import sys
import json
import shutil
from PIL import Image

# Add parent directory to path so we can import app configuration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
UPLOAD_FOLDER = "uploads"
LABELS_FILE = "labels.json"
MATCHES_FILE = "matches.json"
DEFAULT_IMG_DIR = "static/defaults"

def load_json(filename):
    """Load JSON data from file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {filename} is corrupted. Creating backup and starting fresh.")
                # Create backup of corrupted file
                shutil.copy2(filename, filename + ".backup")
                return {}
    return {}

def save_json(data, filename):
    """Save JSON data to file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def check_image_file(file_path):
    """Check if a file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except:
        return False

def main():
    """Main function to synchronize database files."""
    print("Starting database synchronization...")
    
    # Ensure upload directory exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created missing upload folder: {UPLOAD_FOLDER}")
    
    # Ensure default image directory exists
    if not os.path.exists(DEFAULT_IMG_DIR):
        os.makedirs(DEFAULT_IMG_DIR)
        print(f"Created missing default image folder: {DEFAULT_IMG_DIR}")
    
    # Create default image if it doesn't exist
    default_img_path = os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")
    if not os.path.exists(default_img_path):
        try:
            # Create simple colored image as fallback
            img = Image.new('RGB', (300, 300), color=(240, 240, 240))
            # Add a simple clothing icon
            draw = ImageDraw.Draw(img)
            # Draw a T-shirt shape (simplified)
            width, height = img.size
            draw.rectangle([(width//3, height//3), (2*width//3, 2*height//3)], 
                          outline=(150, 150, 150), width=5)
            img.save(default_img_path)
            print(f"Created default image at: {default_img_path}")
        except Exception as e:
            print(f"Warning: Could not create default image: {e}")
    
    # Load data
    labels = load_json(LABELS_FILE)
    matches = load_json(MATCHES_FILE)
    
    # Get actual files in upload directory
    files = set(os.listdir(UPLOAD_FOLDER)) if os.path.exists(UPLOAD_FOLDER) else set()
    
    # Check and clean labels
    original_label_count = len(labels)
    removed_labels = []
    
    for filename in list(labels.keys()):
        if filename not in files:
            removed_labels.append(filename)
            del labels[filename]
            continue
        
        # Check if image is valid
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not check_image_file(file_path):
            print(f"Warning: {filename} is not a valid image. You may want to delete it.")
    
    if removed_labels:
        print(f"Removed {len(removed_labels)} invalid entries from labels file.")
        save_json(labels, LABELS_FILE)
    
    # Check and clean matches
    original_match_count = len(matches)
    valid_matches = []
    
    for match in matches:
        new_item_filename = match.get("new_item", {}).get("filename", "")
        best_match_filename = match.get("best_match", {}).get("filename", "")
        
        # Only keep matches where both files exist
        if new_item_filename in files and best_match_filename in files:
            valid_matches.append(match)
    
    if len(valid_matches) != original_match_count:
        print(f"Cleaned up matches: kept {len(valid_matches)} out of {original_match_count}.")
        save_matches(valid_matches)
    
    # Find files in uploads not in labels
    unlabeled_files = files - set(labels.keys())
    if unlabeled_files:
        print(f"Found {len(unlabeled_files)} files in uploads folder without entries in labels.json:")
        for filename in unlabeled_files:
            print(f"  - {filename}")
    
    # Summary
    print("\nSynchronization Summary:")
    print(f"  - Total files in uploads folder: {len(files)}")
    print(f"  - Total entries in labels.json: {len(labels)}")
    print(f"  - Total matches in matches.json: {len(valid_matches)}")
    print(f"  - Removed {len(removed_labels)} stale entries from labels.json")
    print(f"  - Removed {original_match_count - len(valid_matches)} invalid matches from matches.json")
    print(f"  - Found {len(unlabeled_files)} unlabeled files")
    
    print("\nAll done!")
    return 0

def save_matches(matches):
    """Save matches to the matches file."""
    with open(MATCHES_FILE, "w") as f:
        json.dump(matches, f, indent=2)

if __name__ == "__main__":
    try:
        from PIL import ImageDraw  # Import here for the T-shirt drawing
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation aborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)