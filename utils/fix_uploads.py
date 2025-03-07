#!/usr/bin/env python3
"""
Emergency fix for the uploads directory and database synchronization.

This script:
1. Ensures the uploads directory exists
2. Synchronizes the database with actual files
3. Fixes any permissions issues
4. Creates necessary default images
"""

import os
import sys
import json
import shutil
from PIL import Image, ImageDraw

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
LABELS_FILE = os.path.join(ROOT_DIR, "labels.json")
MATCHES_FILE = os.path.join(ROOT_DIR, "matches.json")
DEFAULT_IMG_DIR = os.path.join(ROOT_DIR, "static/defaults")

def ensure_directory(directory):
    """Ensure directory exists with proper permissions."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Ensure it's writable
    if not os.access(directory, os.W_OK):
        try:
            # Try to make it writable
            os.chmod(directory, 0o755)
            print(f"Fixed permissions for: {directory}")
        except Exception as e:
            print(f"WARNING: Directory {directory} is not writable: {e}")
    
    return os.path.exists(directory) and os.access(directory, os.W_OK)

def load_json(filename):
    """Load JSON data from file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {filename} is corrupted. Creating backup and starting fresh.")
            backup_file = filename + ".backup"
            try:
                shutil.copy2(filename, backup_file)
                print(f"Created backup at: {backup_file}")
            except:
                pass
            return {}
    return {}

def save_json(data, filename):
    """Save JSON data to file."""
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to: {filename}")

def create_default_image():
    """Create a default image for missing files."""
    default_img_path = os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")
    try:
        # Create a better looking default image
        img = Image.new('RGB', (300, 300), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        
        # Draw a T-shirt shape
        width, height = img.size
        center_x, center_y = width//2, height//2
        
        # Shirt body
        body_points = [
            (center_x - 60, center_y - 60),  # Top left
            (center_x + 60, center_y - 60),  # Top right
            (center_x + 80, center_y + 80),  # Bottom right
            (center_x - 80, center_y + 80),  # Bottom left
        ]
        draw.polygon(body_points, outline=(100, 100, 100), fill=(220, 220, 220))
        
        # Left sleeve
        left_sleeve = [
            (center_x - 60, center_y - 60),  # Top shirt
            (center_x - 100, center_y - 90),  # Shoulder
            (center_x - 80, center_y - 30),  # Arm end
            (center_x - 60, center_y - 30),  # Armpit
        ]
        draw.polygon(left_sleeve, outline=(100, 100, 100), fill=(220, 220, 220))
        
        # Right sleeve
        right_sleeve = [
            (center_x + 60, center_y - 60),  # Top shirt
            (center_x + 100, center_y - 90),  # Shoulder
            (center_x + 80, center_y - 30),  # Arm end
            (center_x + 60, center_y - 30),  # Armpit
        ]
        draw.polygon(right_sleeve, outline=(100, 100, 100), fill=(220, 220, 220))
        
        # Neck
        draw.ellipse([center_x - 20, center_y - 80, center_x + 20, center_y - 40], 
                    outline=(100, 100, 100), fill=(245, 245, 245))
        
        # Add text
        draw.text((center_x - 60, center_y + 100), "Image Not Available", fill=(80, 80, 80))
        
        img.save(default_img_path)
        print(f"Created default image at: {default_img_path}")
        return True
    except Exception as e:
        print(f"Failed to create default image: {e}")
        return False

def main():
    """Main function to fix uploads and database synchronization."""
    print("Starting emergency fix for uploads directory and database synchronization...")
    
    # 1. Ensure all required directories exist
    uploads_ok = ensure_directory(UPLOAD_FOLDER)
    defaults_ok = ensure_directory(DEFAULT_IMG_DIR)
    
    if not uploads_ok:
        print("CRITICAL ERROR: Cannot create or access uploads directory!")
        return 1
    
    # 2. Create default image
    if defaults_ok:
        create_default_image()
    
    # 3. Load and validate database files
    labels = load_json(LABELS_FILE)
    matches = load_json(MATCHES_FILE)
    
    # 4. Get actual files in upload directory
    try:
        files = set(os.listdir(UPLOAD_FOLDER))
        print(f"Found {len(files)} files in uploads directory")
    except Exception as e:
        print(f"Error accessing uploads directory: {e}")
        files = set()
    
    # 5. Clean up labels that have no corresponding files
    removed_labels = []
    for filename in list(labels.keys()):
        if filename not in files:
            removed_labels.append(filename)
            del labels[filename]
    
    if removed_labels:
        print(f"Removed {len(removed_labels)} missing entries from labels.json")
        save_json(labels, LABELS_FILE)
    
    # 6. Clean up matches with missing files
    valid_matches = []
    for match in matches:
        new_item = match.get("new_item", {}).get("filename", "")
        best_match = match.get("best_match", {}).get("filename", "")
        
        if new_item in files and best_match in files:
            valid_matches.append(match)
    
    if len(valid_matches) != len(matches):
        print(f"Removed {len(matches) - len(valid_matches)} invalid matches")
        save_json(valid_matches, MATCHES_FILE)
    
    # 7. Test if we can write to uploads directory
    try:
        test_file = os.path.join(UPLOAD_FOLDER, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("Successfully tested write access to uploads directory")
    except Exception as e:
        print(f"WARNING: Cannot write to uploads directory: {e}")
    
    print("\n===== Summary =====")
    print(f"Uploads directory: {'OK' if uploads_ok else 'PROBLEM'}")
    print(f"Default images: {'OK' if defaults_ok else 'PROBLEM'}")
    print(f"Total images in uploads: {len(files)}")
    print(f"Total entries in labels.json: {len(labels)}")
    print(f"Total matches in matches.json: {len(valid_matches)}")
    print(f"Removed {len(removed_labels)} invalid label entries")
    print(f"Removed {len(matches) - len(valid_matches)} invalid matches")
    
    print("\nFix completed. Please check the logs for any remaining issues.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation aborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
