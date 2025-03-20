# Update these imports to use the correct paths
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import time
from werkzeug.utils import secure_filename
import json
from PIL import Image
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import ollama
import shutil
from rembg import remove  
import io
# Fix import paths to use Model_Props directory
from Model_Props.image_analyzer import ClothingImageAnalyzer
from Model_Props.color_analyzer import analyze_colors

# Add a progress indicator function
def show_progress(operation, percent=0, status="", final=False):
    """
    Display a progress indicator in the console.
    
    Args:
        operation: String describing the current operation
        percent: Progress percentage (0-100)
        status: Additional status message
        final: Whether this is the final update for this operation
    """
    bar_length = 20
    filled_length = int(bar_length * percent / 100)
    bar = '■' * filled_length + '□' * (bar_length - filled_length)
    
    if final:
        sys.stdout.write(f"\r{operation}: [{bar}] {percent:.1f}% - {status} ✓ Completed\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r{operation}: [{bar}] {percent:.1f}% - {status}")
        sys.stdout.flush()

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

LABELS_FILE = "labels.json"
MATCHES_FILE = "matches.json"

# Initialize the image analyzer with progress reporting
show_progress("Initializing image analyzer", 0, "Loading modules")
image_analyzer = ClothingImageAnalyzer(use_vqa=True, use_ml_model=True)
show_progress("Initializing image analyzer", 100, "Ready", final=True)

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)

def load_matches():
    if os.path.exists(MATCHES_FILE):
        with open(MATCHES_FILE, "r") as f:
            return json.load(f)
    return []

def save_matches(matches):
    with open(MATCHES_FILE, "w") as f:
        json.dump(matches, f)

def generate_dynamic_reason(new_item, candidate, score):
    prompt = (
        f"Why do these match? New: {new_item['label']}, {new_item['costume']}, {new_item['pattern']}, "
        f"{new_item['color']}. Candidate: {candidate['label']}, {candidate['costume']}, {candidate['pattern']}, "
        f"{candidate['color']}. Score: {score:.2f}. Answer in less than 2 lines."
    )
    response = ollama.chat(model="deepseek-r1:1.5b", messages=[{'role': 'user', 'content': prompt}])
    result_text = response.get('message', {}).get('content', "").strip()
    
    # Clean up the response: remove <think> tags and other unwanted content
    cleaned_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()
    
    # If the result is empty or too short after cleaning, use a default message
    if not cleaned_text or len(cleaned_text) < 10:
        cleaned_text = "These fabrics complement each other based on design and color."
    
    # Return only the first line
    return cleaned_text.split("\n")[0]

# Add a new function to synchronize labels.json with actual files
def synchronize_labels():
    """Ensure labels.json only contains entries for files that actually exist"""
    labels = load_labels()
    files = set(os.listdir(app.config["UPLOAD_FOLDER"])) if os.path.exists(app.config["UPLOAD_FOLDER"]) else set()
    
    # Remove entries from labels that don't exist as files
    removed_entries = []
    for filename in list(labels.keys()):
        if filename not in files:
            removed_entries.append(filename)
            del labels[filename]
    
    # Save the cleaned labels
    if removed_entries:
        print(f"Removed {len(removed_entries)} stale entries from labels.json")
        save_labels(labels)
    
    return removed_entries

# Do the same for matches
def synchronize_matches():
    """Remove matches with missing image files"""
    labels = load_labels()
    files = set(os.listdir(app.config["UPLOAD_FOLDER"])) if os.path.exists(app.config["UPLOAD_FOLDER"]) else set()
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

# Create backup directory for default images if it doesn't exist
DEFAULT_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static/defaults")
if not os.path.exists(DEFAULT_IMG_DIR):
    os.makedirs(DEFAULT_IMG_DIR, exist_ok=True)
    # Create a simple default image if it doesn't exist
    default_img_path = os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")
    if not os.path.exists(default_img_path):
        try:
            # Create simple colored image as fallback
            img = Image.new('RGB', (300, 300), color=(240, 240, 240))
            img.save(default_img_path)
        except Exception as e:
            print(f"Could not create default image: {e}")

# New function to remove background from images
def remove_background(input_image):
    """
    Removes the background from an image and returns the processed image.
    
    Args:
        input_image: PIL Image or file-like object
    Returns:
        PIL Image with transparent background
    """
    try:
        show_progress("Removing background", 10, "Preparing image")
        # Convert PIL Image to bytes if needed
        if isinstance(input_image, Image.Image):
            img_byte_arr = io.BytesIO()
            input_image.save(img_byte_arr, format='PNG')
            img_data = img_byte_arr.getvalue()
            show_progress("Removing background", 30, "Image converted")
        else:
            # Assume input is a file path
            with open(input_image, 'rb') as img_file:
                img_data = img_file.read()
                show_progress("Removing background", 30, "Image loaded")
        
        # Process the image to remove background
        show_progress("Removing background", 50, "Processing")
        output_data = remove(img_data)
        show_progress("Removing background", 80, "Background removed")
        
        # Convert back to PIL Image
        result_image = Image.open(io.BytesIO(output_data))
        show_progress("Removing background", 100, "Complete", final=True)
        return result_image
    except Exception as e:
        show_progress("Removing background", 100, f"Failed: {str(e)}", final=True)
        # Return the original image if background removal fails
        if isinstance(input_image, str) and os.path.exists(input_image):
            return Image.open(input_image)
        return input_image

def parse_clothing_analysis(classification):
    """
    Parse a detailed clothing classification string into separate components.
    
    For example:
    "Men's Top (Full body garment: Suit), gray, solid color pattern, long sleeve"
    becomes:
    {
        "wearable": "Top",
        "sex": "Men's",
        "color": "gray",
        "pattern": "solid color",
        "hand": "long sleeve",
        "costume": "formal"
    }
    """
    result = {
        "wearable": "unknown",
        "sex": "unknown",
        "color": "unknown",
        "pattern": "unknown",
        "hand": "unknown",
        "costume": "unknown"
    }
    
    if not classification or classification == "unknown":
        return result
    
    # Extract gender/sex information
    if classification.startswith("Men's"):
        result["sex"] = "Men's"
        classification = classification[6:].strip()  # Remove "Men's " prefix
    elif classification.startswith("Women's"):
        result["sex"] = "Women's"
        classification = classification[8:].strip()  # Remove "Women's " prefix
    
    # Extract wearable type (Top or Bottom)
    if classification.startswith("Top"):
        result["wearable"] = "top wearable"
    elif classification.startswith("Bottom"):
        result["wearable"] = "bottom wearable"
    
    # Extract full body garment type (usually in parentheses)
    if "Full body garment:" in classification:
        result["costume"] = "formal"
        if "Dress" in classification:
            result["wearable"] = "dress"  # Special category for dresses
    
    # Extract color, pattern, and sleeve information
    parts = classification.split(',')
    for part in parts:
        part = part.strip().lower()
        
        # Check for colors
        colors = ["black", "white", "red", "blue", "green", "yellow", "purple", "pink", "orange", 
                 "brown", "gray", "grey", "navy", "teal", "maroon", "olive", "tan"]
        for color in colors:
            if color in part:
                result["color"] = color
                break
        
        # Check for patterns
        if "pattern" in part or any(p in part for p in ["solid", "striped", "checkered", "plaid", "floral", "polka dot", "dotted"]):
            if "solid" in part:
                result["pattern"] = "solid"
            elif "striped" in part:
                result["pattern"] = "striped"
            elif "checkered" in part or "plaid" in part:
                result["pattern"] = "checkered"
            elif "floral" in part:
                result["pattern"] = "floral"
            elif "polka dot" in part or "dotted" in part:
                result["pattern"] = "polka dot"
            else:
                result["pattern"] = part
        
        # Check for sleeve type
        if "sleeve" in part:
            if "long" in part:
                result["hand"] = "full hand"
            elif "short" in part or "half" in part:
                result["hand"] = "half hand"
            elif "no" in part or "sleeveless" in part:
                result["hand"] = "no hand"
            else:
                result["hand"] = part
    
    return result

@app.route("/")
def index():
    # Synchronize the database with actual files
    synchronize_labels()
    
    # Continue with existing code
    labels = load_labels()
    files = os.listdir(app.config["UPLOAD_FOLDER"]) if os.path.exists(app.config["UPLOAD_FOLDER"]) else []
    images_data = []
    
    for f in files:
        entry = labels.get(f, {})
        # Verify file is readable as image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
        is_valid_image = os.path.exists(file_path)
        
        if is_valid_image:
            try:
                # Attempt to open the image to verify it's valid
                with Image.open(file_path) as img:
                    img.verify()  # Verify it's an image
            except Exception:
                is_valid_image = False
        
        # Include a flag to frontend indicating if image is valid
        images_data.append({
            "filename": f,
            "label": entry.get("label", "unknown"),
            "wearable": entry.get("wearable", "unknown"),
            "costume": entry.get("costume", "unknown"),
            "color": entry.get("color", "unknown"),
            "pattern": entry.get("pattern", "unknown"),
            "sex": entry.get("sex", "unknown"),
            "hand": entry.get("hand", "unknown"),
            "is_valid": is_valid_image
        })
    
    return render_template("index.html", images=images_data)

@app.route("/upload", methods=["POST"])
def upload():
    if "images" not in request.files:
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist("images")
    labels = load_labels()
    results = []
    
    # Ensure uploads directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        except Exception as e:
            return jsonify({"error": f"Failed to create upload directory: {str(e)}"}), 500
    
    total_files = len(files)
    for i, file in enumerate(files):
        if file.filename == "":
            continue
        try:
            show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30, file.filename)
            filename = secure_filename(file.filename)
            
            # Create temporary file path
            temp_path = os.path.join("/tmp", filename)
            file.save(temp_path)
            show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 10, "File saved")
            
            # Remove background from image
            try:
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 20, "Removing background")
                processed_image = remove_background(temp_path)
                
                # Save processed image (with transparent background)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save with transparent background
                processed_image.save(file_path, format="PNG")
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 40, "Background removed")
                
                print(f"Background removed and saved: {filename}")
            except Exception as e:
                print(f"Background removal failed for {filename}: {str(e)}")
                # If background removal fails, use original image
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy(temp_path, file_path)
                
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # MODIFIED: Use ONLY the image analyzer for classification
            show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 70, "Analyzing image")
            analysis_results = image_analyzer.analyze_image(file_path)
            
            # Extract the classification directly from the analyzer results
            label = analysis_results.get("classification", "unknown")
            confidence = analysis_results.get("confidence", 0)
            wearable_position = analysis_results.get("wearable_position", "unknown")  # Get wearable position
            
            # Ensure we always have a valid wearable position
            if wearable_position == "unknown":
                # Try to determine from label as fallback
                if "shirt" in label.lower() or "top" in label.lower() or "jacket" in label.lower():
                    wearable_position = "top wearable"
                elif "pants" in label.lower() or "jeans" in label.lower() or "skirt" in label.lower():
                    wearable_position = "bottom wearable"
                else:
                    # If we really can't determine, assign as top wearable by default
                    wearable_position = "top wearable"
            
            show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 90, f"Analysis completed - {wearable_position}")
            
            # Add color analysis
            try:
                color_info = analyze_colors(file_path)
                primary_color = color_info['primary_color']
                colors_text = color_info['colors_text']
                
                print(f"Detected colors: {colors_text}")
            except Exception as e:
                print(f"Color analysis failed: {str(e)}")
                primary_color = "unknown"
                colors_text = "unknown"
            
            # Store in labels with wearable position and color
            labels[filename] = {
                "label": label,
                "wearable": wearable_position,  # Use wearable_position for the wearable field
                "costume": "unknown",
                "color": primary_color,
                "color_detail": colors_text,
                "pattern": "unknown",
                "sex": "unknown",
                "hand": "unknown"
            }
            
            results.append({
                "filename": filename,
                "label": label,
                "wearable": wearable_position,  # Use wearable_position for the wearable field
                "costume": "unknown",
                "color": primary_color,
                "color_detail": colors_text,
                "pattern": "unknown",
                "sex": "unknown",
                "hand": "unknown",
                "path": file_path,
                "background_removed": True,
                "analysis_confidence": confidence
            })
            show_progress(f"Processing file {i+1}/{total_files}", (i + 1) / total_files * 100, "Complete", final=(i==total_files-1))
        except Exception as e:
            # Continue with other files if one fails
            show_progress(f"Processing file {i+1}/{total_files}", 100, f"Error: {str(e)}", final=True)
            print(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename if file.filename else "unknown",
                "error": str(e)
            })
    
    # Save the updated labels
    try:
        show_progress("Saving metadata", 50, "Writing to disk")
        save_labels(labels)
        show_progress("Saving metadata", 100, "Complete", final=True)
    except Exception as e:
        show_progress("Saving metadata", 100, f"Failed: {str(e)}", final=True)
        return jsonify({"error": f"Failed to save labels: {str(e)}", "partial_results": results}), 500
    
    # Include progress information in the response
    return jsonify({
        "uploaded": results,
        "progress_complete": True
    }), 200

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    # Secure the filename
    secure_name = secure_filename(filename)
    
    # Check if file exists in uploads directory with absolute path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        # Return default image instead of 404
        default_img_path = os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")
        return send_from_directory(os.path.dirname(default_img_path), 
                                  os.path.basename(default_img_path))
    
    # Try to verify if it's a valid image
    try:
        with Image.open(file_path) as img:
            img.verify()  # Will raise exception if not valid image
    except Exception as e:
        print(f"Invalid image file {filename}: {str(e)}")
        # Return default image for corrupted images
        default_img_path = os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")
        return send_from_directory(os.path.dirname(default_img_path), 
                                  os.path.basename(default_img_path))
    
    # Set cache control headers and return the file
    try:
        response = send_from_directory(app.config['UPLOAD_FOLDER'], secure_name)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        print(f"Error sending file {filename}: {str(e)}")
        default_img_path = os.path.join(DEFAULT_IMG_DIR, "default_clothing.png")
        return send_from_directory(os.path.dirname(default_img_path), 
                                  os.path.basename(default_img_path))

@app.route("/delete", methods=["POST"]) 
def delete_image():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided."}), 400
    secure_name = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
    labels = load_labels()
    entry = labels.get(secure_name, {})
    label = entry.get("label", "unknown")
    wearable = entry.get("wearable", "unknown")
    if os.path.exists(file_path):
        os.remove(file_path)
        if secure_name in labels:
            del labels[secure_name]
            save_labels(labels)
        return jsonify({"deleted": secure_name, "label": label, "wearable": wearable}), 200
    return jsonify({"error": "File not found."}), 404

@app.route("/delete_all", methods=["POST"])
def delete_all():
    labels = load_labels()
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        if filename in labels:
            del labels[filename]
    save_labels(labels)
    return jsonify({"deleted_all": True, "message": "All files deleted."}), 200

# New: Match-making page. Pass stored images for matching history.
@app.route("/match", methods=["GET"])
def match_page():
    # Synchronize both labels and matches
    synchronize_labels()
    synchronize_matches()
    
    # Continue with existing code
    labels = load_labels()
    files = os.listdir(app.config["UPLOAD_FOLDER"]) if os.path.exists(app.config["UPLOAD_FOLDER"]) else []
    images_data = []
    
    for f in files:
        entry = labels.get(f, {})
        # Verify file is readable as image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
        is_valid_image = os.path.exists(file_path)
        
        if is_valid_image:
            try:
                # Attempt to open the image to verify it's valid
                with Image.open(file_path) as img:
                    img.verify()  # Verify it's an image
            except Exception:
                is_valid_image = False
        
        images_data.append({
            "filename": f,
            "label": entry.get("label", "unknown"),
            "wearable": entry.get("wearable", "unknown"),
            "costume": entry.get("costume", "unknown"),
            "color": entry.get("color", "unknown"),
            "pattern": entry.get("pattern", "unknown"),
            "sex": entry.get("sex", "unknown"),
            "hand": entry.get("hand", "unknown"),
            "is_valid": is_valid_image
        })
    
    return render_template("match.html", images=images_data)

# New: Handle match upload: compare the uploaded item with stored items for the best pair.
@app.route("/match_upload", methods=["POST"])
def match_upload():
    if "match_image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["match_image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    show_progress("Match processing", 10, "Starting match analysis")    
    filename = secure_filename(file.filename)
    
    # Create temporary file path
    temp_path = os.path.join("/tmp", filename)
    file.save(temp_path)
    show_progress("Match processing", 20, "File saved")
    
    # Remove background from image
    try:
        show_progress("Match processing", 30, "Removing background")
        processed_image = remove_background(temp_path)
        
        # Save processed image (with transparent background)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        processed_image.save(file_path, format="PNG")
        show_progress("Match processing", 40, "Background removed")
    except Exception as e:
        print(f"Background removal failed for {filename}: {str(e)}")
        # If background removal fails, use original image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        shutil.copy(temp_path, file_path)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # Process the image - CHANGED: use image_analyzer instead of classify_cloth
    show_progress("Match processing", 50, "Analyzing image")
    analysis_result = image_analyzer.analyze_image(file_path)
    new_label = analysis_result.get("classification", "unknown")
    
    # Parse classification into separate fields
    classification = analysis_result.get("classification", "unknown")
    parsed_details = parse_clothing_analysis(classification)
    new_wearable = parsed_details["wearable"]
    
    show_progress("Match processing", 60, "Processing completed")
    
    new_text = f"{new_label} {new_wearable} unknown unknown unknown unknown"
    stored = load_labels()
    candidates = []
    for fname, data in stored.items():
        if fname == filename:  # Skip matching with self
            continue
        cand_text = f"{data.get('label','')} {data.get('wearable','')} {data.get('costume','')} {data.get('pattern','')} {data.get('color','')} {data.get('sex','')}"
        candidates.append((fname, cand_text, data))
    if not candidates:
        show_progress("Match processing", 100, "No candidates found", final=True)
        return jsonify({"error": "No candidate found for matching."}), 404
    
    # Initialize SBERT model for semantic matching.
    show_progress("Match processing", 70, "Finding best match")
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    new_embedding = sbert_model.encode(new_text, convert_to_tensor=True)
    best_candidate, best_score = None, -1
    for fname, cand_text, data in candidates:
        cand_embedding = sbert_model.encode(cand_text, convert_to_tensor=True)
        score = util.cos_sim(new_embedding, cand_embedding).item()
        if score > best_score:
            best_score = score
            best_candidate = {"filename": fname, "data": data, "score": best_score}
    show_progress("Match processing", 80, "Match found")
    
    result = {
        "new_item": {
            "filename": filename,
            "label": new_label,
            "wearable": new_wearable,
            "color": "unknown",
            "costume": "unknown",
            "pattern": "unknown",
            "sex": "unknown",
            "hand": "unknown"
        },
        "best_match": best_candidate
    }
    
    # Instead of a fixed reason, generate a dynamic reason using our NLP model.
    show_progress("Match processing", 90, "Generating match explanation")
    reason_text = generate_dynamic_reason(result["new_item"], best_candidate["data"], best_score)
    result["reason"] = reason_text
    matches = load_matches()
    matches.append(result)
    save_matches(matches)
    show_progress("Match processing", 100, "Match completed", final=True)
    
    return jsonify({"matched": result}), 200

@app.route("/matches", methods=["GET"])
def get_matches():
    matches = load_matches()
    return jsonify({"matches": matches}), 200

@app.route("/auto_match", methods=["GET"])
def auto_match():
    labels = load_labels()
    tops = []
    bottoms = []
    for filename, data in labels.items():
        text = f"{data.get('label','')} {data.get('costume','')} {data.get('pattern','')} {data.get('color','')} {data.get('sex','')}"
        if data.get("wearable") == "top wearable":
            tops.append({"filename": filename, "text": text, "data": data})
        elif data.get("wearable") == "bottom wearable":
            bottoms.append({"filename": filename, "text": text, "data": data})
    auto_matches = []
    
    # Initialize SBERT model for semantic matching
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    
    for top in tops:
        top_emb = sbert_model.encode(top["text"], convert_to_tensor=True)
        best_bottom, best_score = None, -1
        for bottom in bottoms:
            bottom_emb = sbert_model.encode(bottom["text"], convert_to_tensor=True)
            score = util.cos_sim(top_emb, bottom_emb).item()
            if score > best_score:
                best_score = score
                best_bottom = bottom
        if best_bottom:
            # Generate a dynamic reason here as well.
            reason_text = generate_dynamic_reason(top["data"], best_bottom["data"], best_score)
            auto_matches.append({
                "top": top,
                "bottom": best_bottom,
                "score": best_score,
                "reason": reason_text
            })
    return jsonify({"auto_matches": auto_matches}), 200

@app.route("/delete_match", methods=["POST"])
def delete_match():
    index = request.form.get("index")
    if index is None:
        return jsonify({"error": "No match index provided."}), 400
    try:
        index = int(index)
    except ValueError:
        return jsonify({"error": "Invalid index."}), 400
    matches = load_matches()
    if index < 0 or index >= len(matches):
        return jsonify({"error": "Index out of range."}), 400
    del matches[index]
    save_matches(matches)
    return jsonify({"message": "Match deleted successfully."}), 200

@app.route("/delete_all_matches", methods=["POST"])
def delete_all_matches():
    save_matches([])
    return jsonify({"message": "All matches deleted successfully."}), 200

# Add a new route to validate an image
@app.route("/validate_image/<filename>")
def validate_image(filename):
    # Secure the filename
    secure_name = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
    
    if not os.path.exists(file_path):
        return jsonify({"valid": False, "error": "File not found"})
    
    try:
        with Image.open(file_path) as img:
            img.verify()
            # Get basic image info
            return jsonify({
                "valid": True, 
                "format": img.format,
                "size": os.path.getsize(file_path)
            })
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})

# Add new route for pose rigging visualization
@app.route("/pose_rigging")
def pose_rigging():
    all_images = get_all_images()
    return render_template('pose_visualization.html', images=all_images)

def get_all_images():
    """Retrieve all images from the database or storage.
    This function serves as a central point for image retrieval across the application.
    Returns a list of image objects/dictionaries with properties like filename, label, wearable, etc.
    """
    # First, try to use the labels.json file which has the correct metadata
    labels = load_labels()
    
    # Get the list of actual files in the uploads directory
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    files = os.listdir(upload_dir) if os.path.exists(upload_dir) else []
    
    images = []
    for filename in files:
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            continue
            
        # Get metadata from labels.json if available
        entry = labels.get(filename, {})
        
        # Check if file exists and is a valid image
        file_path = os.path.join(upload_dir, filename)
        is_valid_image = os.path.exists(file_path)
        
        if is_valid_image:
            try:
                # Verify the image is valid
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                is_valid_image = False
        
        # Get or determine label
        label = entry.get("label", os.path.splitext(filename)[0])
        
        # Get wearable type
        wearable = entry.get("wearable", "Unknown")
        
        # SPECIAL HANDLING FOR DRESS-LIKE ITEMS:
        # Check if this item should be classified as a dress regardless of its current classification
        label_lower = label.lower()
        
        # These are specific items that should be classified as dresses
        dress_items = ['dress', 'gown', 'sarong', 'frock', 'skirt', 'lehenga', 'saree', 'sari']
        
        # For debugging: Mark all clothing items that match our dress keywords in the metadata
        # This will help us identify these items in the UI
        entry_dress_match = any(item in label_lower for item in dress_items)
        
        # Override wearable type for items that should be classified as dresses
        if entry_dress_match:
            # Set metadata to indicate this is a dress-like item
            wearable = "dress"
        
        # Add the image data - with proper dress identification
        images.append({
            "filename": filename,
            "label": label,
            "wearable": wearable,
            "costume": entry.get("costume", "Unknown"),
            "color": entry.get("color", "#cccccc"),
            "pattern": entry.get("pattern", "Unknown"),
            "sex": entry.get("sex", "Unknown"),
            "hand": entry.get("hand", "Unknown"),
            "is_valid": is_valid_image,
            "is_dress_item": entry_dress_match  # Add this flag to easily identify dress items in the template
        })
    
    return images

@app.route("/analyze_colors/<filename>")
def analyze_image_colors(filename):
    """Analyze an image to extract dominant colors."""
    # Secure the filename
    secure_name = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        # Run color analysis
        color_info = analyze_colors(file_path)
        return jsonify({
            "filename": filename,
            "color_analysis": color_info
        })
    except Exception as e:
        print(f"Error analyzing colors in {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# Modify the if __name__ == '__main__' block to use a different port
if __name__ == '__main__':
    show_progress("Starting server", 100, "Server initialized", final=True)
    print("Fashion application running with pose rigging feature at: http://127.0.0.1:8080/")
    print("Access pose rigging visualization at: http://127.0.0.1:8080/pose_rigging")
    app.run(debug=True, threaded=True, port=8080)  # Changed port from 5000 to 8080