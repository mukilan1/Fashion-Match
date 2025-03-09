from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from model import classify_cloth  # Our clothing classifier
import json
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import re
from hand_classifier import predict_hand_type  # Our hand classifier
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline  # NEW import
import requests  # NEW import for local API call
import ollama  # ensure ollama is imported
import shutil  # Add this import for file operations

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

LABELS_FILE = "labels.json"
MATCHES_FILE = "matches.json"

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

# Determine if a garment is top or bottom wearable.
def determine_wearable_type(label):
    label_lower = label.lower()
    if any(kw in label_lower for kw in ["shirt", "suit", "jacket", "blouse", "sweater", "t-shirt", "vest"]):
        return "top wearable"
    if any(kw in label_lower for kw in ["jean", "trouser", "shorts", "skirt", "trunks"]):  # Added 'trunks'
        return "bottom wearable"
    return "top wearable"  # Default if unclear

def determine_costume_type(label):
    label_lower = label.lower()
    if any(word in label_lower for word in ["suit", "tux", "formal"]):
        return "formal"
    if any(word in label_lower for word in ["party", "dress", "gown"]):
        return "party"
    return "casual"

def determine_pattern_type(label):
    label_lower = label.lower()
    patterns = [
        (r'\bpolka\s*dot\b', "polka dot"),
        (r'\bstriped\b', "striped"),
        (r'\bchecked\b', "checked"),
        (r'\bplaid\b', "plaid"),
        (r'\bfloral\b', "floral"),
        (r'\bpaisley\b', "paisley"),
        (r'\bcamouflage\b', "camouflage"),
        (r'\bgeometric\b', "geometric"),
        (r'\bgraphic\b', "graphic"),
        (r'\bprint(ed)?\b', "printed"),
        (r'\bdotted\b', "dot pattern"),
        (r'\bdot\b', "dot pattern")
    ]
    for regex, pattern in patterns:
        if re.search(regex, label_lower):
            return pattern
    return "plain"

def extract_dominant_color(image_path):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    # Crop center region to avoid background
    img_cropped = img.crop((width*0.25, height*0.25, width*0.75, height*0.75)).resize((150,150))
    ar = np.array(img_cropped).reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(ar)
    counts = np.bincount(kmeans.labels_)
    dominant_color = tuple(int(x) for x in kmeans.cluster_centers_[np.argmax(counts)])
    
    basic_colors = {
        'red': (255, 0, 0),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'grey': (128, 128, 128),
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    def closest_color(requested):
        best, min_dist = None, float('inf')
        for name, rgb in basic_colors.items():
            dist = sum((requested[i]-rgb[i])**2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                best = name
        return best
    return closest_color(dominant_color)

def determine_sex(label):
    label_lower = label.lower()
    if any(word in label_lower for word in ["dress", "gown", "skirt", "blouse", "heels"]):
        return "female"
    if any(word in label_lower for word in ["suit", "tux", "man", "mens", "trouser", "male"]):
        return "male"
    return "unisex"

def determine_hand_style(label, image_path):
    try:
        return predict_hand_type(image_path)
    except Exception as e:
        label_lower = label.lower()
        if any(word in label_lower for word in ["sleeveless", "tank"]):
            return "no hand"
        elif any(word in label_lower for word in ["half sleeve", "short sleeve"]):
            return "half hand"
        elif any(word in label_lower for word in ["full sleeve", "long sleeve"]):
            return "full hand"
        return "undetermined"

# Initialize SBERT model for semantic matching.
sbert_model = SentenceTransformer('all-mpnet-base-v2')
# Initialize text-generation pipeline with a lightweight pretrained model.
reason_generator = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=50)  # UPDATED

# Set REASON_MODEL to either "deepseek-r1" or "llama3.2"
REASON_MODEL = "deepseek-r1"  # or "llama3.2" as needed

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
    
    for file in files:
        if file.filename == "":
            continue
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file
            file.save(file_path)
            
            # Process the image
            label = classify_cloth(file_path)
            wearable = determine_wearable_type(label)
            color = extract_dominant_color(file_path)
            costume = determine_costume_type(label)
            pattern = determine_pattern_type(label)
            sex = determine_sex(label)
            hand = determine_hand_style(label, file_path) if wearable == "top wearable" else "N/A"
            
            # Store in labels
            labels[filename] = {
                "label": label,
                "wearable": wearable,
                "costume": costume,
                "color": color,
                "pattern": pattern,
                "sex": sex,
                "hand": hand
            }
            
            results.append({
                "filename": filename,
                "label": label,
                "wearable": wearable,
                "costume": costume,
                "color": color,
                "pattern": pattern,
                "sex": sex,
                "hand": hand,
                "path": file_path  # Include the full path for debugging
            })
        except Exception as e:
            # Continue with other files if one fails
            print(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename if file.filename else "unknown",
                "error": str(e)
            })
    
    # Save the updated labels
    try:
        save_labels(labels)
    except Exception as e:
        return jsonify({"error": f"Failed to save labels: {str(e)}", "partial_results": results}), 500
        
    return jsonify({"uploaded": results}), 200

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
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    new_label = classify_cloth(file_path)
    new_wearable = determine_wearable_type(new_label)
    new_color = extract_dominant_color(file_path)
    new_costume = determine_costume_type(new_label)
    new_pattern = determine_pattern_type(new_label)
    new_sex = determine_sex(new_label)
    new_hand = determine_hand_style(new_label, file_path) if new_wearable=="top wearable" else "N/A"
    
    new_text = f"{new_label} {new_costume} {new_pattern} {new_color} {new_sex} {new_hand}"
    stored = load_labels()
    candidates = []
    for fname, data in stored.items():
        if new_wearable == "top wearable" and data.get("wearable") != "bottom wearable":
            continue
        if new_wearable == "bottom wearable" and data.get("wearable") != "top wearable":
            continue
        cand_text = f"{data.get('label','')} {data.get('costume','')} {data.get('pattern','')} {data.get('color','')} {data.get('sex','')} {data.get('hand','')}"
        candidates.append((fname, cand_text, data))
    if not candidates:
        return jsonify({"error": "No candidate found for matching."}), 404
    
    new_embedding = sbert_model.encode(new_text, convert_to_tensor=True)
    best_candidate, best_score = None, -1
    for fname, cand_text, data in candidates:
        cand_embedding = sbert_model.encode(cand_text, convert_to_tensor=True)
        score = util.cos_sim(new_embedding, cand_embedding).item()
        if score > best_score:
            best_score = score
            best_candidate = {"filename": fname, "data": data, "score": best_score}
    
    result = {
        "new_item": {
            "filename": filename,
            "label": new_label,
            "wearable": new_wearable,
            "color": new_color,
            "costume": new_costume,
            "pattern": new_pattern,
            "sex": new_sex,
            "hand": new_hand
        },
        "best_match": best_candidate
    }
    
    # Instead of a fixed reason, generate a dynamic reason using our NLP model.
    reason_text = generate_dynamic_reason(result["new_item"], best_candidate["data"], best_score)
    result["reason"] = reason_text
    matches = load_matches()
    matches.append(result)
    save_matches(matches)
    
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

# Update the get_all_images function to properly use the labels.json data

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

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# Modify the if __name__ == '__main__' block if needed
if __name__ == '__main__':
    print("Fashion application running with pose rigging feature at: http://127.0.0.1:5000/")
    print("Access pose rigging visualization at: http://127.0.0.1:5000/pose_rigging")
    app.run(debug=True, threaded=True)