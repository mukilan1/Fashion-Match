from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from model import classify_cloth  # Using our classification function
import json
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import webcolors
import re  # New import for regex matching

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

LABELS_FILE = "labels.json"

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)

# Determine if a garment is top or bottom wearable
def determine_wearable_type(label):
    label_lower = label.lower()
    if any(kw in label_lower for kw in ["shirt", "suit", "jacket", "blouse", "sweater", "t-shirt", "vest"]):
        return "top wearable"
    if any(kw in label_lower for kw in ["jean", "trouser", "shorts", "skirt"]):
        return "bottom wearable"
    return "top wearable"  # Defaulting to a precise type if no keyword is matched

# Determine costume type from the predicted label, defaulting to "casual"
def determine_costume_type(label):
    label_lower = label.lower()
    if any(word in label_lower for word in ["suit", "tux", "formal"]):
        return "formal"
    if any(word in label_lower for word in ["party", "dress", "gown"]):
        return "party"
    return "casual"

# Updated: Determine pattern type from the predicted label using regex for accurate matching
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

# Extract dominant color from the center and map to a basic color name
def extract_dominant_color(image_path):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    # Crop center region to avoid background interference
    img_cropped = img.crop((width * 0.25, height * 0.25, width * 0.75, height * 0.75))
    img_cropped = img_cropped.resize((150, 150))
    ar = np.array(img_cropped).reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=0).fit(ar)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    dominant_color = tuple(int(x) for x in dominant_color)
    
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
    
    def closest_color(requested_color):
        min_distance = float('inf')
        closest_name = None
        for name, rgb in basic_colors.items():
            dist = sum((requested_color[i]-rgb[i])**2 for i in range(3))
            if dist < min_distance:
                min_distance = dist
                closest_name = name
        return closest_name
    
    return closest_color(dominant_color)

# New: Determine the sex of the dress based on the predicted label
def determine_sex(label):
    label_lower = label.lower()
    if any(word in label_lower for word in ["dress", "gown", "skirt", "blouse", "heels"]):
        return "female"
    if any(word in label_lower for word in ["suit", "tux", "man", "mens", "trouser", "male"]):
        return "male"
    return "unisex"

@app.route("/")
def index():
    labels = load_labels()
    files = os.listdir(app.config["UPLOAD_FOLDER"]) if os.path.exists(app.config["UPLOAD_FOLDER"]) else []
    images_data = []
    for f in files:
        entry = labels.get(f, {})
        images_data.append({
            "filename": f, 
            "label": entry.get("label", "unknown"), 
            "wearable": entry.get("wearable", "unknown"),
            "color": entry.get("color", "unknown"),
            "costume": entry.get("costume", "unknown"),
            "pattern": entry.get("pattern", "unknown"),
            "sex": entry.get("sex", "unknown")
        })
    return render_template("index.html", images=images_data)

@app.route("/upload", methods=["POST"])
def upload():
    if "images" not in request.files:
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist("images")
    labels = load_labels()
    results = []
    for file in files:
        if file.filename == "":
            continue
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        label = classify_cloth(file_path)
        wearable = determine_wearable_type(label)
        color = extract_dominant_color(file_path)
        costume = determine_costume_type(label)
        pattern = determine_pattern_type(label)
        sex = determine_sex(label)
        labels[filename] = {
            "label": label,
            "wearable": wearable,
            "color": color,
            "costume": costume,
            "pattern": pattern,
            "sex": sex
        }
        results.append({
            "filename": filename,
            "label": label,
            "wearable": wearable,
            "color": color,
            "costume": costume,
            "pattern": pattern,
            "sex": sex
        })
    save_labels(labels)
    return jsonify({"uploaded": results}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete', methods=["POST"])
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

@app.route('/delete_all', methods=["POST"])
def delete_all():
    # Delete all files from uploads folder and clear labels
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

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
