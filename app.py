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

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
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
            "costume": entry.get("costume", "unknown"),
            "color": entry.get("color", "unknown"),
            "pattern": entry.get("pattern", "unknown"),
            "sex": entry.get("sex", "unknown"),
            "hand": entry.get("hand", "unknown")
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
        hand = determine_hand_style(label, file_path) if wearable == "top wearable" else "N/A"
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
            "hand": hand
        })
    save_labels(labels)
    return jsonify({"uploaded": results}), 200

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
    labels = load_labels()
    files = os.listdir(app.config["UPLOAD_FOLDER"]) if os.path.exists(app.config["UPLOAD_FOLDER"]) else []
    images_data = []
    for f in files:
        entry = labels.get(f, {})
        images_data.append({
            "filename": f,
            "label": entry.get("label", "unknown"),
            "wearable": entry.get("wearable", "unknown"),
            "costume": entry.get("costume", "unknown"),
            "color": entry.get("color", "unknown"),
            "pattern": entry.get("pattern", "unknown"),
            "sex": entry.get("sex", "unknown"),
            "hand": entry.get("hand", "unknown")
        })
    return render_template("match.html", images=images_data)

# New: Handle match upload: compare the uploaded item with the stored items for the best pair.
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
    
    # Save this match record in matches.json
    matches = load_matches()
    matches.append(result)
    save_matches(matches)
    
    return jsonify({"matched": result}), 200

@app.route("/matches", methods=["GET"])
def get_matches():
    matches = load_matches()
    return jsonify({"matches": matches}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
