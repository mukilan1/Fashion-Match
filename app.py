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
from hand_classifier import predict_hand_type  # New import for hand prediction
import math

# Add precompiled regex patterns at module level
PATTERN_REGEXES = [
    (re.compile(r'\bpolka\s*dot\b'), "polka dot"),
    (re.compile(r'\bstriped\b'), "striped"),
    (re.compile(r'\bchecked\b'), "checked"),
    (re.compile(r'\bplaid\b'), "plaid"),
    (re.compile(r'\bfloral\b'), "floral"),
    (re.compile(r'\bpaisley\b'), "paisley"),
    (re.compile(r'\bcamouflage\b'), "camouflage"),
    (re.compile(r'\bgeometric\b'), "geometric"),
    (re.compile(r'\bgraphic\b'), "graphic"),
    (re.compile(r'\bprint(?:ed)?\b'), "printed"),
    (re.compile(r'\bdotted\b'), "dot pattern"),
    (re.compile(r'\bdot\b'), "dot pattern")
]

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
    bottom_keywords = ["jean", "trouser", "shorts", "skirt", "pant", "track"]
    top_keywords = ["shirt", "jacket", "blouse", "sweater", "t-shirt", "vest", "suit"]
    
    # First, if any bottom keyword is present, classify as bottom wearable.
    if any(kw in label_lower for kw in bottom_keywords):
        return "bottom wearable"
    
    # If label contains "suit" possibly from side view of pants, check for override
    if "suit" in label_lower:
        # If ambiguous, check if any bottom keyword is loosely present
        if any(kw in label_lower for kw in ["pant", "track"]):
            return "bottom wearable"
        return "top wearable"
    
    # Then classify as top if top keyword exists.
    if any(kw in label_lower for kw in top_keywords):
        return "top wearable"
    
    return "unknown"

# Determine costume type from the predicted label, defaulting to "casual"
def determine_costume_type(label):
    label_lower = label.lower()
    if any(word in label_lower for word in ["suit", "tux", "formal"]):
        return "formal"
    if any(word in label_lower for word in ["party", "dress", "gown"]):
        return "party"
    return "casual"

# Updated: Determine pattern type using precompiled regex for efficiency
def determine_pattern_type(label):
    label_lower = label.lower()
    for pattern_re, pattern in PATTERN_REGEXES:
        if pattern_re.search(label_lower):
            return pattern
    return "plain"

def rgb_to_lab(rgb):
    # Convert RGB (0-255) to LAB using D65 white reference
    r, g, b = [x / 255.0 for x in rgb]
    r = r/12.92 if r <= 0.04045 else ((r+0.055)/1.055)**2.4
    g = g/12.92 if g <= 0.04045 else ((g+0.055)/1.055)**2.4
    b = b/12.92 if b <= 0.04045 else ((b+0.055)/1.055)**2.4
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    X /= 0.95047; Y /= 1.00000; Z /= 1.08883
    def f(t):
        delta = 6/29
        return t**(1/3) if t > delta**3 else t/(3*delta**2) + 4/29
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b_val = 200 * (f(Y) - f(Z))
    return (L, a, b_val)

def rgb_to_hsl(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    l = (max_c + min_c) / 2
    if max_c == min_c:
        h = 0
        s = 0
    else:
        delta = max_c - min_c
        s = delta / (max_c + min_c) if l <= 0.5 else delta / (2 - max_c - min_c)
        if max_c == r:
            h = ((g - b) / delta) % 6
        elif max_c == g:
            h = ((b - r) / delta) + 2
        else:
            h = ((r - g) / delta) + 4
        h *= 60
    return h, s, l

# New: Precompute the color map and its LAB conversion once for efficiency.
COLOR_MAP = {
    'maroon': (128, 0, 0), 'dark red': (139, 0, 0), 'red': (255, 0, 0),
    'salmon': (250, 128, 114), 'coral': (255, 127, 80), 'crimson': (220, 20, 60),
    'tomato': (255, 99, 71), 'dark orange': (255, 140, 0), 'orange': (255, 165, 0),
    'orange red': (255, 69, 0), 'gold': (255, 215, 0), 'yellow': (255, 255, 0),
    'light yellow': (255, 255, 224), 'brown': (165, 42, 42), 'saddle brown': (139, 69, 19),
    'sienna': (160, 82, 45), 'chocolate': (210, 105, 30), 'peru': (205, 133, 63),
    'sandy brown': (244, 164, 96), 'tan': (210, 180, 140), 'dark green': (0, 100, 0),
    'green': (0, 128, 0), 'lime green': (50, 205, 50), 'lime': (0, 255, 0),
    'olive': (128, 128, 0), 'teal': (0, 128, 128), 'mint': (189, 252, 201),
    'forest green': (34, 139, 34), 'navy': (0, 0, 128), 'dark blue': (0, 0, 139),
    'medium blue': (0, 0, 205), 'blue': (0, 0, 255), 'sky blue': (135, 206, 235),
    'light blue': (173, 216, 230), 'aqua': (0, 255, 255), 'cyan': (0, 255, 255),
    'turquoise': (64, 224, 208), 'indigo': (75, 0, 130), 'purple': (128, 0, 128),
    'dark violet': (148, 0, 211), 'violet': (238, 130, 238), 'magenta': (255, 0, 255),
    'orchid': (218, 112, 214), 'medium orchid': (186, 85, 211), 'pink': (255, 192, 203),
    'hot pink': (255, 105, 180), 'deep pink': (255, 20, 147), 'black': (0, 0, 0),
    'dark gray': (64, 64, 64), 'gray': (128, 128, 128), 'silver': (192, 192, 192),
    'light gray': (211, 211, 211), 'white': (255, 255, 255), 'beige': (245, 245, 220),
    'ivory': (255, 255, 240)
}
LAB_COLOR_MAP = {name: rgb_to_lab(rgb) for name, rgb in COLOR_MAP.items()}

def refine_color(dominant_rgb, current_candidate, color_map):
    h, s, l = rgb_to_hsl(dominant_rgb)
    # Neutral refinement for low saturation
    if s < 0.15:
        if l > 0.85:
            return 'white'
        elif l < 0.15:
            return 'black'
        else:
            return 'gray'
    # Candidate groups for further refinement
    candidate_groups = {
         'red': ['maroon', 'dark red', 'red', 'salmon', 'coral', 'crimson', 'tomato'],
         'orange': ['dark orange', 'orange', 'orange red'],
         'yellow': ['gold', 'yellow', 'light yellow'],
         'brown': ['brown', 'saddle brown', 'sienna', 'chocolate', 'peru', 'sandy brown', 'tan'],
         'green': ['dark green', 'green', 'lime green', 'lime', 'olive', 'teal', 'mint', 'forest green'],
         'blue': ['navy', 'dark blue', 'medium blue', 'blue', 'sky blue', 'light blue', 'aqua', 'cyan', 'turquoise'],
         'purple': ['indigo', 'purple', 'dark violet', 'violet', 'magenta', 'orchid', 'medium orchid'],
         'pink': ['pink', 'hot pink', 'deep pink']
    }
    # Identify group of current candidate
    for group, names in candidate_groups.items():
        if current_candidate in names:
            best_candidate = current_candidate
            min_hue_diff = float('inf')
            for candidate in names:
                candidate_h, _, _ = rgb_to_hsl(color_map[candidate])
                diff = abs(h - candidate_h)
                if diff < min_hue_diff:
                    min_hue_diff = diff
                    best_candidate = candidate
            return best_candidate
    return current_candidate

# Updated extract_dominant_color with ensemble methods and global color maps for improved efficiency
def extract_dominant_color(image_path):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    img_np = np.array(img)
    
    # Compute border pixels to detect background and compute median background color.
    margin_w, margin_h = int(0.1*width), int(0.1*height)
    borders = np.concatenate((
        img_np[:margin_h].reshape(-1, 3),
        img_np[-margin_h:].reshape(-1, 3),
        img_np[:, :margin_w].reshape(-1, 3),
        img_np[:, -margin_w:].reshape(-1, 3)
    ), axis=0)
    background_color = np.median(borders, axis=0)
    
    # Exclude similar-to-background pixels.
    diff = np.linalg.norm(img_np - background_color, axis=2)
    mask = diff > 30
    fg_pixels = img_np[mask]
    if fg_pixels.size < 100:
        x1, y1, x2, y2 = int(0.25*width), int(0.25*height), int(0.75*width), int(0.75*height)
        fg_pixels = np.array(img.crop((x1, y1, x2, y2)).resize((150,150))).reshape(-1,3)
    
    # Method 1: KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(fg_pixels)
    kmeans_color = tuple(int(x) for x in kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))])
    
    # Method 2: Quantized palette mode
    quantized = img.quantize(colors=8)
    palette = quantized.getpalette()
    colors, counts = zip(*quantized.getcolors() or [(0,0)])
    if counts:
        mode_idx = counts.index(max(counts))
        quant_color = tuple(palette[max(mode_idx,0)*3:max(mode_idx,0)*3+3])
    else:
        quant_color = kmeans_color
    
    # Method 3: Histogram mode (by computing mode in each channel separately)
    hist_r, _ = np.histogram(fg_pixels[:,0], bins=256, range=(0,255))
    hist_g, _ = np.histogram(fg_pixels[:,1], bins=256, range=(0,255))
    hist_b, _ = np.histogram(fg_pixels[:,2], bins=256, range=(0,255))
    mode_color = (int(np.argmax(hist_r)), int(np.argmax(hist_g)), int(np.argmax(hist_b)))
    
    # Ensemble: average the RGB values from the three methods.
    ensemble_color = tuple(int((a + b + c) / 3) for a, b, c in zip(kmeans_color, quant_color, mode_color))
    
    # LAB matching and smart HSL refinement using global COLOR_MAP and LAB_COLOR_MAP.
    dominant_lab = rgb_to_lab(ensemble_color)
    best_name, min_dist = None, float('inf')
    for name, lab in LAB_COLOR_MAP.items():
        d = math.sqrt(sum((dominant_lab[i]-lab[i])**2 for i in range(3)))
        if d < min_dist:
            min_dist, best_name = d, name
    best_name = refine_color(ensemble_color, best_name, COLOR_MAP)
    return best_name

# New: Determine the sex of the dress based on the predicted label
def determine_sex(label):
    label_lower = label.lower()
    if any(word in label_lower for word in ["dress", "gown", "skirt", "blouse", "heels"]):
        return "female"
    if any(word in label_lower for word in ["suit", "tux", "man", "mens", "trouser", "male"]):
        return "male"
    return "unisex"

# New: Determine hand style for top wearable garments using the AI model with fallback
def determine_hand_style(label, image_path):
    try:
        # Prefer model prediction for hand style
        hand = predict_hand_type(image_path)
        return hand
    except Exception as e:
        # Fallback heuristic if AI model isn't available
        label_lower = label.lower()
        if any(word in label_lower for word in ["sleeveless", "tank"]):
            return "no hand"
        elif any(word in label_lower for word in ["half sleeve", "short sleeve"]):
            return "half hand"
        elif any(word in label_lower for word in ["full sleeve", "long sleeve"]):
            return "full hand"
        return "undetermined"

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
        # For top wearable, determine hand style; else set to "N/A"
        hand = determine_hand_style(label, file_path) if wearable == "top wearable" else "N/A"
        labels[filename] = {
            "label": label,
            "wearable": wearable,
            "color": color,
            "costume": costume,
            "pattern": pattern,
            "sex": sex,
            "hand": hand
        }
        results.append({
            "filename": filename,
            "label": label,
            "wearable": wearable,
            "color": color,
            "costume": costume,
            "pattern": pattern,
            "sex": sex,
            "hand": hand
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
