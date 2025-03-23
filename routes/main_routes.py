"""
Main route handlers for the Fashion Matching application
"""
import os
import shutil
from flask import render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, DEFAULT_IMG_DIR
from utils.progress import show_progress
from utils.file_utils import load_labels, save_labels, synchronize_labels
from utils.image_utils import validate_image, get_all_images
from services.background_service import remove_background

# Import analysis modules
from Model_Props.image_analyzer import ClothingImageAnalyzer
from Model_Props.color_analyzer import analyze_colors
from Model_Props.gender_analyzer import analyze_gender
from Model_Props.costume_analyzer import analyze_costume
from Model_Props.sleeve_analyzer import analyze_sleeve
from Model_Props.pattern_analyzer import analyze_pattern

# Initialize the image analyzer
image_analyzer = ClothingImageAnalyzer(use_vqa=True, use_ml_model=True)

def register_main_routes(app):
    @app.route("/", endpoint="index")  # Add original endpoint name as alias
    @app.route("/", endpoint="main_index")
    def main_index():
        # Synchronize the database with actual files
        synchronize_labels()
        
        # Get all images with their metadata
        labels = load_labels()
        images_data = get_all_images(labels)
        
        return render_template("index.html", images=images_data)

    @app.route("/upload", methods=["POST"])
    def main_upload():
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
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 5, "File saved")
                
                # Remove background from image
                try:
                    show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 10, "Removing background")
                    processed_image = remove_background(temp_path)
                    
                    # Save processed image (with transparent background)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Save with transparent background
                    processed_image.save(file_path, format="PNG")
                    show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 20, "Background removed")
                    
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
                
                # COMBINED ANALYSIS PIPELINE
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 30, "Analyzing clothing item")
                
                # 1. Image classification analysis
                analysis_results = image_analyzer.analyze_image(file_path)
                label = analysis_results.get("classification", "unknown")
                confidence = analysis_results.get("confidence", 0)
                wearable_position = analysis_results.get("wearable_position", "unknown")
                
                # Default wearable position as fallback
                if wearable_position == "unknown":
                    if "shirt" in label.lower() or "top" in label.lower() or "jacket" in label.lower():
                        wearable_position = "top wearable"
                    elif "pants" in label.lower() or "jeans" in label.lower() or "skirt" in label.lower():
                        wearable_position = "bottom wearable"
                    else:
                        wearable_position = "top wearable"
                
                # 2. Color analysis
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 40, "Analyzing colors")
                try:
                    color_info = analyze_colors(file_path)
                    primary_color = color_info['primary_color']
                    colors_text = color_info['colors_text']
                except Exception as e:
                    print(f"Color analysis failed: {str(e)}")
                    primary_color = "unknown"
                    colors_text = "unknown"
                
                # 3. Gender analysis
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 50, "Determining gender")
                metadata = {
                    "label": label,
                    "wearable": wearable_position,
                    "color": primary_color
                }
                try:
                    gender_info = analyze_gender(file_path, metadata)
                    gender = gender_info.get("gender", "unknown")
                    gender_confidence = gender_info.get("confidence", 0)
                    gender_probabilities = gender_info.get("probabilities", {})
                    print(f"Detected gender: {gender} (confidence: {gender_confidence:.2f})")
                except Exception as e:
                    print(f"Gender analysis failed: {str(e)}")
                    gender = "unknown"
                    gender_confidence = 0
                    gender_probabilities = {"men": 0.33, "women": 0.33, "unisex": 0.34}
                    
                # 4. Costume analysis
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 60, "Determining costume style")
                metadata.update({
                    "gender": gender,
                    "label": label,
                    "color": primary_color,
                    "pattern": "unknown"
                })
                try:
                    costume_info = analyze_costume(file_path, metadata)
                    costume = costume_info.get("costume", "unknown")
                    costume_display_name = costume_info.get("costume_display_name", "Unknown")
                    costume_confidence = costume_info.get("confidence", 0)
                    costume_description = costume_info.get("description", "")
                    print(f"Detected costume: {costume_display_name} (confidence: {costume_confidence:.2f})")
                except Exception as e:
                    print(f"Costume analysis failed: {str(e)}")
                    costume = "unknown"
                    costume_display_name = "Unknown"
                    costume_confidence = 0
                    costume_description = ""
                
                # 5. Pattern detection
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 65, "Detecting pattern type")
                pattern_metadata = {
                    "label": label,
                    "wearable": wearable_position,
                    "color": primary_color
                }
                
                try:
                    pattern_info = analyze_pattern(file_path, pattern_metadata)
                    pattern = pattern_info.get("pattern_display", "unknown")
                    pattern_confidence = pattern_info.get("confidence", 0)
                    print(f"Detected pattern: {pattern} (confidence: {pattern_confidence:.2f})")
                except Exception as e:
                    print(f"Pattern analysis failed: {str(e)}")
                    pattern = "unknown"
                    pattern_confidence = 0
                
                # 6. Sleeve/hand type detection
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 70, "Detecting sleeve length")
                current_metadata = {
                    "label": label,
                    "wearable": wearable_position,
                    "color": primary_color
                }
                try:
                    sleeve_info = analyze_sleeve(file_path, current_metadata)
                    hand = sleeve_info.get("sleeve_display", "unknown")
                    # Only use the result if it's not bottom wear
                    if sleeve_info.get("is_bottom_wear", False):
                        hand = "N/A (Bottom Wear)"
                    print(f"Detected sleeve length: {hand}")
                except Exception as e:
                    print(f"Sleeve analysis failed: {str(e)}")
                    hand = "unknown"
                
                show_progress(f"Processing file {i+1}/{total_files}", (i / total_files) * 30 + 75, "All analysis completed")
                
                # Store all metadata together
                labels[filename] = {
                    "label": label,
                    "wearable": wearable_position,
                    "costume": costume, 
                    "costume_display": costume_display_name,
                    "costume_confidence": costume_confidence,
                    "costume_description": costume_description,
                    "color": primary_color,
                    "color_detail": colors_text,
                    "pattern": pattern,
                    "pattern_confidence": pattern_confidence,
                    "sex": gender,
                    "gender_confidence": gender_confidence,
                    "hand": hand
                }
                
                # Return comprehensive analysis results
                results.append({
                    "filename": filename,
                    "label": label,
                    "wearable": wearable_position,
                    "costume": costume,
                    "costume_display": costume_display_name,
                    "costume_confidence": costume_confidence,
                    "costume_description": costume_description,
                    "color": primary_color,
                    "color_detail": colors_text,
                    "pattern": pattern,
                    "pattern_confidence": pattern_confidence,
                    "sex": gender,
                    "gender_confidence": gender_confidence,
                    "gender_probabilities": gender_probabilities,
                    "hand": hand,
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
        
        return jsonify({
            "uploaded": results,
            "progress_complete": True
        }), 200

    @app.route("/uploads/<filename>", endpoint="uploaded_file")
    @app.route("/uploads/<filename>", endpoint="main_uploaded_file")
    def main_uploaded_file(filename):
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
        is_valid, _ = validate_image(file_path)
        if not is_valid:
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
    def main_delete_image():
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
    def main_delete_all():
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

    @app.route("/validate_image/<filename>")
    def main_validate_image(filename):
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)
        
        if not os.path.exists(file_path):
            return jsonify({"valid": False, "error": "File not found"})
        
        is_valid, error_message = validate_image(file_path)
        if is_valid:
            return jsonify({
                "valid": True, 
                "format": "image",
                "size": os.path.getsize(file_path)
            })
        else:
            return jsonify({"valid": False, "error": error_message})

    @app.route('/navigation')
    def navigation():
        """Return navigation data for frontend"""
        nav_items = [
            {"name": "Home", "url": "/", "icon": "home"},
            {"name": "Match", "url": "/match", "icon": "tshirt"},
            {"name": "Pose Rigging", "url": "/pose_rigging", "icon": "user"},
            {"name": "Dress Search", "url": "/dress_search", "icon": "search"},  # Added dress search
            # ...existing items...
        ]
        return jsonify({"navigation": nav_items})

    @app.errorhandler(404)
    def main_page_not_found(e):
        return render_template("404.html"), 404
