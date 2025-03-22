"""
Match-related route handlers
"""
import os
import shutil
from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER
from utils.progress import show_progress
from utils.file_utils import load_labels, save_labels, load_matches, save_matches, synchronize_labels, synchronize_matches
from utils.image_utils import get_all_images
from services.background_service import remove_background
from services.matching_service import find_best_match, auto_match, generate_match_reason

# Import analysis modules
from Model_Props.image_analyzer import ClothingImageAnalyzer
from Model_Props.color_analyzer import analyze_colors
from Model_Props.gender_analyzer import analyze_gender
from Model_Props.costume_analyzer import analyze_costume
from Model_Props.sleeve_analyzer import analyze_sleeve
from Model_Props.pattern_analyzer import analyze_pattern

# Initialize the image analyzer
image_analyzer = ClothingImageAnalyzer(use_vqa=True, use_ml_model=True)

def register_match_routes(app):
    @app.route("/match", methods=["GET"])
    def match_page():
        # Synchronize both labels and matches
        synchronize_labels()
        synchronize_matches()
        
        # Get images for display
        labels = load_labels()
        images_data = get_all_images(labels)
        
        return render_template("match.html", images=images_data)

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
        show_progress("Match processing", 15, "File saved")
        
        # Remove background from image
        try:
            show_progress("Match processing", 20, "Removing background")
            processed_image = remove_background(temp_path)
            
            # Save processed image (with transparent background)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            processed_image.save(file_path, format="PNG")
            show_progress("Match processing", 25, "Background removed")
        except Exception as e:
            print(f"Background removal failed for {filename}: {str(e)}")
            # If background removal fails, use original image
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            shutil.copy(temp_path, file_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # PERFORM FULL IMAGE ANALYSIS PIPELINE - same as upload endpoint
        show_progress("Match processing", 30, "Analyzing clothing item")
        
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
        
        # Perform complete analysis similar to upload endpoint
        # 2. Color analysis
        show_progress("Match processing", 35, "Analyzing colors")
        try:
            color_info = analyze_colors(file_path)
            primary_color = color_info['primary_color']
            colors_text = color_info['colors_text']
        except Exception as e:
            print(f"Color analysis failed: {str(e)}")
            primary_color = "unknown"
            colors_text = "unknown"
        
        # 3. Gender analysis
        show_progress("Match processing", 40, "Determining gender")
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
        except Exception as e:
            print(f"Gender analysis failed: {str(e)}")
            gender = "unknown"
            gender_confidence = 0
            gender_probabilities = {"men": 0.33, "women": 0.33, "unisex": 0.34}
        
        # 4. Costume analysis
        show_progress("Match processing", 45, "Determining costume style")
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
        except Exception as e:
            print(f"Costume analysis failed: {str(e)}")
            costume = "unknown"
            costume_display_name = "Unknown"
            costume_confidence = 0
            costume_description = ""
        
        # 5. Pattern detection
        show_progress("Match processing", 50, "Detecting pattern type")
        pattern_metadata = {
            "label": label,
            "wearable": wearable_position,
            "color": primary_color
        }
        try:
            pattern_info = analyze_pattern(file_path, pattern_metadata)
            pattern = pattern_info.get("pattern_display", "unknown")
            pattern_confidence = pattern_info.get("confidence", 0)
        except Exception as e:
            print(f"Pattern analysis failed: {str(e)}")
            pattern = "unknown"
            pattern_confidence = 0
        
        # 6. Sleeve/hand type detection
        show_progress("Match processing", 55, "Detecting sleeve length")
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
        except Exception as e:
            print(f"Sleeve analysis failed: {str(e)}")
            hand = "unknown"
        
        # Store metadata for the newly uploaded item
        labels = load_labels()
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
        save_labels(labels)
        
        show_progress("Match processing", 60, "Full analysis completed")
        
        # Find best matching item
        new_item = {
            "filename": filename,
            "label": label,
            "wearable": wearable_position,
            "costume": costume,
            "color": primary_color,
            "pattern": pattern,
            "sex": gender,
            "hand": hand
        }
        
        show_progress("Match processing", 70, "Finding best fashion match")
        
        try:
            best_candidate, error_msg = find_best_match(new_item, labels)
            
            if error_msg:
                return jsonify({"error": error_msg}), 404
                
            show_progress("Match processing", 80, "Perfect match found")
            
            # Generate a reason for the match
            show_progress("Match processing", 90, "Generating fashion advice")
            reason = generate_match_reason(new_item, best_candidate["data"], best_candidate["score"])
            
            # Create the match result
            result = {
                "new_item": new_item,
                "best_match": best_candidate,
                "reason": reason
            }
            
            # Save match record
            matches = load_matches()
            matches.append(result)
            save_matches(matches)
            
            show_progress("Match processing", 100, "Outfit match completed", final=True)
            
            return jsonify({"matched": result}), 200
            
        except Exception as e:
            show_progress("Match processing", 100, f"Error in matching: {str(e)}", final=True)
            return jsonify({"error": f"Match processing failed: {str(e)}"}), 500

    @app.route("/matches", methods=["GET"])
    def match_get_matches():
        matches = load_matches()
        return jsonify({"matches": matches}), 200

    @app.route("/auto_match", methods=["GET"])
    def match_auto_match():
        labels = load_labels()
        auto_matches = auto_match(labels)
        return jsonify({"auto_matches": auto_matches}), 200

    @app.route("/delete_match", methods=["POST"])
    def match_delete_match():
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
    def match_delete_all_matches():
        save_matches([])
        return jsonify({"message": "All matches deleted successfully."}), 200
