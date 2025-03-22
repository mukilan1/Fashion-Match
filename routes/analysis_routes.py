"""
Analysis-related route handlers
"""
import os
from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER
from utils.file_utils import load_labels, save_labels
from utils.image_utils import validate_image
from utils.structure_analyzer import StructureDetector

# Import analysis modules
from Model_Props.color_analyzer import analyze_colors
from Model_Props.gender_analyzer import analyze_gender
from Model_Props.costume_analyzer import analyze_costume
from Model_Props.sleeve_analyzer import analyze_sleeve
from Model_Props.pattern_analyzer import analyze_pattern

# Create a singleton instance of the structure detector
structure_detector = StructureDetector()

def register_analysis_routes(app):
    @app.route("/analyze_colors/<filename>")
    def analysis_colors_route(filename):
        """Analyze an image to extract dominant colors."""
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
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

    @app.route("/analyze_gender/<filename>", methods=["POST"])
    def analysis_gender_route(filename):
        """
        Internal API endpoint to re-analyze gender for an image.
        This endpoint is not exposed in the UI but available for programmatic use.
        """
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Get existing metadata
            labels = load_labels()
            metadata = labels.get(secure_name, {})
            
            # Run gender analysis
            gender_info = analyze_gender(file_path, metadata)
            
            # Update labels with the gender information
            if secure_name in labels:
                labels[secure_name]['sex'] = gender_info['gender']
                labels[secure_name]['gender_confidence'] = gender_info['confidence']
                save_labels(labels)
            
            return jsonify({
                "filename": filename,
                "gender_analysis": gender_info,
                "updated": True
            })
        except Exception as e:
            print(f"Error analyzing gender in {filename}: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/analyze_costume/<filename>", methods=["POST"])
    def analysis_costume_route(filename):
        """Analyze an image to determine its costume/outfit style."""
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Get existing metadata to help with analysis
            labels = load_labels()
            metadata = labels.get(secure_name, {})
            
            # Run costume analysis
            costume_info = analyze_costume(file_path, metadata)
            
            # Update labels with the costume information
            if secure_name in labels:
                labels[secure_name]['costume'] = costume_info['costume']
                labels[secure_name]['costume_display'] = costume_info['costume_display_name']
                labels[secure_name]['costume_confidence'] = costume_info['confidence']
                labels[secure_name]['costume_description'] = costume_info.get('description', '')
                save_labels(labels)
            
            return jsonify({
                "filename": filename,
                "costume_analysis": costume_info,
                "updated": True
            })
        except Exception as e:
            print(f"Error analyzing costume in {filename}: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/analyze_sleeve/<filename>", methods=["POST"])
    def analysis_sleeve_route(filename):
        """
        Analyze sleeve length for an image and update metadata.
        This will determine if an item has full, half, or no sleeves.
        """
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Get existing metadata
            labels = load_labels()
            metadata = labels.get(secure_name, {})
            
            # Run sleeve analysis
            sleeve_info = analyze_sleeve(file_path, metadata)
            
            # Only update if this isn't bottom wear
            if not sleeve_info.get('is_bottom_wear', False):
                # Update labels with the sleeve information
                if secure_name in labels:
                    labels[secure_name]['hand'] = sleeve_info['sleeve_display']
                    save_labels(labels)
                    
                return jsonify({
                    "filename": filename,
                    "sleeve_analysis": sleeve_info,
                    "updated": True
                })
            else:
                return jsonify({
                    "filename": filename,
                    "sleeve_analysis": sleeve_info,
                    "updated": False,
                    "message": "Sleeve analysis not applicable for bottom wear"
                })
        except Exception as e:
            print(f"Error analyzing sleeve in {filename}: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/analyze_pattern/<filename>", methods=["POST"])
    def analysis_pattern_route(filename):
        """
        Analyze pattern type for an image and update metadata.
        This will determine if the clothing has solid, striped, checkered, etc. pattern.
        """
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Get existing metadata
            labels = load_labels()
            metadata = labels.get(secure_name, {})
            
            # Run pattern analysis
            pattern_info = analyze_pattern(file_path, metadata)
            
            # Update labels with the pattern information
            if secure_name in labels:
                labels[secure_name]['pattern'] = pattern_info['pattern_display']
                labels[secure_name]['pattern_confidence'] = pattern_info['confidence']
                save_labels(labels)
                
            return jsonify({
                "filename": filename,
                "pattern_analysis": pattern_info,
                "updated": True
            })
        except Exception as e:
            print(f"Error analyzing pattern in {filename}: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/analyze_structure/<filename>")
    def analysis_structure_route(filename):
        """Analyze the structure of a clothing item and return key points."""
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Get clothing metadata from database
            labels = load_labels()
            metadata = labels.get(secure_name, {})
            clothing_type = metadata.get("wearable", "unknown")
            sleeve_type = metadata.get("hand", "unknown")
            
            # Detect structure based on image analysis
            structure_points = structure_detector.detect_structure(
                file_path, clothing_type, sleeve_type
            )
            
            # Cache the structure points for future use
            if secure_name in labels:
                labels[secure_name]['structure_points'] = structure_points
                save_labels(labels)
            
            return jsonify({
                "filename": filename,
                "clothing_type": clothing_type,
                "sleeve_type": sleeve_type,
                "structure_points": structure_points
            })
        except Exception as e:
            print(f"Error analyzing structure in {filename}: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Change this function name to avoid the conflict with main_routes.py
    @app.route("/analysis/validate_image/<filename>")
    def analysis_validate_image_route(filename):
        # Secure the filename
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
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
