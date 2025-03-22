"""
Media visualization route handlers for pose rigging
"""
from flask import render_template, jsonify, send_file, request
from werkzeug.utils import secure_filename
import os
import io
from PIL import Image

from config import UPLOAD_FOLDER
from utils.file_utils import load_labels, synchronize_labels, save_labels
from utils.image_utils import get_all_images
# Import from our newly created modules
from utils.structure_analyzer import StructureDetector
from utils.rigging_utils import (
    apply_rigging, is_rigged_image_cached, get_rigged_image_path, 
    detect_precise_structure, draw_top_skeleton, draw_bottom_skeleton, draw_full_skeleton,
    generate_rigged_image
)
from utils.garment_analyzer import validate_sleeve_type

def register_media_routes(app):
    @app.route("/pose_rigging", endpoint="pose_rigging")
    def media_pose_rigging():
        """Display the pose rigging visualization page"""
        synchronize_labels()
        labels = load_labels()
        all_images = get_all_images(labels)
        return render_template('pose_visualization.html', images=all_images)

    @app.route("/rigged_images/<filename>")
    def serve_rigged_image(filename):
        """Serve the pre-rendered rigged image"""
        rigged_path = get_rigged_image_path(filename)
        
        # Check if the rigged image exists
        if os.path.exists(rigged_path):
            return send_file(rigged_path, mimetype='image/png')
        
        # If not, generate it on the fly and serve it
        secure_name = secure_filename(filename)
        labels = load_labels()
        metadata = labels.get(secure_name, {})
        
        try:
            # Generate the rigged image
            rigged_img = generate_rigged_image(
                os.path.join(UPLOAD_FOLDER, secure_name),
                metadata.get('structure_points', {}),
                metadata.get('wearable', 'unknown'),
                metadata.get('hand', 'unknown')
            )
            
            # Convert to bytes and serve
            img_bytes = io.BytesIO()
            rigged_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            return send_file(img_bytes, mimetype='image/png')
        except Exception as e:
            # Just serve the original image if rigging fails
            print(f"Error generating rigged image: {e}")
            return app.send_static_file(f"uploads/{secure_name}")

    @app.route("/verify_rigging/<filename>", methods=["POST"])
    def verify_rigging(filename):
        """
        Route to verify and regenerate rigging for a specific clothing item
        Useful for fixing problematic rigging
        """
        secure_name = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, secure_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            # Get clothing type and sleeve info from request
            data = request.get_json() or {}
            labels = load_labels()
            metadata = labels.get(secure_name, {})
            
            # Override with request data if provided
            clothing_type = data.get("clothing_type", metadata.get("wearable", "unknown"))
            sleeve_type = data.get("sleeve_type", metadata.get("hand", "unknown"))
            
            # Force reanalysis of the structure
            structure_points = detect_precise_structure(
                file_path, clothing_type, sleeve_type
            )
            
            # Update the metadata
            if secure_name in labels:
                labels[secure_name]['structure_points'] = structure_points
                save_labels(labels)
            
            # Generate new rigged image
            rigged_img = apply_rigging(file_path, {
                "wearable": clothing_type,
                "hand": sleeve_type
            })
            
            # Return success response
            return jsonify({
                "success": True,
                "message": "Rigging regenerated",
                "clothing_type": clothing_type,
                "sleeve_type": sleeve_type
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
