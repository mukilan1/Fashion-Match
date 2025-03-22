"""
Utilities for applying rigging and structure detection to clothing images
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER
from utils.file_utils import load_labels, save_labels
from utils.structure_analyzer import StructureDetector

# Create a singleton instance of the structure detector
structure_detector = StructureDetector()

def get_rigged_image_path(filename):
    """Get the path where the rigged image should be stored"""
    secure_name = secure_filename(filename)
    rigged_dir = os.path.join(UPLOAD_FOLDER, 'rigged')
    
    # Create the rigged directory if it doesn't exist
    if not os.path.exists(rigged_dir):
        os.makedirs(rigged_dir, exist_ok=True)
    
    rigged_path = os.path.join(rigged_dir, f"rigged_{secure_name}")
    return rigged_path

def is_rigged_image_cached(filename):
    """Check if a rigged version of the image is already cached"""
    rigged_path = get_rigged_image_path(filename)
    return os.path.exists(rigged_path)

def apply_rigging(file_path, metadata=None):
    """Apply rigging to a clothing image and save the result"""
    if metadata is None:
        # Get metadata from labels if not provided
        filename = os.path.basename(file_path)
        secure_name = secure_filename(filename)
        labels = load_labels()
        metadata = labels.get(secure_name, {})
    
    # Get metadata we need for rigging
    clothing_type = metadata.get("wearable", "unknown")
    sleeve_type = metadata.get("hand", "unknown")
    
    # Use enhanced structure detection with contour analysis
    structure_points = detect_precise_structure(
        file_path, clothing_type, sleeve_type
    )
    
    # Cache the structure points in metadata
    filename = os.path.basename(file_path)
    secure_name = secure_filename(filename)
    labels = load_labels()
    if secure_name in labels:
        labels[secure_name]['structure_points'] = structure_points
        save_labels(labels)
    
    # Now generate the rigged image with precise alignment
    rigged_img = generate_rigged_image(
        file_path, 
        structure_points, 
        clothing_type, 
        sleeve_type
    )
    
    # Save the rigged image
    rigged_path = get_rigged_image_path(filename)
    rigged_img.save(rigged_path, format='PNG')
    
    return rigged_img

def detect_precise_structure(file_path, clothing_type, sleeve_type):
    """
    Enhanced structure detection for clothing items
    
    Args:
        file_path: Path to the image file
        clothing_type: Type of clothing (top, bottom, dress, etc.)
        sleeve_type: Type of sleeve (full, half, none)
        
    Returns:
        Dictionary containing detailed structure points
    """
    # Use the structure detector to get basic structure points
    structure_points = structure_detector.detect_structure(file_path, clothing_type, sleeve_type)
    
    # Enhance with additional detail based on clothing and sleeve type
    if clothing_type and "top" in str(clothing_type).lower():
        # Add additional points for tops
        if "sleeve_ends" not in structure_points and sleeve_type:
            if "full" in str(sleeve_type).lower():
                structure_points["sleeve_ends"] = {
                    "left": {"x": 40, "y": 180},
                    "right": {"x": 260, "y": 180}
                }
            elif "half" in str(sleeve_type).lower():
                structure_points["sleeve_ends"] = {
                    "left": {"x": 40, "y": 100},
                    "right": {"x": 260, "y": 100}
                }
    
    # Add sleeve type detection information
    structure_points["detected_sleeve_type"] = {
        "has_sleeves": sleeve_type and not ('no ' in str(sleeve_type).lower() or 'none' in str(sleeve_type).lower()),
        "is_full_sleeve": sleeve_type and ('full' in str(sleeve_type).lower() or 'long' in str(sleeve_type).lower()),
        "is_half_sleeve": sleeve_type and ('half' in str(sleeve_type).lower() or 'short' in str(sleeve_type).lower())
    }
    
    return structure_points

def generate_rigged_image(file_path, structure_points, clothing_type, sleeve_type):
    """Generate a rigged version of the image with skeleton overlay"""
    # Open the original image
    img = Image.open(file_path)
    
    # Create a transparent overlay for the skeleton
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw the skeleton based on clothing type
    if "top" in str(clothing_type).lower():
        draw_top_skeleton(draw, structure_points, img.width, img.height, sleeve_type)
    elif "bottom" in str(clothing_type).lower():
        draw_bottom_skeleton(draw, structure_points, img.width, img.height)
    elif "dress" in str(clothing_type).lower():
        draw_full_skeleton(draw, structure_points, img.width, img.height, sleeve_type)
    else:
        # Default to full skeleton for unknown types
        draw_full_skeleton(draw, structure_points, img.width, img.height, sleeve_type)
    
    # Ensure original image has alpha channel
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Composite the original image and the skeleton overlay
    rigged_img = Image.alpha_composite(img, overlay)
    
    return rigged_img

def draw_top_skeleton(draw, structure_points, width, height, sleeve_type):
    """Draw skeleton for tops with proper alignment to the garment structure"""
    # Scale structure points to match actual drawing dimensions
    scaleX = width / 300
    scaleY = height / 300
    
    # Calculate key points based on structure analysis
    centerX = structure_points['neckline']['x'] * scaleX
    neckY = structure_points['neckline']['y'] * scaleY
    shoulderY = structure_points['shoulders']['left']['y'] * scaleY
    leftShoulderX = structure_points['shoulders']['left']['x'] * scaleX
    rightShoulderX = structure_points['shoulders']['right']['x'] * scaleX
    waistLeftX = structure_points['waistline']['left']['x'] * scaleX
    waistRightX = structure_points['waistline']['right']['x'] * scaleX
    waistY = structure_points['waistline']['left']['y'] * scaleY
    
    # Top color - semi-transparent for better visibility
    top_color = (255, 126, 103, 180)  # RGBA
    joint_color = (248, 210, 16, 200)  # RGBA
    line_width = max(4, int(min(width, height) / 75))  # Scale line width by image size
    
    # Draw shoulders - follows the actual shoulder line of the garment
    draw.line([(leftShoulderX, shoulderY), (rightShoulderX, shoulderY)], fill=top_color, width=line_width)
    
    # Draw neck to center shoulder
    draw.line([(centerX, neckY), (centerX, shoulderY)], fill=top_color, width=line_width)
    
    # Draw center line (sternum)
    draw.line([(centerX, shoulderY), (centerX, waistY)], fill=top_color, width=line_width)
    
    # Draw sides - follow the actual shape of the garment
    draw.line([(leftShoulderX, shoulderY), (waistLeftX, waistY)], fill=top_color, width=line_width)
    draw.line([(rightShoulderX, shoulderY), (waistRightX, waistY)], fill=top_color, width=line_width)
    
    # Draw bottom hem
    draw.line([(waistLeftX, waistY), (waistRightX, waistY)], fill=top_color, width=line_width)
    
    # Check if we should draw sleeves
    has_sleeves = False
    is_full_sleeve = False
    is_half_sleeve = False
    
    # Check the detected sleeve type
    detected_sleeve_info = structure_points.get("detected_sleeve_type", {})
    if detected_sleeve_info:
        has_sleeves = detected_sleeve_info.get("has_sleeves", False)
        is_full_sleeve = detected_sleeve_info.get("is_full_sleeve", False)
        is_half_sleeve = detected_sleeve_info.get("is_half_sleeve", False)
    else:
        # Fall back to sleeve_type parameter
        sleeve_type_lower = str(sleeve_type).lower()
        has_sleeves = not ('no ' in sleeve_type_lower or 'n/a' in sleeve_type_lower or 'none' in sleeve_type_lower)
        is_full_sleeve = 'full' in sleeve_type_lower or 'long' in sleeve_type_lower
        is_half_sleeve = 'half' in sleeve_type_lower or 'short' in sleeve_type_lower
    
    # Only draw arms if we detect sleeves
    if has_sleeves and 'sleeve_ends' in structure_points:
        # Calculate sleeve endpoints
        leftSleeveEndX = structure_points['sleeve_ends']['left']['x'] * scaleX
        leftSleeveEndY = structure_points['sleeve_ends']['left']['y'] * scaleY
        rightSleeveEndX = structure_points['sleeve_ends']['right']['x'] * scaleX
        rightSleeveEndY = structure_points['sleeve_ends']['right']['y'] * scaleY
        
        if is_full_sleeve:
            # For full sleeves, add an elbow joint
            if 'sleeve_elbows' in structure_points:
                # Use defined elbow points
                leftElbowX = structure_points['sleeve_elbows']['left']['x'] * scaleX
                leftElbowY = structure_points['sleeve_elbows']['left']['y'] * scaleY
                rightElbowX = structure_points['sleeve_elbows']['right']['x'] * scaleX
                rightElbowY = structure_points['sleeve_elbows']['right']['y'] * scaleY
            else:
                # Calculate elbow points 
                leftElbowX = leftShoulderX + (leftSleeveEndX - leftShoulderX) * 0.5
                leftElbowY = shoulderY + (leftSleeveEndY - shoulderY) * 0.5
                rightElbowX = rightShoulderX + (rightSleeveEndX - rightShoulderX) * 0.5
                rightElbowY = shoulderY + (rightSleeveEndY - shoulderY) * 0.5
            
            # Draw left sleeve with elbow
            draw.line([(leftShoulderX, shoulderY), (leftElbowX, leftElbowY)], 
                     fill=top_color, width=line_width)
            draw.line([(leftElbowX, leftElbowY), (leftSleeveEndX, leftSleeveEndY)], 
                     fill=top_color, width=line_width)
            
            # Draw right sleeve with elbow
            draw.line([(rightShoulderX, shoulderY), (rightElbowX, rightElbowY)], 
                     fill=top_color, width=line_width)
            draw.line([(rightElbowX, rightElbowY), (rightSleeveEndX, rightSleeveEndY)], 
                     fill=top_color, width=line_width)
            
            # Draw elbow joints
            joint_radius = max(5, int(min(width, height) / 100))
            draw.ellipse([(leftElbowX-joint_radius, leftElbowY-joint_radius), 
                         (leftElbowX+joint_radius, leftElbowY+joint_radius)], 
                         fill=joint_color)
            draw.ellipse([(rightElbowX-joint_radius, rightElbowY-joint_radius), 
                         (rightElbowX+joint_radius, rightElbowY+joint_radius)], 
                         fill=joint_color)
            
            # Draw sleeve end joints
            draw.ellipse([(leftSleeveEndX-joint_radius, leftSleeveEndY-joint_radius), 
                         (leftSleeveEndX+joint_radius, leftSleeveEndY+joint_radius)], 
                         fill=joint_color)
            draw.ellipse([(rightSleeveEndX-joint_radius, rightSleeveEndY-joint_radius), 
                         (rightSleeveEndX+joint_radius, rightSleeveEndY+joint_radius)], 
                         fill=joint_color)
        
        elif is_half_sleeve:
            # For half sleeves, draw straight from shoulder to end
            draw.line([(leftShoulderX, shoulderY), (leftSleeveEndX, leftSleeveEndY)], 
                     fill=top_color, width=line_width)
            draw.line([(rightShoulderX, shoulderY), (rightSleeveEndX, rightSleeveEndY)], 
                     fill=top_color, width=line_width)
            
            # Draw sleeve end joints
            joint_radius = max(5, int(min(width, height) / 100))
            draw.ellipse([(leftSleeveEndX-joint_radius, leftSleeveEndY-joint_radius), 
                         (leftSleeveEndX+joint_radius, leftSleeveEndY+joint_radius)], 
                         fill=joint_color)
            draw.ellipse([(rightSleeveEndX-joint_radius, rightSleeveEndY-joint_radius), 
                         (rightSleeveEndX+joint_radius, rightSleeveEndY+joint_radius)], 
                         fill=joint_color)
    
    # Draw main body joints
    joint_radius = max(5, int(min(width, height) / 100))
    joints = [
        (centerX, neckY),  # Neck
        (centerX, shoulderY),  # Sternum top
        (leftShoulderX, shoulderY),  # Left shoulder
        (rightShoulderX, shoulderY),  # Right shoulder
        (waistLeftX, waistY),  # Left hip
        (waistRightX, waistY),  # Right hip
        (centerX, waistY)  # Sternum bottom
    ]
    
    for joint in joints:
        draw.ellipse([(joint[0]-joint_radius, joint[1]-joint_radius), 
                     (joint[0]+joint_radius, joint[1]+joint_radius)], 
                     fill=joint_color)

def draw_bottom_skeleton(draw, structure_points, width, height):
    """Draw skeleton for bottom wearable items"""
    # Scale structure points to match actual drawing dimensions
    scaleX = width / 300
    scaleY = height / 300
    
    # Bottom color
    bottom_color = (0, 184, 169, 180)  # RGBA (teal with alpha)
    joint_color = (248, 210, 16, 200)  # RGBA (gold with alpha)
    line_width = 4
    
    # Extract key points from structure_points
    waistlineY = structure_points['waistline']['y'] * scaleY
    waistlineWidth = structure_points['waistline']['width'] * scaleX
    hiplineY = structure_points['hipline']['y'] * scaleY
    hiplineWidth = structure_points['hipline']['width'] * scaleX
    
    # Calculate center and side positions
    centerX = width / 2
    waistLeftX = centerX - (waistlineWidth / 2)
    waistRightX = centerX + (waistlineWidth / 2)
    hipLeftX = centerX - (hiplineWidth / 2)
    hipRightX = centerX + (hiplineWidth / 2)
    
    # Get knee and hem positions
    leftKneeX = structure_points['knees']['left']['x'] * scaleX
    leftKneeY = structure_points['knees']['left']['y'] * scaleY
    rightKneeX = structure_points['knees']['right']['x'] * scaleX
    rightKneeY = structure_points['knees']['right']['y'] * scaleY
    
    leftHemX = structure_points['hems']['left']['x'] * scaleX
    leftHemY = structure_points['hems']['left']['y'] * scaleY
    rightHemX = structure_points['hems']['right']['x'] * scaleX
    rightHemY = structure_points['hems']['right']['y'] * scaleY
    
    # Draw waistline
    draw.line([(waistLeftX, waistlineY), (waistRightX, waistlineY)], 
              fill=bottom_color, width=line_width)
    
    # Draw center line from waist to hip level
    centerHipY = (hiplineY + waistlineY) / 2
    draw.line([(centerX, waistlineY), (centerX, centerHipY)], 
              fill=bottom_color, width=line_width)
    
    # Draw hip outline
    draw.line([(hipLeftX, hiplineY), (hipRightX, hiplineY)], 
              fill=bottom_color, width=line_width)
    draw.line([(waistLeftX, waistlineY), (hipLeftX, hiplineY)], 
              fill=bottom_color, width=line_width)
    draw.line([(waistRightX, waistlineY), (hipRightX, hiplineY)], 
              fill=bottom_color, width=line_width)
    
    # Draw legs
    draw.line([(hipLeftX, hiplineY), (leftKneeX, leftKneeY)], 
              fill=bottom_color, width=line_width)
    draw.line([(hipRightX, hiplineY), (rightKneeX, rightKneeY)], 
              fill=bottom_color, width=line_width)
    
    # Draw from knees to hems
    draw.line([(leftKneeX, leftKneeY), (leftHemX, leftHemY)], 
              fill=bottom_color, width=line_width)
    draw.line([(rightKneeX, rightKneeY), (rightHemX, rightHemY)], 
              fill=bottom_color, width=line_width)
    
    # Draw joints
    joint_radius = 5
    joints = [
        (waistLeftX, waistlineY),  # Left waist
        (waistRightX, waistlineY),  # Right waist
        (centerX, waistlineY),  # Center waist
        (hipLeftX, hiplineY),  # Left hip
        (hipRightX, hiplineY),  # Right hip
        (leftKneeX, leftKneeY),  # Left knee
        (rightKneeX, rightKneeY),  # Right knee
        (leftHemX, leftHemY),  # Left hem
        (rightHemX, rightHemY),  # Right hem
    ]
    
    for joint in joints:
        draw.ellipse([(joint[0]-joint_radius, joint[1]-joint_radius), 
                     (joint[0]+joint_radius, joint[1]+joint_radius)], 
                     fill=joint_color)

def draw_full_skeleton(draw, structure_points, width, height, sleeve_type):
    """Draw skeleton for full body items like dresses"""
    # Draw top parts first
    draw_top_skeleton(draw, structure_points, width, height, sleeve_type)
    
    # Then add the skirt/bottom part
    if 'hemline' in structure_points:
        # Draw hemline with bottom color
        bottom_color = (0, 184, 169, 180)  # RGBA
        joint_color = (248, 210, 16, 200)  # RGBA
        line_width = 4
        
        # Scale structure points
        scaleX = width / 300
        scaleY = height / 300
        
        # Get waist position from the top skeleton
        centerX = structure_points['neckline']['x'] * scaleX
        waistY = structure_points['waistline']['left']['y'] * scaleY
        waistLeftX = structure_points['waistline']['left']['x'] * scaleX
        waistRightX = structure_points['waistline']['right']['x'] * scaleX
        
        # Get hemline
        leftHemX = structure_points['hemline']['left']['x'] * scaleX
        leftHemY = structure_points['hemline']['left']['y'] * scaleY
        rightHemX = structure_points['hemline']['right']['x'] * scaleX
        rightHemY = structure_points['hemline']['right']['y'] * scaleY
        
        # Draw connection from waist to hem
        draw.line([(waistLeftX, waistY), (leftHemX, leftHemY)], 
                  fill=bottom_color, width=line_width)
        draw.line([(waistRightX, waistY), (rightHemX, rightHemY)], 
                  fill=bottom_color, width=line_width)
        
        # Draw hemline
        draw.line([(leftHemX, leftHemY), (rightHemX, rightHemY)], 
                  fill=bottom_color, width=line_width)
        
        # Draw hem joints
        draw.ellipse([(leftHemX-5, leftHemY-5), (leftHemX+5, leftHemY+5)], 
                     fill=joint_color)
        draw.ellipse([(rightHemX-5, rightHemY-5), (rightHemX+5, rightHemY+5)], 
                     fill=joint_color)
