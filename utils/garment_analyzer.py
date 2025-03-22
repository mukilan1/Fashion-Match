"""
Advanced garment analysis tools for structure detection and validation
"""
import cv2
import numpy as np
from PIL import Image

def validate_sleeve_type(image_path, metadata):
    """
    ENHANCED: Analyzes an image to validate and correctly identify sleeve type,
    even with wrinkled or imperfectly aligned garments
    """
    # Load image and declared sleeve type
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return metadata.get("hand", "unknown")
    
    sleeve_type = metadata.get("hand", "unknown")
    wearable_type = metadata.get("wearable", "unknown")
    
    # If it's not a top, don't try to validate sleeve type
    if not wearable_type or "bottom" in str(wearable_type).lower():
        return sleeve_type
    
    # Get binary mask of the garment
    if img.shape[2] == 4:  # Has alpha
        mask = img[:,:,3]
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        # Convert to grayscale and threshold
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_mask = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return sleeve_type
    
    # Get main contour
    main_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # IMPROVED: Check the actual garment shape beyond just contour
    # This is better for wrinkled garments where contours might be misleading
    
    # Apply skeletonization to get the core structure
    # Skeletonization reduces the binary image to a thin line core structure
    # This helps identify how far sleeves extend despite wrinkles
    
    skeleton = np.zeros_like(binary_mask)
    binary_mask_copy = binary_mask.copy()
    while True:
        # Erosion
        eroded = cv2.erode(binary_mask_copy, kernel)
        
        # Opening operation
        temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        
        # Subtract to get the skeleton
        temp = cv2.subtract(eroded, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_mask_copy = eroded.copy()
        
        # Check if the image is empty
        if cv2.countNonZero(eroded) == 0:
            break
    
    # Analyze the skeleton endpoints - these often correspond to sleeve ends
    # Endpoints are pixels with exactly one neighbor
    endpoints = np.zeros_like(skeleton)
    for i in range(1, skeleton.shape[0]-1):
        for j in range(1, skeleton.shape[1]-1):
            if skeleton[i, j] == 255:  # If this is part of the skeleton
                neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2] > 0) - 1  # -1 to exclude the pixel itself
                if neighbors == 1:  # It's an endpoint
                    endpoints[i, j] = 255
    
    # Find coordinates of endpoints
    endpoint_coords = np.where(endpoints == 255)
    endpoint_points = list(zip(endpoint_coords[1], endpoint_coords[0]))  # (x,y) format
    
    # Analyze the distance of endpoints from the center
    # Endpoints far from the center are likely sleeve ends
    center_x, center_y = x + w/2, y + h/2
    
    # Get the left and right halves
    left_side_points = [p for p in endpoint_points if p[0] < center_x]
    right_side_points = [p for p in endpoint_points if p[0] >= center_x]
    
    # Score endpoints by how likely they are to be sleeve ends
    # This is based on their position relative to the garment
    sleeve_candidates = []
    
    # Define the main body rectangle (exclude extremities)
    body_rect = (x + w * 0.2, y, w * 0.6, h * 0.9)
    
    # Check left side endpoints
    for px, py in left_side_points:
        # Skip if the point is within the main body area
        if body_rect[0] <= px <= body_rect[0] + body_rect[2] and body_rect[1] <= py <= body_rect[1] + body_rect[3]:
            continue
        
        # Skip if at the very bottom (likely not a sleeve)
        if py > y + h * 0.9:
            continue
        
        # Calculate how far out to the left and how far down
        left_extension = (center_x - px) / w
        vertical_position = (py - y) / h
        
        # Score - higher for farther left and in middle vertical section
        # The 0.2-0.7 vertical range is where sleeves typically are
        if 0.2 <= vertical_position <= 0.7:
            sleeve_candidates.append((left_extension, vertical_position, (px, py)))
    
    # Check right side endpoints (similar approach)
    for px, py in right_side_points:
        if body_rect[0] <= px <= body_rect[0] + body_rect[2] and body_rect[1] <= py <= body_rect[1] + body_rect[3]:
            continue
        
        if py > y + h * 0.9:
            continue
        
        right_extension = (px - center_x) / w
        vertical_position = (py - y) / h
        
        if 0.2 <= vertical_position <= 0.7:
            sleeve_candidates.append((right_extension, vertical_position, (px, py)))
    
    # Sort by extension (how far out they go)
    sleeve_candidates.sort(reverse=True)
    
    # Determine sleeve type based on extension and position
    if not sleeve_candidates:
        # No clear sleeve endpoints found - might be sleeveless
        return "No Sleeves"
    
    # Check the top candidates to determine sleeve length
    max_extension = sleeve_candidates[0][0]
    max_vertical = max([vc[1] for vc in sleeve_candidates[:min(3, len(sleeve_candidates))]])
    
    # Analyze how far the sleeve extends horizontally
    # Full sleeves typically extend further horizontally
    if max_extension > 0.3 and max_vertical > 0.4:
        # Likely full sleeve - extends significantly to the side and down
        return "Full Hand"
    elif max_extension > 0.2:
        # Moderate extension - likely half sleeve
        return "Half Hand"
    else:
        # Minimal extension - likely sleeveless or cap sleeve
        return "No Sleeves"

def extract_garment_features(image_path):
    """
    Extract detailed features of the garment for advanced analysis
    """
    # Placeholder for future implementation
    return {}
