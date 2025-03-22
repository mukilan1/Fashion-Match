"""
Structure detection and analysis for clothing items
"""
import os
import cv2
import numpy as np
from PIL import Image

class StructureDetector:
    """
    Detects the structural components of clothing items
    for use in rigging and pose visualization.
    """
    def __init__(self):
        """Initialize the structure detector with default parameters"""
        self.default_top_structure = {
            "neckline": {
                "x": 150,  # Center x position (normalized to 300)
                "y": 30,   # Top position (normalized to 300)
                "width": 60  # Neckline width (normalized to 300)
            },
            "shoulders": {
                "left": {"x": 80, "y": 60},
                "right": {"x": 220, "y": 60},
                "width": 140  # Shoulder width
            },
            "waistline": {
                "left": {"x": 100, "y": 250},
                "right": {"x": 200, "y": 250}
            },
            "chest": {
                "width": 160,
                "y": 120
            }
        }
        
        self.default_bottom_structure = {
            "waistline": {
                "width": 140,
                "y": 30
            },
            "hipline": {
                "width": 160,
                "y": 70
            },
            "knees": {
                "left": {"x": 120, "y": 180},
                "right": {"x": 180, "y": 180}
            },
            "hems": {
                "left": {"x": 120, "y": 270},
                "right": {"x": 180, "y": 270}
            }
        }
        
        self.default_dress_structure = {
            "neckline": {
                "x": 150,
                "y": 30,
                "width": 60
            },
            "shoulders": {
                "left": {"x": 80, "y": 60},
                "right": {"x": 220, "y": 60},
                "width": 140
            },
            "waistline": {
                "left": {"x": 100, "y": 150},
                "right": {"x": 200, "y": 150}
            },
            "chest": {
                "width": 160,
                "y": 100
            },
            "hemline": {
                "left": {"x": 100, "y": 270},
                "right": {"x": 200, "y": 270}
            }
        }
    
    def detect_structure(self, file_path, clothing_type, sleeve_type):
        """
        Detect the structure of a clothing item
        
        Args:
            file_path: Path to the image file
            clothing_type: Type of clothing (top, bottom, dress, etc.)
            sleeve_type: Type of sleeve (full, half, none)
            
        Returns:
            Dictionary containing structure points normalized to 300x300 space
        """
        # Read the image
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read image: {file_path}")
                
            # Get image dimensions
            h, w = img.shape[:2]
            
            # Use contour-based structure detection if possible
            structure_points = self.detect_contour_structure(img, clothing_type, sleeve_type)
            
            # If contour detection fails, use default structure based on clothing type
            if not structure_points:
                structure_points = self.get_default_structure(clothing_type, sleeve_type)
                
            return structure_points
            
        except Exception as e:
            print(f"Error detecting structure in {file_path}: {str(e)}")
            # Return default structure if analysis fails
            return self.get_default_structure(clothing_type, sleeve_type)
    
    def detect_contour_structure(self, img, clothing_type, sleeve_type):
        """
        Detect structure based on image contours
        
        Args:
            img: Image as numpy array
            clothing_type: Type of clothing
            sleeve_type: Type of sleeves
            
        Returns:
            Dictionary of structure points or None if detection fails
        """
        try:
            # Create binary mask
            if img.shape[2] == 4:  # Has alpha
                mask = img[:,:,3]
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                binary_mask = cv2.adaptiveThreshold(img_gray, 255, 
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY_INV, 11, 2)
            
            # Clean up with morphological operations
            kernel = np.ones((5,5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Get largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Normalize to 300x300 space
            img_h, img_w = img.shape[:2]
            
            # Function to normalize coordinates
            def norm_x(px): return int((px / img_w) * 300)
            def norm_y(py): return int((py / img_h) * 300)
            
            # Check clothing type and construct appropriate structure points
            structure_points = {}
            
            if clothing_type and "top" in clothing_type.lower() or "dress" in clothing_type.lower():
                # Detect top or dress structure
                # ... (simplified for illustration) ...
                center_x = norm_x(x + w/2)
                neckline_y = norm_y(y + h * 0.05)
                shoulder_y = norm_y(y + h * 0.15)
                left_shoulder_x = norm_x(x + w * 0.2)
                right_shoulder_x = norm_x(x + w * 0.8)
                
                # Set appropriate waist position based on garment type
                if clothing_type and "dress" in clothing_type.lower():
                    waist_y = norm_y(y + h * 0.4)  # Higher waist for dresses
                    hem_y = norm_y(y + h * 0.95)
                else:
                    waist_y = norm_y(y + h * 0.8)  # Lower waist for tops
                
                waist_left_x = norm_x(x + w * 0.25)
                waist_right_x = norm_x(x + w * 0.75)
                
                # Create structure dictionary
                structure_points["neckline"] = {
                    "x": center_x,
                    "y": neckline_y,
                    "width": int(w * 0.3)
                }
                
                structure_points["shoulders"] = {
                    "left": {"x": left_shoulder_x, "y": shoulder_y},
                    "right": {"x": right_shoulder_x, "y": shoulder_y},
                    "width": right_shoulder_x - left_shoulder_x
                }
                
                structure_points["waistline"] = {
                    "left": {"x": waist_left_x, "y": waist_y},
                    "right": {"x": waist_right_x, "y": waist_y}
                }
                
                structure_points["chest"] = {
                    "width": norm_x(w * 0.7),
                    "y": norm_y(y + h * 0.3)
                }
                
                if "dress" in clothing_type.lower():
                    structure_points["hemline"] = {
                        "left": {"x": norm_x(x + w * 0.3), "y": hem_y},
                        "right": {"x": norm_x(x + w * 0.7), "y": hem_y}
                    }
                
                # Add sleeve endpoints if applicable
                if sleeve_type and ("full" in sleeve_type.lower() or "half" in sleeve_type.lower()):
                    # Basic sleeve ends - would be refined in actual implementation
                    if "full" in sleeve_type.lower():
                        structure_points["sleeve_ends"] = {
                            "left": {"x": norm_x(x), "y": norm_y(y + h * 0.6)},
                            "right": {"x": norm_x(x + w), "y": norm_y(y + h * 0.6)}
                        }
                    else:  # half sleeves
                        structure_points["sleeve_ends"] = {
                            "left": {"x": norm_x(x), "y": norm_y(y + h * 0.3)},
                            "right": {"x": norm_x(x + w), "y": norm_y(y + h * 0.3)}
                        }
            
            elif clothing_type and "bottom" in clothing_type.lower():
                # Detect bottom wear structure
                # ... (simplified for illustration) ...
                structure_points = self.default_bottom_structure.copy()
                
                # Adjust based on actual dimensions
                structure_points["waistline"]["width"] = norm_x(w * 0.7)
                structure_points["waistline"]["y"] = norm_y(y + h * 0.05)
                
                structure_points["hipline"]["width"] = norm_x(w * 0.85)
                structure_points["hipline"]["y"] = norm_y(y + h * 0.2)
                
                structure_points["knees"]["left"]["x"] = norm_x(x + w * 0.35)
                structure_points["knees"]["left"]["y"] = norm_y(y + h * 0.6)
                structure_points["knees"]["right"]["x"] = norm_x(x + w * 0.65)
                structure_points["knees"]["right"]["y"] = norm_y(y + h * 0.6)
                
                structure_points["hems"]["left"]["x"] = norm_x(x + w * 0.35)
                structure_points["hems"]["left"]["y"] = norm_y(y + h * 0.95)
                structure_points["hems"]["right"]["x"] = norm_x(x + w * 0.65)
                structure_points["hems"]["right"]["y"] = norm_y(y + h * 0.95)
            
            return structure_points
            
        except Exception as e:
            print(f"Error in contour-based structure detection: {str(e)}")
            return None
    
    def get_default_structure(self, clothing_type, sleeve_type):
        """
        Get default structure based on clothing type
        
        Args:
            clothing_type: Type of clothing
            sleeve_type: Type of sleeve
            
        Returns:
            Dictionary with default structure points
        """
        if clothing_type and "bottom" in clothing_type.lower():
            return self.default_bottom_structure.copy()
        elif clothing_type and "dress" in clothing_type.lower():
            structure = self.default_dress_structure.copy()
        else:
            # Default to top structure
            structure = self.default_top_structure.copy()
        
        # Add sleeve endpoints if applicable
        if sleeve_type:
            if "full" in sleeve_type.lower():
                structure["sleeve_ends"] = {
                    "left": {"x": 40, "y": 180},
                    "right": {"x": 260, "y": 180}
                }
            elif "half" in sleeve_type.lower():
                structure["sleeve_ends"] = {
                    "left": {"x": 40, "y": 100},
                    "right": {"x": 260, "y": 100}
                }
        
        return structure
