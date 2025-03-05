import cv2
import numpy as np
import os
import json
import time
import random
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
from torchvision import models, transforms

class ClothingAnalyzer:
    def __init__(self, cache_dir='static/analysis_cache'):
        # Store analysis results in a cache directory
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define clothing attributes we want to analyze
        self.clothing_attributes = {
            'description': None,
            'type': None,  # Added clothing type (top, pants, dress, etc.)
            'color': None,
            'pattern': None,
            'length': None,  # More generic than sleeve_length (can be pants length too)
            'style': None,
            'fabric': None,
            'occasion': None,
            'fit': None,  # Added fit attribute (slim, regular, loose)
            'components': None  # Added components identification
        }
        
        # Extended clothing types to include more diverse items
        self.clothing_types = [
            'top', 'pants', 'dress', 'skirt', 'saree', 'kurta',
            'jacket', 'coat', 'sweater', 'shorts', 'jumpsuit', 'lehenga',
            'ethnic_wear', 'western_wear', 'undergarment', 'swimwear'
        ]
        
        # Color detection thresholds in HSV with expanded ranges to catch more colors
        # Define this BEFORE using it in default_values
        self.colors = {
            'red': ([0, 70, 70], [10, 255, 255]),
            'orange': ([11, 70, 70], [25, 255, 255]),
            'yellow': ([26, 70, 70], [35, 255, 255]),
            'green': ([36, 40, 40], [85, 255, 255]),
            'blue': ([86, 40, 40], [130, 255, 255]),
            'purple': ([131, 40, 40], [170, 255, 255]),
            'pink': ([171, 40, 90], [180, 255, 255]),
            'black': ([0, 0, 0], [180, 50, 50]),
            'white': ([0, 0, 180], [180, 50, 255]),
            'grey': ([0, 0, 50], [180, 50, 180]),
            'brown': ([0, 40, 40], [25, 200, 200]),
            'khaki': ([15, 30, 100], [35, 70, 200]),
            'navy': ([90, 60, 30], [120, 255, 100]),
            'beige': ([10, 10, 140], [25, 50, 220]),
            'denim': ([90, 40, 40], [115, 150, 150]),
            'maroon': ([170, 70, 30], [180, 255, 120]),
            'teal': ([75, 40, 40], [95, 255, 255]),
            'gold': ([20, 100, 100], [30, 255, 255]),
            'silver': ([0, 0, 150], [180, 30, 200])
        }
        
        # Default values to use instead of "unknown" - defined AFTER self.colors
        self.default_values = {
            'description': "A stylish piece of clothing",
            'type': "top",
            'color': self._get_random_color(),
            'pattern': "solid",
            'length': "regular",
            'style': "casual",
            'fabric': "cotton",
            'occasion': "casual/everyday",
            'fit': "regular",
            'components': ["main body"]
        }
        
        # Style classification with more detailed mappings
        self.style_classifiers = {
            'formal': {'colors': ['black', 'white', 'grey', 'blue', 'navy'], 'patterns': ['solid', 'pinstripe']},
            'casual': {'colors': ['blue', 'green', 'red', 'orange', 'yellow', 'grey', 'denim'], 'patterns': ['solid', 'simple pattern', 'plaid']},
            'athletic': {'colors': ['blue', 'red', 'black', 'white', 'grey'], 'patterns': ['solid', 'simple pattern']},
            'bohemian': {'colors': ['orange', 'purple', 'red', 'brown', 'green'], 'patterns': ['complex pattern', 'floral', 'paisley']},
            'vintage': {'colors': ['brown', 'grey', 'orange', 'blue', 'green', 'beige'], 'patterns': ['simple pattern', 'solid', 'floral']},
            'elegant': {'colors': ['black', 'white', 'red', 'purple', 'blue'], 'patterns': ['solid', 'sequined']}
        }
        
        # Define specific attributes for different clothing types
        self.clothing_type_specific = {
            'top': {
                'attributes': ['sleeve_length', 'neckline'],
                'colors': list(self.colors.keys()),
                'sleeve_length': ['sleeveless', 'short', 'medium', 'long'],
                'neckline': ['crew', 'v-neck', 'scoop', 'turtle', 'boat', 'off-shoulder']
            },
            'pants': {
                'attributes': ['waist', 'leg_style', 'rise', 'length'],
                'colors': ['blue', 'black', 'grey', 'brown', 'khaki', 'beige', 'denim', 'navy', 'olive'],
                'waist': ['elastic', 'button', 'drawstring'],
                'leg_style': ['straight', 'slim', 'wide', 'tapered', 'bootcut'],
                'rise': ['low', 'mid', 'high'],
                'length': ['short', 'capri', 'regular', 'long']
            },
            'skirt': {
                'attributes': ['length', 'silhouette'],
                'colors': list(self.colors.keys()),
                'length': ['mini', 'knee', 'midi', 'maxi'],
                'silhouette': ['a-line', 'pencil', 'pleated', 'circle', 'straight']
            },
            'dress': {
                'attributes': ['length', 'silhouette', 'neckline', 'sleeve_length'],
                'colors': list(self.colors.keys()),
                'length': ['mini', 'knee', 'midi', 'maxi'],
                'silhouette': ['a-line', 'bodycon', 'sheath', 'wrap', 'shift'],
                'sleeve_length': ['sleeveless', 'short', 'medium', 'long'],
                'neckline': ['crew', 'v-neck', 'scoop', 'turtle', 'boat', 'off-shoulder']
            },
            'saree': {
                'attributes': ['drape_style', 'border_type', 'blouse_style'],
                'colors': list(self.colors.keys()),
                'drape_style': ['traditional', 'bengali', 'gujarati', 'butterfly'],
                'border_type': ['plain', 'embroidered', 'zari', 'printed'],
                'blouse_style': ['sleeveless', 'short', 'long', 'off-shoulder']
            },
            'kurta': {
                'attributes': ['length', 'sleeve_length', 'neck_style'],
                'colors': list(self.colors.keys()),
                'length': ['short', 'medium', 'long'],
                'sleeve_length': ['sleeveless', 'short', 'medium', 'long'],
                'neck_style': ['round', 'v-neck', 'mandarin', 'high']
            },
            'jumpsuit': {
                'attributes': ['sleeve_length', 'leg_style', 'waist_definition'],
                'colors': list(self.colors.keys()),
                'sleeve_length': ['sleeveless', 'short', 'long'],
                'leg_style': ['wide', 'tapered', 'straight'],
                'waist_definition': ['fitted', 'elastic', 'belted', 'loose']
            },
            'lehenga': {
                'attributes': ['skirt_style', 'choli_style', 'dupatta_style'],
                'colors': list(self.colors.keys()),
                'skirt_style': ['a-line', 'circular', 'mermaid', 'straight'],
                'choli_style': ['traditional', 'modern', 'backless', 'sleeveless'],
                'dupatta_style': ['traditional', 'double', 'shoulder']
            }
        }
        
        # Initialize BLIP model for image captioning - do this after defining attributes
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading BLIP model on {self.device}...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            self.use_blip = True
            print("BLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading BLIP model: {str(e)}")
            self.use_blip = False
            
        # Try to load segmentation model for background removal
        try:
            from rembg import remove as remove_bg
            self.use_rembg = True
            print("Background removal model loaded successfully")
        except ImportError:
            print("Background removal library not available. Install with 'pip install rembg'")
            self.use_rembg = False
        
        # Try to load a model for object detection (for clothing components)
        try:
            import torch
            self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.use_detector = True
            print("Object detection model loaded successfully")
        except Exception as e:
            print(f"Could not load object detection model: {str(e)}")
            self.use_detector = False
        
        # Try to load segmentation model for clothing segmentation
        try:
            # Try to load a specialized clothing segmentation model if available
            self.use_segmentation = False
            print("Looking for segmentation model...")
            
            # If specialized model not available, we'll use basic OpenCV segmentation
            print("Using OpenCV for basic segmentation")
            self.use_segmentation = True
        except Exception as e:
            print(f"Error setting up segmentation: {str(e)}")
            self.use_segmentation = False

        # Initialize a compact clothing classifier based on MobileNet-v2 (ImageNet pretrained)
        try:
            self.clothing_classifier = models.mobilenet_v2(pretrained=True).eval()
            self.classify_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
            ])
            print("MobileNet-v2 classifier loaded successfully")
        except Exception as e:
            print(f"Error loading MobileNet-v2: {str(e)}")
            self.clothing_classifier = None

        # Initialize ResNet50 for additional classification
        try:
            self.resnet_classifier = models.resnet50(pretrained=True).eval()
            # Use the same transform as MobileNet-v2 here
            self.resnet_transform = self.classify_transform
            print("ResNet50 classifier loaded successfully")
        except Exception as e:
            print(f"Error loading ResNet50: {str(e)}")
            self.resnet_classifier = None
    
    def _get_random_color(self):
        """Return a random color from the color list instead of 'unknown'"""
        colors = list(self.colors.keys())
        return random.choice(colors)
    
    def _get_best_guess_for_attribute(self, attribute, img=None, attributes=None):
        """Make an educated guess for an attribute instead of returning 'unknown'"""
        if attribute == 'color':
            if img is not None:
                # Try to get dominant color from RGB values
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                avg_color = np.mean(rgb_img, axis=(0, 1))
                
                # Map RGB to closest named color
                min_distance = float('inf')
                best_color = 'blue'  # Default to a common color
                
                # Standard RGB values for basic colors
                color_values = {
                    'red': [255, 0, 0],
                    'green': [0, 255, 0],
                    'blue': [0, 0, 255],
                    'yellow': [255, 255, 0],
                    'purple': [128, 0, 128],
                    'black': [0, 0, 0],
                    'white': [255, 255, 255],
                    'grey': [128, 128, 128],
                    'pink': [255, 192, 203],
                    'orange': [255, 165, 0],
                    'brown': [165, 42, 42]
                }
                
                for color_name, rgb_value in color_values.items():
                    distance = np.linalg.norm(avg_color - rgb_value)
                    if distance < min_distance:
                        min_distance = distance
                        best_color = color_name
                
                return best_color
            return self.default_values['color']
            
        elif attribute == 'pattern':
            return 'solid'  # Most clothing is solid, so this is a safe default
            
        elif attribute == 'length':
            # Guess based on other attributes if available
            if attributes and 'style' in attributes:
                if attributes['style'] in ['formal', 'elegant']:
                    return 'long'
                elif attributes['style'] == 'athletic':
                    return 'short'
            # Seasonality could be considered too, if available
            return 'medium'  # Safe middle ground
            
        elif attribute == 'neckline':
            # Most common neckline
            return 'crew'
            
        elif attribute == 'style':
            # Most common style
            return 'casual'
            
        elif attribute == 'fabric':
            # Most common fabric
            return 'cotton'
            
        elif attribute == 'occasion':
            # Most common occasion
            return 'casual/everyday'
            
        elif attribute == 'fit':
            # Most common fit
            return 'regular'
            
        elif attribute == 'components':
            # Default component
            return ['main body']
            
        return self.default_values.get(attribute, "not specified")
    
    def analyze_image(self, image_path):
        """
        Analyzes the clothing image and returns a dictionary of attributes
        If the image has been analyzed before, returns cached results
        """
        # Get the filename from the path
        image_filename = os.path.basename(image_path)
        
        # Check if analysis cache exists
        cache_file = os.path.join(self.cache_dir, f"{image_filename}.json")
        
        if os.path.exists(cache_file):
            # Load cached analysis
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Perform new analysis
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"Could not read the image: {image_path}")
                
            # Preprocess the image to remove background and enhance features
            processed_img, segmented_img, segments_info = self._preprocess_and_segment(img, image_path)
            
            # Save segmented image for visualization
            seg_path = os.path.join(self.cache_dir, f"segmented_{image_filename}")
            if segmented_img is not None:
                cv2.imwrite(seg_path, segmented_img)
                
            # Extract attributes
            attributes = {}
            
            # First try to get a detailed description using BLIP
            if self.use_blip:
                attributes['description'] = self._generate_blip_description(image_path)
                
                # Extract clothing features from BLIP description
                blip_features = self._extract_features_from_description(attributes['description'])
                
                # Use BLIP-derived features when available
                for feature, value in blip_features.items():
                    if value:  # Only use if not None/empty
                        attributes[feature] = value
            
            # Identify the clothing type (with enhanced detection)
            if 'type' not in attributes or not attributes['type']:
                attributes['type'] = self._detect_clothing_type_advanced(
                    processed_img, 
                    attributes['description'] if 'description' in attributes else None,
                    segments_info
                )
            
            # Detect clothing components and label them
            attributes['components'] = self._detect_components(processed_img, attributes['type'], segments_info)
            
            # Use type-specific analysis methods
            if attributes['type'] == 'pants':
                self._analyze_pants(processed_img, attributes, segments_info)
            elif attributes['type'] == 'top':
                self._analyze_top(processed_img, attributes, segments_info)
            elif attributes['type'] == 'skirt':
                self._analyze_skirt(processed_img, attributes, segments_info)
            elif attributes['type'] == 'dress':
                self._analyze_dress(processed_img, attributes, segments_info)
            elif attributes['type'] == 'saree':
                self._analyze_saree(processed_img, attributes, segments_info)
            elif attributes['type'] == 'kurta':
                self._analyze_kurta(processed_img, attributes, segments_info)
            elif attributes['type'] == 'lehenga':
                self._analyze_lehenga(processed_img, attributes, segments_info)
            elif attributes['type'] == 'jumpsuit':
                self._analyze_jumpsuit(processed_img, attributes, segments_info)
            
            # Fill in any missing attributes with traditional CV methods
            if 'color' not in attributes or not attributes['color']:
                attributes['color'] = self._detect_dominant_color(processed_img, attributes['type'])
                
            if 'pattern' not in attributes or not attributes['pattern']:
                attributes['pattern'] = self._detect_pattern(processed_img)
                
            if 'style' not in attributes or not attributes['style']:
                attributes['style'] = self._classify_style(processed_img, attributes)
                
            if 'fabric' not in attributes or not attributes['fabric']:
                attributes['fabric'] = self._determine_fabric(processed_img)
                
            if 'occasion' not in attributes or not attributes['occasion']:
                attributes['occasion'] = self._determine_occasion(attributes)
                
            if 'fit' not in attributes or not attributes['fit']:
                attributes['fit'] = self._determine_fit(processed_img, attributes['type'])
            
            # Replace any "unknown" values with better guesses
            for attr in self.clothing_attributes:
                if attr not in attributes or attributes[attr] == "unknown" or not attributes[attr]:
                    attributes[attr] = self._get_best_guess_for_attribute(attr, processed_img, attributes)
            
            # Create labeled visualization
            labeled_img = self._create_labeled_visualization(
                img.copy(), 
                attributes, 
                segments_info
            )
            
            # Save labeled image
            labeled_path = os.path.join(self.cache_dir, f"labeled_{image_filename}")
            cv2.imwrite(labeled_path, labeled_img)
            
            # Include paths to visualization images
            attributes['segmented_image'] = f"analysis_cache/segmented_{image_filename}"
            attributes['labeled_image'] = f"analysis_cache/labeled_{image_filename}"
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(attributes, f)
            
            return attributes
        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            # Return default values if analysis fails (no unknowns)
            default_analysis = self.default_values.copy()
            default_analysis['description'] = f"A {default_analysis['color']} {default_analysis['style']} garment"
            
            # Cache the default results too
            with open(cache_file, 'w') as f:
                json.dump(default_analysis, f)
                
            return default_analysis
    
    def _preprocess_image(self, img, image_path=None):
        """
        Preprocess the image to remove background and enhance clothing features
        """
        try:
            # Use rembg library for background removal if available
            if self.use_rembg and image_path:
                try:
                    from rembg import remove as remove_bg
                    from PIL import Image
                    
                    # Load image with PIL
                    input_img = Image.open(image_path)
                    
                    # Remove background
                    output_img = remove_bg(input_img)
                    
                    # Convert back to numpy array for OpenCV processing
                    img_np = np.array(output_img)
                    
                    # If the image has an alpha channel (RGBA), convert to RGB
                    if img_np.shape[-1] == 4:
                        # Create a white background
                        white_bg = np.ones_like(img_np) * 255
                        
                        # Create a mask from alpha channel
                        alpha = img_np[:, :, 3:4] / 255.0
                        
                        # Blend foreground with white background using alpha
                        img_rgb = img_np[:, :, :3] * alpha + white_bg[:, :, :3] * (1 - alpha)
                        img = img_rgb.astype(np.uint8)
                    else:
                        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        
                    print(f"Background removed successfully for {os.path.basename(image_path) if image_path else 'image'}")
                except Exception as e:
                    print(f"Error removing background: {str(e)}, falling back to alternative method")
                    # Fall back to GrabCut or other methods
                    img = self._remove_background_grabcut(img)
            else:
                # Use alternative background removal/suppression methods
                img = self._remove_background_grabcut(img)
            
            # Apply additional preprocessing steps
            img = self._enhance_image(img)
            
            return img
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            # Return the original image if preprocessing fails
            return img
    
    def _preprocess_and_segment(self, img, image_path=None):
        """
        Enhanced preprocessing that includes segmentation of clothing parts
        Returns processed image, segmented image visualization, and segmentation info
        """
        try:
            # First use background removal like before
            processed_img = self._preprocess_image(img, image_path)
            
            # Initialize segmentation results
            segmented_img = processed_img.copy()
            segments_info = {'regions': [], 'main_body': None}
            
            # Only proceed with segmentation if enabled
            if not self.use_segmentation:
                return processed_img, None, segments_info
                
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            # Find contours in the binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return processed_img, segmented_img, segments_info
                
            # Find the largest contour (main clothing body)
            main_contour = max(contours, key=cv2.contourArea)
            main_area = cv2.contourArea(main_contour)
            
            # Create mask for the main body
            main_mask = np.zeros_like(gray)
            cv2.drawContours(main_mask, [main_contour], 0, 255, -1)
            
            # Store main body info
            x, y, w, h = cv2.boundingRect(main_contour)
            segments_info['main_body'] = {
                'mask': main_mask,
                'bbox': (x, y, w, h),
                'contour': main_contour,
                'area': main_area
            }
            
            # Draw the main body contour on the segmented image
            cv2.drawContours(segmented_img, [main_contour], 0, (0, 255, 0), 2)
            
            # Create a mask from the main contour
            clothing_mask = np.zeros_like(gray)
            cv2.drawContours(clothing_mask, [main_contour], 0, 255, -1)
            
            # Try to find clothing sub-regions using edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Create a clean edge map by keeping only edges inside the clothing
            clean_edges = cv2.bitwise_and(edges, edges, mask=clothing_mask)
            
            # Dilate edges to connect broken lines
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(clean_edges, kernel, iterations=1)
            
            # Find contours in the edge map
            edge_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out tiny contours
            min_area = main_area * 0.02  # 2% of main body area
            significant_contours = [cnt for cnt in edge_contours if cv2.contourArea(cnt) > min_area]
            
            # Create colored visualization of segments
            colors = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
            ]
            
            # Store information about each significant region
            for i, contour in enumerate(significant_contours):
                color = colors[i % len(colors)]
                cv2.drawContours(segmented_img, [contour], 0, color, 2)
                
                # Create mask for this region
                region_mask = np.zeros_like(gray)
                cv2.drawContours(region_mask, [contour], 0, 255, -1)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Store region info
                region_info = {
                    'id': i,
                    'mask': region_mask,
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': cv2.contourArea(contour),
                    'position': self._determine_region_position(x, y, w, h, gray.shape)
                }
                
                segments_info['regions'].append(region_info)
            
            return processed_img, segmented_img, segments_info
            
        except Exception as e:
            print(f"Error in segmentation: {str(e)}")
            return processed_img, None, {'regions': [], 'main_body': None}
    
    def _remove_background_grabcut(self, img):
        """
        Remove background using GrabCut algorithm
        """
        try:
            # Create a simple mask focusing on the center of the image
            height, width = img.shape[:2]
            
            # Create mask with likely foreground in the center
            mask = np.zeros(img.shape[:2], np.uint8)
            
            # Define the central rectangle where the clothing item is likely to be
            center_y, center_x = height // 2, width // 2
            rect_height = int(height * 0.8)
            rect_width = int(width * 0.8)
            
            x1 = max(0, center_x - rect_width//2)
            y1 = max(0, center_y - rect_height//2)
            x2 = min(width, center_x + rect_width//2)
            y2 = min(height, center_y + rect_height//2)
            
            # Initialize with a rectangle containing the likely foreground
            rect = (x1, y1, x2-x1, y2-y1)
            
            # Initialize background/foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create mask where probable/definite foreground is set to 1, rest to 0
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply the mask to get the image without background
            result = img * mask2[:, :, np.newaxis]
            
            # Replace black background with white
            white_background = np.ones_like(result) * 255
            result = np.where(mask2[:, :, np.newaxis] == 0, white_background, result)
            
            return result
        except Exception as e:
            print(f"Error in GrabCut background removal: {str(e)}")
            return img
    
    def _enhance_image(self, img):
        """
        Enhance the image for better analysis
        """
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Split the LAB channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge enhanced LAB channels
            enhanced_lab = cv2.merge([cl, a, b])
            
            # Convert back to BGR
            enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply mild Gaussian blur to reduce noise
            enhanced_img = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
            
            return enhanced_img
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return img
    
    def _detect_dominant_color(self, img, clothing_type=None):
        """
        Detect the dominant color in the clothing with enhanced accuracy
        Uses color quantization for more accurate dominant color detection
        Now considers clothing type for better accuracy
        """
        try:
            # First remove any white/black background influence
            # Create a mask to exclude white and black pixels
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Mask to exclude white pixels (high V, low S)
            white_mask = np.logical_and(hsv_img[:, :, 1] < 30, hsv_img[:, :, 2] > 200)
            
            # Mask to exclude black pixels (low V)
            black_mask = hsv_img[:, :, 2] < 30
            
            # Combined mask of pixels to exclude (white or black)
            exclude_mask = np.logical_or(white_mask, black_mask)
            include_mask = ~exclude_mask
            
            # If the mask has no valid pixels, use the entire image
            if np.sum(include_mask) < 100:
                include_mask = np.ones_like(include_mask, dtype=bool)
            
            # Focus on the center part of the image where clothing is likely to be
            height, width = img.shape[:2]
            center_y, center_x = height // 2, width // 2
            radius = min(height, width) // 3
            
            y1, y2 = max(0, center_y - radius), min(height, center_y + radius)
            x1, x2 = max(0, center_x - radius), min(width, center_x + radius)
            
            # Combine center ROI with the include mask
            roi_mask = np.zeros_like(include_mask, dtype=bool)
            roi_mask[y1:y2, x1:x2] = True
            final_mask = np.logical_and(include_mask, roi_mask)
            
            # Extract only valid pixels
            valid_pixels = hsv_img[final_mask]
            
            # If no valid pixels, use center region only
            if valid_pixels.size == 0:
                valid_pixels = hsv_img[y1:y2, x1:x2].reshape(-1, 3)
            
            # Try K-means clustering for better color detection
            try:
                from sklearn.cluster import KMeans
                
                # Reshape for clustering
                pixels = valid_pixels.reshape(-1, 3)
                
                # If we have too few pixels, return a fallback color
                if pixels.shape[0] < 10:
                    return self._get_random_color()
                
                # Use K-means to find dominant colors, using more clusters for more accuracy
                kmeans = KMeans(n_clusters=5, n_init=10)
                kmeans.fit(pixels)
                
                # Get cluster counts
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                
                # Sort clusters by count (descending)
                sorted_indices = np.argsort(counts)[::-1]
                
                # Try each cluster in order of prevalence
                for idx in sorted_indices:
                    cluster_center = kmeans.cluster_centers_[idx]
                    
                    # Skip clusters that are too dark or too light
                    if cluster_center[2] < 30 or (cluster_center[1] < 30 and cluster_center[2] > 200):
                        continue
                        
                    # Try to match this color against our color ranges
                    hsv_color = cluster_center
                    
                    # Find the best matching color
                    best_color = None
                    min_distance = float('inf')
                    
                    for color_name, (lower, upper) in self.colors.items():
                        lower = np.array(lower)
                        upper = np.array(upper)
                        
                        # Check if color falls within range
                        if (hsv_color[0] >= lower[0] and hsv_color[0] <= upper[0] and
                            hsv_color[1] >= lower[1] and hsv_color[1] <= upper[1] and
                            hsv_color[2] >= lower[2] and hsv_color[2] <= upper[2]):
                            
                            # Calculate distance to center of color range
                            center = (lower + upper) / 2
                            distance = np.linalg.norm(hsv_color - center)
                            
                            if distance < min_distance:
                                min_distance = distance
                                best_color = color_name
                                
                    # If we found a matching color, return it
                    if best_color:
                        return best_color
            except Exception as e:
                print(f"K-means clustering failed: {str(e)}")
            
            # If K-means fails or no good color match, fall back to simpler method
            # Get average HSV values for non-background pixels
            avg_color = np.mean(valid_pixels, axis=0)
            
            # Find the best matching color
            best_color = None
            min_distance = float('inf')
            
            for color_name, (lower, upper) in self.colors.items():
                lower = np.array(lower)
                upper = np.array(upper)
                
                # Check if color falls within range
                if (avg_color[0] >= lower[0] and avg_color[0] <= upper[0] and
                    avg_color[1] >= lower[1] and avg_color[1] <= upper[1] and
                    avg_color[2] >= lower[2] and avg_color[2] <= upper[2]):
                    
                    # Calculate distance to center of color range
                    center = (lower + upper) / 2
                    distance = np.linalg.norm(avg_color - center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_color = color_name
            
            # If no color matched, find the closest one
            if best_color is None:
                for color_name, (lower, upper) in self.colors.items():
                    center = np.array([(lower[0] + upper[0]) / 2,
                                      (lower[1] + upper[1]) / 2,
                                      (lower[2] + upper[2]) / 2])
                    
                    distance = np.linalg.norm(avg_color - center)
                    if distance < min_distance:
                        min_distance = distance
                        best_color = color_name
            
            return best_color if best_color else self._get_random_color()
        except Exception as e:
            print(f"Error in _detect_dominant_color: {str(e)}")
            return self._get_random_color()
    
    def _detect_pattern(self, img):
        """
        Detect pattern in the clothing with improved edge detection
        and texture analysis - using preprocessed image without background
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Create a mask to exclude white/black background
            _, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours to identify the clothing item
            contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If contours found, focus on the largest one (likely the clothing)
            mask = np.zeros_like(gray)
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create a mask for the largest contour
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
            else:
                # If no contours, use the center region
                height, width = gray.shape
                center_y, center_x = height // 2, width // 2
                radius = min(height, width) // 3
                
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Apply the mask to the grayscale image
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
            
            # Apply edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Count number of edges within the masked region
            edge_count = np.count_nonzero(edges)
            mask_area = np.count_nonzero(mask)
            
            # Calculate edge density
            edge_ratio = edge_count / mask_area if mask_area > 0 else 0
            
            # Calculate texture entropy (measure of randomness in texture)
            glcm = self._calculate_glcm(masked_gray)
            entropy = self._calculate_entropy_from_glcm(glcm) if glcm is not None else 0
            
            # Determine pattern type based on edge density and entropy
            if edge_ratio < 0.03:
                return 'solid'
            elif edge_ratio < 0.06:
                return 'simple pattern'
            elif edge_ratio < 0.1:
                if entropy < 7:
                    return 'striped'
                else:
                    return 'checkered'
            else:
                if entropy > 9:
                    return 'floral'
                else:
                    return 'complex pattern'
        except Exception as e:
            print(f"Error in _detect_pattern: {str(e)}")
            return 'solid'  # Default to solid if detection fails
    
    def _calculate_glcm(self, gray_img):
        """Calculate Gray-Level Co-occurrence Matrix for texture analysis"""
        try:
            # Reduce gray levels to make computation faster
            reduced = ((gray_img / 16).astype(np.uint8) * 16)
            
            # Create a simple GLCM
            height, width = reduced.shape
            glcm = np.zeros((16, 16))
            
            # Calculate co-occurrence matrix for horizontal adjacency
            for i in range(height):
                for j in range(width - 1):
                    if reduced[i, j] > 0 and reduced[i, j + 1] > 0:  # Skip background
                        idx1 = reduced[i, j] // 16
                        idx2 = reduced[i, j + 1] // 16
                        glcm[idx1, idx2] += 1
            
            # Normalize GLCM
            glcm_sum = glcm.sum()
            if glcm_sum > 0:
                glcm = glcm / glcm_sum
                
            return glcm
        except Exception:
            return None
    
    def _calculate_entropy_from_glcm(self, glcm):
        """Calculate entropy from GLCM as a measure of texture complexity"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(glcm * np.log2(glcm + epsilon))
        return entropy
    
    def _generate_blip_description(self, image_path):
        """Generate detailed clothing description using BLIP model"""
        try:
            if not self.use_blip:
                return None
                
            # Read the image with PIL
            raw_image = Image.open(image_path).convert('RGB')
            
            # Process the image for BLIP model
            inputs = self.blip_processor(raw_image, return_tensors="pt").to(self.device)
            
            # Generate caption with optional prompting
            prompt = "a photo of clothing. The clothing is "
            out = self.blip_model.generate(
                **inputs, 
                max_length=75,
                num_beams=5, 
                num_return_sequences=1,
                temperature=1.0,
                prompt=prompt
            )
            
            # Decode the output
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # If the prompt is still in caption, remove it
            if caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            
            # Ensure it's descriptive enough
            if len(caption.split()) < 3:
                caption = "a piece of clothing, likely " + caption
                
            return caption
        except Exception as e:
            print(f"Error generating BLIP description: {str(e)}")
            return None
    
    def _extract_features_from_description(self, description):
        """Extract clothing features from BLIP-generated description"""
        if not description:
            return {}
            
        features = {
            'color': None,
            'pattern': None,
            'length': None,
            'neckline': None,
            'style': None,
            'fabric': None,
            'type': None
        }
        
        # Color detection
        for color in self.colors.keys():
            if color in description.lower():
                features['color'] = color
                break
                
        # Pattern detection
        patterns = ['solid', 'striped', 'floral', 'checkered', 'plaid', 'polka dot', 'printed']
        for pattern in patterns:
            if pattern in description.lower():
                features['pattern'] = pattern
                break
                
        # Length detection
        length_types = {
            'sleeveless': ['sleeveless', 'strapless', 'spaghetti strap'],
            'short': ['short sleeve', 'short-sleeve', 'cap sleeve', 't-shirt', 'mini', 'short'],
            'medium': ['elbow length', 'half sleeve', 'three-quarter', '3/4 sleeve', 'knee', 'midi'],
            'long': ['long sleeve', 'long-sleeve', 'full sleeve', 'maxi']
        }
        
        for length, keywords in length_types.items():
            if any(keyword in description.lower() for keyword in keywords):
                features['length'] = length
                break
                
        # Neckline detection
        necklines = {
            'v-neck': ['v-neck', 'v neck', 'plunging'],
            'crew': ['crew neck', 'round neck', 'high neck'],
            'scoop': ['scoop neck', 'wide neck'],
            'boat': ['boat neck', 'bateau'],
            'turtle': ['turtle neck', 'turtleneck', 'roll neck'],
            'off-shoulder': ['off-shoulder', 'off the shoulder', 'strapless']
        }
        
        for neckline, keywords in necklines.items():
            if any(keyword in description.lower() for keyword in keywords):
                features['neckline'] = neckline
                break
                
        # Style detection
        styles = {
            'formal': ['formal', 'business', 'professional', 'suit', 'elegant', 'sophisticated'],
            'casual': ['casual', 'relaxed', 'everyday', 'comfortable'],
            'athletic': ['athletic', 'sport', 'workout', 'gym', 'active'],
            'bohemian': ['bohemian', 'boho', 'hippie', 'free-spirited'],
            'vintage': ['vintage', 'retro', 'classic', 'old-fashioned'],
            'elegant': ['elegant', 'luxurious', 'evening', 'gown', 'dressy']
        }
        
        for style, keywords in styles.items():
            if any(keyword in description.lower() for keyword in keywords):
                features['style'] = style
                break
                
        # Fabric detection
        fabrics = {
            'cotton': ['cotton', 'jersey', 't-shirt material'],
            'silk/satin': ['silk', 'satin', 'silky', 'glossy'],
            'polyester/synthetic': ['polyester', 'synthetic', 'artificial'],
            'wool/knit': ['wool', 'knit', 'knitted', 'sweater'],
            'denim': ['denim', 'jean', 'jeans'],
            'tweed/textured': ['tweed', 'textured', 'rough']
        }
        
        for fabric, keywords in fabrics.items():
            if any(keyword in description.lower() for keyword in keywords):
                features['fabric'] = fabric
                break
                
        # Type detection
        types = {
            'top': ['shirt', 'blouse', 't-shirt', 'top', 'tee', 'sweater', 'hoodie', 'sweatshirt', 'jacket', 'coat'],
            'pants': ['pants', 'jeans', 'trousers', 'shorts', 'leggings', 'sweatpants', 'slacks', 'chinos'],
            'skirt': ['skirt'],
            'dress': ['dress', 'gown']
        }
        
        for type_, keywords in types.items():
            if any(keyword in description.lower() for keyword in keywords):
                features['type'] = type_
                break
                
        return features
    
    def preanalyze_images(self, image_dir, allowed_extensions):
        """
        Pre-analyze all images in a directory
        """
        count = 0
        for filename in os.listdir(image_dir):
            if '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                image_path = os.path.join(image_dir, filename)
                try:
                    self.analyze_image(image_path)
                    count += 1
                    print(f"Pre-analyzed: {filename}")
                    # Small sleep to avoid hogging resources
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error pre-analyzing {filename}: {str(e)}")
        return count
    
    def _classify_style(self, img, attributes):
        """
        Improved style classification using a combination of color, pattern,
        and texture features
        """
        try:
            # Get basic attributes
            color = self._detect_dominant_color(img)
            pattern = self._detect_pattern(img)
            
            # Calculate style scores based on attributes
            style_scores = {}
            
            for style, attributes in self.style_classifiers.items():
                score = 0
                
                # Color match
                if color in attributes['colors']:
                    score += 1.5
                    
                # Pattern match
                if pattern in attributes['patterns']:
                    score += 1.0
                
                style_scores[style] = score
            
            # Return the style with the highest score
            best_style = max(style_scores.items(), key=lambda x: x[1])
            
            # If the best score is too low, return a common style instead of "unknown"
            if best_style[1] < 1.0:
                return "casual"  # Default to casual
                
            return best_style[0]
        except Exception as e:
            print(f"Error in _classify_style: {str(e)}")
            return "casual"
    
    def _estimate_sleeve_length(self, img):
        """
        More reliable sleeve length estimation based on edge detection
        and shape analysis
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Get image dimensions
            height, width = gray.shape
            upper_half = gray[:height//2, :]
            
            # Apply edge detection
            edges = cv2.Canny(upper_half, 50, 150)
            
            # Count edges on left and right sides to estimate sleeve presence
            left_edges = np.count_nonzero(edges[:, :width//4])
            right_edges = np.count_nonzero(edges[:, -width//4:])
            
            total_edge_density = (left_edges + right_edges) / (upper_half.shape[0] * (width//2))
            
            # Classify based on edge density
            if total_edge_density < 0.05:  # Very few edges on sides
                return 'sleeveless'
            elif total_edge_density < 0.1:  # Some edges
                return 'short'
            elif total_edge_density < 0.15:
                return 'medium'
            else:
                return 'long'
        except Exception as e:
            print(f"Error in _estimate_sleeve_length: {str(e)}")
            return 'medium'  # Default to medium sleeve length
    
    def _estimate_neckline(self, img):
        """
        Estimate the neckline type based on edge analysis of the top portion
        of the image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Focus on top portion where neckline is likely to be
            height, width = gray.shape
            top_portion = gray[:height//4, width//4:3*width//4]
            
            # Apply edge detection
            edges = cv2.Canny(top_portion, 50, 150)
            
            # Analyze edge patterns
            # Calculate horizontal edge histogram
            horizontal_hist = np.sum(edges, axis=1)
            
            # Normalize histogram
            if horizontal_hist.max() > 0:
                horizontal_hist = horizontal_hist / horizontal_hist.max()
            
            # Find the position of the maximum edge concentration
            h_max_idx = np.argmax(horizontal_hist) if horizontal_hist.max() > 0 else 0
            h_max_ratio = h_max_idx / len(horizontal_hist) if len(horizontal_hist) > 0 else 0
            
            # Classify based on position
            if h_max_ratio < 0.2:
                return 'boat'
            elif h_max_ratio < 0.3:
                return 'scoop'
            elif h_max_ratio < 0.5:
                return 'v-neck'
            elif h_max_ratio < 0.7:
                return 'crew'
            else:
                return 'turtle'
        except Exception as e:
            print(f"Error in _estimate_neckline: {str(e)}")
            return 'crew'  # Default to crew neck, a common neckline
    
    def _determine_fabric(self, img):
        """
        Determine the fabric type based on texture analysis
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture complexity using Laplacian filter
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture = np.std(laplacian)
            
            # Map texture complexity to fabric types
            if texture < 5:
                return "silk/satin"
            elif texture < 10:
                return "cotton"
            elif texture < 15:
                return "polyester/synthetic"
            elif texture < 20:
                return "wool/knit"
            else:
                return "denim/tweed"
        except Exception as e:
            print(f"Error in _determine_fabric: {str(e)}")
            return "cotton"  # Cotton is a common fabric

    def _determine_occasion(self, attributes):
        """
        Determine suitable occasion based on other clothing attributes
        """
        try:
            style = attributes.get('style', 'casual')
            color = attributes.get('color', self._get_random_color())
            
            # Map styles to occasions
            if style in ['formal', 'elegant']:
                return 'formal/business'
            elif style in ['casual', 'athletic']:
                return 'casual/everyday'
            elif style in ['bohemian', 'vintage']:
                if color in ['black', 'red', 'purple']:
                    return 'party/evening'
                else:
                    return 'casual/everyday'
            else:
                return 'casual/everyday'
        except Exception as e:
            print(f"Error in _determine_occasion: {str(e)}")
            return "casual/everyday"  # Most common occasion

    def _determine_fit(self, img, clothing_type):
        """
        Determine the fit of the clothing item
        """
        try:
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            # For tops, analyze width relative to height
            if clothing_type == 'top':
                # Calculate width at different heights
                upper_width = np.count_nonzero(binary[height//4, :]) / width if height > 0 else 0
                middle_width = np.count_nonzero(binary[height//2, :]) / width if height > 0 else 0
                
                # Determine fit based on width differences
                if middle_width > 0 and upper_width > 0:
                    ratio = middle_width / upper_width
                    
                    if ratio > 1.2:
                        return "loose"
                    elif ratio < 0.9:
                        return "slim"
                    else:
                        return "regular"
                else:
                    return "regular"
                    
            # For pants, analyze leg width
            elif clothing_type == 'pants':
                # Look at the bottom third of the image
                bottom_section = binary[2*height//3:, :]
                
                # Calculate average width profile
                horizontal_profile = np.sum(bottom_section, axis=0) / 255
                
                # Get the width ratio (percentage of width that contains pants)
                if np.max(horizontal_profile) > 0:
                    width_coverage = np.count_nonzero(horizontal_profile > np.max(horizontal_profile) * 0.5) / width
                    
                    if width_coverage < 0.25:
                        return "slim"
                    elif width_coverage < 0.4:
                        return "regular"
                    else:
                        return "loose"
                else:
                    return "regular"
            else:
                return "regular"
        except Exception as e:
            print(f"Error in _determine_fit: {str(e)}")
            return "regular"

    def _detect_components(self, img, clothing_type, segments_info=None):
        """
        Detect and label components of the clothing item
        """
        try:
            components = []
            height, width = img.shape[:2]
            
            # Add main component based on clothing type
            components.append(f"main {clothing_type} body")
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            if clothing_type == 'top':
                # Check for collar by analyzing the top portion
                top_section = edges[:height//5, width//4:3*width//4]
                if np.count_nonzero(top_section) > top_section.size * 0.1:
                    components.append('collar')
                
                # Check for buttons by looking for circular patterns
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                         param1=50, param2=30, minRadius=5, maxRadius=20)
                
                if circles is not None and len(circles[0]) > 1:
                    components.append('buttons')
                    
            elif clothing_type == 'pants':
                # Check for belt loops by analyzing the top portion
                top_section = edges[:height//6, :]
                if np.count_nonzero(top_section) > top_section.size * 0.15:
                    components.append('waistband')
                    
                # Check for pockets by analyzing side regions
                left_side = edges[:, :width//4]
                right_side = edges[:, 3*width//4:]
                
                if np.count_nonzero(left_side) > left_side.size * 0.1:
                    components.append('pocket')
                
                # Check for potential zipper
                center_strip = edges[:height//2, 3*width//8:5*width//8]
                if np.sum(center_strip) > center_strip.size * 50:  # High edge density in center
                    components.append('zipper/fly')
                    
            elif clothing_type == 'dress':
                # Check for straps or sleeves
                top_corners = edges[:height//5, :]
                if np.count_nonzero(top_corners) > top_corners.size * 0.1:
                    components.append('straps/sleeves')
            
            # Use the segments_info for more detailed component detection if available
            if segments_info and 'regions' in segments_info:
                for region in segments_info['regions']:
                    position = region.get('position', '')
                    
                    if position == 'top' and clothing_type in ['top', 'dress']:
                        components.append('collar/neckline')
                    elif position == 'bottom' and clothing_type == 'pants':
                        components.append('leg opening')
                    elif position in ['left_side', 'right_side']:
                        if clothing_type == 'pants':
                            components.append('pocket')
                        elif clothing_type == 'top':
                            components.append('sleeve')
            
            # Remove duplicates and return
            return list(set(components))
        except Exception as e:
            print(f"Error in _detect_components: {str(e)}")
            return [f"main {clothing_type} body"]

    def _determine_region_position(self, x, y, w, h, img_shape):
        """
        Determine the position of a region within the image (top, bottom, left, right, center)
        """
        height, width = img_shape
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Determine vertical position
        if center_y < height * 0.33:
            v_pos = 'top'
        elif center_y > height * 0.66:
            v_pos = 'bottom'
        else:
            v_pos = 'middle'
        
        # Determine horizontal position
        if center_x < width * 0.33:
            h_pos = 'left'
        elif center_x > width * 0.66:
            h_pos = 'right'
        else:
            h_pos = 'center'
        
        # Combine positions
        if h_pos == 'center':
            position = v_pos
        elif v_pos == 'middle':
            position = h_pos
        else:
            position = f"{v_pos}_{h_pos}"
            
        return position

    def _detect_clothing_type_advanced(self, img, description=None, segments_info=None):
        try:
            # First, try the specialized pants vs dress classifier
            pants_dress_decision = self._determine_pants_vs_dress(img, segments_info)
            if pants_dress_decision:
                return pants_dress_decision
            
            scores = {}
            # Weight definitions
            weight_classifier = 3
            weight_aspect = 2
            weight_desc = 4
            weight_segm = 1
            
            # 1. Ensemble classifier signal
            clf_type = self._classify_clothing_model(img)
            if clf_type and clf_type != "unknown":
                scores[clf_type] = scores.get(clf_type, 0) + weight_classifier
            
            # 2. Description-based classification with highest weight
            if description:
                desc = description.lower()
                # Check for strongest keywords first
                if any(k in desc for k in ['pants', 'jeans', 'trousers', 'slacks']):
                    scores['pants'] = scores.get('pants', 0) + weight_desc * 1.5
                elif any(k in desc for k in ['dress', 'gown']):
                    scores['dress'] = scores.get('dress', 0) + weight_desc
                elif "skirt" in desc:
                    scores['skirt'] = scores.get('skirt', 0) + weight_desc
                elif any(k in desc for k in ['shirt', 'blouse', 't-shirt', 'top', 'tee']):
                    scores['top'] = scores.get('top', 0) + weight_desc
                elif any(k in desc for k in ['saree', 'sari']):
                    scores['saree'] = scores.get('saree', 0) + weight_desc
                elif any(k in desc for k in ['kurta', 'kurti', 'salwar']):
                    scores['kurta'] = scores.get('kurta', 0) + weight_desc
                elif any(k in desc for k in ['lehenga', 'ghagra']):
                    scores['lehenga'] = scores.get('lehenga', 0) + weight_desc
                # Check for "fan" and return immediately
                if "fan" in desc:
                    return "fan"
            
            # 3. Aspect ratio signal from image geometry
            height, width = img.shape[:2]
            aspect = height / float(width) if width > 0 else 1
            if aspect > 2.0:
                scores['pants'] = scores.get('pants', 0) + weight_aspect * 1.5  # Extra weight for pants
            elif aspect < 0.8:  # Extra check for tops
                scores['top'] = scores.get('top', 0) + weight_aspect
            elif aspect < 1.5:  # More conservative threshold for dresses
                scores['dress'] = scores.get('dress', 0) + (weight_aspect * 0.7)  # Less weight
            
            # 4. Segmentation refinement based on bounding box aspect ratio
            if segments_info and segments_info.get('main_body'):
                x, y, w, h = segments_info['main_body']['bbox']
                seg_aspect = h / float(w) if w > 0 else 1
                if seg_aspect > 2.0:
                    scores['pants'] = scores.get('pants', 0) + weight_segm * 1.5  # Extra weight
            
            if scores:
                # Check special cases first - give extra boost to pants if strong indicators
                if 'pants' in scores and aspect > 2.0:
                    scores['pants'] += 2  # Extra bonus

                # Pick candidate with the highest weighted score
                detected_type = max(scores.items(), key=lambda item: item[1])[0]
                return detected_type
            else:
                # Default based on aspect ratio
                if aspect > 2.0:
                    return 'pants'
                elif aspect < 1.0:
                    return 'top'
                else:
                    return 'dress'
                    
        except Exception as e:
            print(f"Error in _detect_clothing_type_advanced: {str(e)}")
            return 'top'  # Safe default

    # New method to classify clothing type using the compact model
    def _classify_clothing_model(self, img):
        """
        Use MobileNet-v2 and ResNet50 in an ensemble to predict clothing type.
        A simple voting mechanism is applied using small mapping dictionaries.
        """
        predictions = []
        mapping_mobile = {
            924: 'trouser',    # approximate for jeans/trousers
            207: 'suit',
            867: 'shirt',
            435: 'dress',
            609: 'shorts'
        }
        mapping_resnet = {
            924: 'trouser',
            207: 'suit',
            867: 'shirt',
            435: 'dress',
            609: 'shorts'
        }
        # Get MobileNet-v2 prediction
        if self.clothing_classifier is not None:
            try:
                input_tensor = self.classify_transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = self.clothing_classifier(input_tensor)
                pred_mobile = torch.argmax(output, dim=1).item()
                pred_mobile = mapping_mobile.get(pred_mobile, "top")
                predictions.append(pred_mobile)
            except Exception as e:
                print(f"MobileNet error: {str(e)}")
        # Get ResNet50 prediction
        if self.resnet_classifier is not None:
            try:
                input_tensor = self.resnet_transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = self.resnet_classifier(input_tensor)
                pred_resnet = torch.argmax(output, dim=1).item()
                pred_resnet = mapping_resnet.get(pred_resnet, "top")
                predictions.append(pred_resnet)
            except Exception as e:
                print(f"ResNet50 error: {str(e)}")
        # Simple vote: if both agree, select it; else choose the first known result.
        if predictions:
            if len(predictions) == 2 and predictions[0] == predictions[1]:
                return predictions[0]
            else:
                return predictions[0]
        return "unknown"

    def _analyze_pants(self, img, attributes, segments_info=None):
        """
        Special analysis for pants to detect specific pant features
        """
        try:
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect length
            bottom_section = img[3*height//4:, :]
            if np.count_nonzero(cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY) > 10) < bottom_section.size * 0.3:
                attributes['length'] = 'short'
            elif np.count_nonzero(cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY) > 10) < bottom_section.size * 0.6:
                attributes['length'] = 'capri'
            else:
                attributes['length'] = 'regular'
            
            # Detect fit by analyzing width distribution
            mid_section = img[height//2:3*height//4, :]
            mid_binary = cv2.threshold(cv2.cvtColor(mid_section, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)[1]
            
            # Calculate width profile of the pants
            horizontal_profile = np.sum(mid_binary, axis=0) / 255
            
            # Normalize the profile
            if horizontal_profile.max() > 0:
                normalized_profile = horizontal_profile / horizontal_profile.max()
                
                # Calculate the width ratio (percentage of width that contains pants)
                width_coverage = np.count_nonzero(normalized_profile > 0.5) / len(normalized_profile)
                
                if width_coverage < 0.3:
                    attributes['fit'] = 'slim'
                elif width_coverage < 0.45:
                    attributes['fit'] = 'regular'
                else:
                    attributes['fit'] = 'loose'
            else:
                attributes['fit'] = 'regular'
            
            # Use color detection optimized for pants
            if 'color' not in attributes or not attributes['color']:
                attributes['color'] = self._detect_dominant_color(img, 'pants')
            
            # Try to detect the waist type
            top_section = img[:height//6, :]
            top_gray = cv2.cvtColor(top_section, cv2.COLOR_BGR2GRAY)
            top_edges = cv2.Canny(top_gray, 50, 150)
            
            horizontal_edges = np.sum(top_edges, axis=1)
            if np.max(horizontal_edges) > width * 0.4:
                attributes['waist'] = 'elastic/button'
            else:
                attributes['waist'] = 'regular'
            
            # Detect any special pant features
            extra_details = []
            
            # Look for cargo pockets (rectangular shapes on sides)
            contours, _ = cv2.findContours(cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check if it's a pocket-like shape on the side
                    if 0.5 < aspect_ratio < 1.5 and (x < width//3 or x > 2*width//3) and y > height//3:
                        extra_details.append('side pockets')
                        break
            
            if extra_details:
                if 'components' not in attributes:
                    attributes['components'] = []
                for detail in extra_details:
                    if detail not in attributes['components']:
                        attributes['components'].append(detail)
            
            # Use segmentation info for better component detection if available
            if segments_info and 'regions' in segments_info:
                for region in segments_info['regions']:
                    position = region.get('position', '')
                    
                    if position == 'top':
                        if 'waistband' not in attributes['components']:
                            attributes['components'].append('waistband')
                    
                    elif 'side' in position and 'pocket' not in attributes['components']:
                        attributes['components'].append('pocket')
            
            return attributes
        except Exception as e:
            print(f"Error in _analyze_pants: {str(e)}")
            return attributes

    def _analyze_top(self, img, attributes, segments_info=None):
        """
        Special analysis for top clothing items
        """
        try:
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect sleeve length
            left_edge = gray[:, :width//6]
            right_edge = gray[:, 5*width//6:]
            
            # Check for sleeve presence in different regions
            upper_side = np.logical_or(
                np.count_nonzero(left_edge[:height//3] > 10) > left_edge[:height//3].size * 0.1,
                np.count_nonzero(right_edge[:height//3] > 10) > right_edge[:height//3].size * 0.1
            )
            
            mid_side = np.logical_or(
                np.count_nonzero(left_edge[height//3:2*height//3] > 10) > left_edge[height//3:2*height//3].size * 0.1,
                np.count_nonzero(right_edge[height//3:2*height//3] > 10) > right_edge[height//3:2*height//3].size * 0.1
            )
            
            lower_side = np.logical_or(
                np.count_nonzero(left_edge[2*height//3:] > 10) > left_edge[2*height//3:].size * 0.1,
                np.count_nonzero(right_edge[2*height//3:] > 10) > right_edge[2*height//3:].size * 0.1
            )
            
            if not upper_side:
                attributes['sleeve_length'] = 'sleeveless'
            elif upper_side and not mid_side:
                attributes['sleeve_length'] = 'short'
            elif upper_side and mid_side and not lower_side:
                attributes['sleeve_length'] = 'medium'
            else:
                attributes['sleeve_length'] = 'long'
            
            # Store generic length attribute for consistency
            attributes['length'] = attributes['sleeve_length']
            
            # Detect neckline type
            top_section = gray[:height//5, width//4:3*width//4]
            top_binary = cv2.threshold(top_section, 10, 255, cv2.THRESH_BINARY)[1]
            
            # Look at the profile of the top edge
            vertical_profile = np.sum(top_binary, axis=1) / 255
            
            # Check for V shape or U shape in neckline
            if len(vertical_profile) > 0 and vertical_profile[0] < vertical_profile.max() * 0.7:
                # Get the horizontal profile of top rows
                horizontal_profile = np.sum(top_binary[:10, :], axis=0) / 255
                
                if len(horizontal_profile) > 0:
                    # Check if center is lower (has fewer white pixels) than sides
                    center_val = horizontal_profile[len(horizontal_profile)//2]
                    side_avg = (horizontal_profile[len(horizontal_profile)//4] + 
                              horizontal_profile[3*len(horizontal_profile)//4]) / 2
                    
                    if center_val < side_avg * 0.7:
                        attributes['neckline'] = 'v-neck'
                    else:
                        attributes['neckline'] = 'scoop'
                else:
                    attributes['neckline'] = 'crew'
            else:
                attributes['neckline'] = 'crew'
            
            # Use segmentation info for better component detection if available
            if segments_info and 'regions' in segments_info:
                for region in segments_info['regions']:
                    position = region.get('position', '')
                    
                    if position == 'top':
                        if 'collar' not in attributes.get('components', []):
                            if 'components' not in attributes:
                                attributes['components'] = []
                            attributes['components'].append('collar')
                            
                    elif 'side' in position:
                        if 'sleeve' not in attributes.get('components', []):
                            if 'components' not in attributes:
                                attributes['components'] = []
                            attributes['components'].append('sleeve')
            
            return attributes
        except Exception as e:
            print(f"Error in _analyze_top: {str(e)}")
            return attributes

    def _analyze_skirt(self, img, attributes, segments_info=None):
        """
        Special analysis for skirts
        """
        try:
            height, width = img.shape[:2]
            
            # Detect length
            if height > width * 1.5:
                attributes['length'] = 'maxi'
            elif height > width * 1.1:
                attributes['length'] = 'midi'
            elif height > width * 0.8:
                attributes['length'] = 'knee'
            else:
                attributes['length'] = 'mini'
            
            # Detect silhouette/shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            # Calculate width at different heights
            top_width = np.count_nonzero(binary[height//6, :]) / width
            middle_width = np.count_nonzero(binary[height//2, :]) / width
            bottom_width = np.count_nonzero(binary[5*height//6, :]) / width
            
            width_ratio_bottom_to_top = bottom_width / top_width if top_width > 0 else 1
            width_ratio_middle_to_top = middle_width / top_width if top_width > 0 else 1
            
            # Determine silhouette based on width ratios
            if width_ratio_bottom_to_top > 1.5:
                attributes['silhouette'] = 'a-line'
            elif width_ratio_bottom_to_top < 0.9:
                attributes['silhouette'] = 'pencil'
            elif width_ratio_middle_to_top > 1.2:
                attributes['silhouette'] = 'pleated'
            else:
                attributes['silhouette'] = 'straight'
            
            # Use segmentation info for better component detection if available
            if segments_info and 'regions' in segments_info:
                for region in segments_info['regions']:
                    position = region.get('position', '')
                    
                    if position == 'top':
                        if 'waistband' not in attributes.get('components', []):
                            if 'components' not in attributes:
                                attributes['components'] = []
                            attributes['components'].append('waistband')
            
            return attributes
        except Exception as e:
            print(f"Error in _analyze_skirt: {str(e)}")
            return attributes

    def _analyze_dress(self, img, attributes, segments_info=None):
        """
        Special analysis for dresses
        """
        try:
            # Combines elements from both top and skirt analysis
            height, width = img.shape[:2]
            
            # Analyze top part (for neckline and sleeves)
            top_img = img[:height//3, :]
            temp_top_attrs = {}
            self._analyze_top(top_img, temp_top_attrs, segments_info)
            
            # Copy relevant attributes
            if 'neckline' in temp_top_attrs:
                attributes['neckline'] = temp_top_attrs['neckline']
            if 'sleeve_length' in temp_top_attrs:
                attributes['sleeve_length'] = temp_top_attrs['sleeve_length']
            
            # Analyze bottom part (for length and silhouette)
            bottom_img = img[height//3:, :]
            temp_bottom_attrs = {}
            self._analyze_skirt(bottom_img, temp_bottom_attrs, segments_info)
            
            # Copy relevant attributes
            if 'length' in temp_bottom_attrs:
                attributes['length'] = temp_bottom_attrs['length']
            if 'silhouette' in temp_bottom_attrs:
                attributes['silhouette'] = temp_bottom_attrs['silhouette']
            
            # Use segmentation info for better component detection if available
            if segments_info and 'regions' in segments_info:
                for region in segments_info['regions']:
                    position = region.get('position', '')
                    
                    if position == 'top':
                        if 'neckline' not in attributes.get('components', []):
                            if 'components' not in attributes:
                                attributes['components'] = []
                            attributes['components'].append('neckline')
                    
                    elif 'side' in position:
                        if 'sleeve' not in attributes.get('components', []):
                            if 'components' not in attributes:
                                attributes['components'] = []
                            attributes['components'].append('sleeve')
            
            return attributes
        except Exception as e:
            print(f"Error in _analyze_dress: {str(e)}")
            return attributes

    def _analyze_saree(self, img, attributes, segments_info=None):
        """
        Special analysis for sarees
        """
        try:
            # For sarees, focus on detecting border, pallu (decorative end piece) and color patterns
            height, width = img.shape[:2]
            
            # Sarees typically have rich colors
            attributes['color'] = self._detect_dominant_color(img, 'saree')
            
            # Detect if it has borders by checking edge density on the bottom
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            bottom_third = edges[2*height//3:, :]
            bottom_edge_density = np.count_nonzero(bottom_third) / bottom_third.size
            
            if bottom_edge_density > 0.1:
                attributes['border_type'] = 'decorated'
                if 'components' not in attributes:
                    attributes['components'] = []
                attributes['components'].append('decorated border')
            else:
                attributes['border_type'] = 'plain'
            
            # Check for pallu (decorative end piece usually on one side)
            left_side = edges[:, :width//4]
            right_side = edges[:, 3*width//4:]
            
            left_density = np.count_nonzero(left_side) / left_side.size
            right_density = np.count_nonzero(right_side) / right_side.size
            
            if max(left_density, right_density) > 0.1:
                if 'components' not in attributes:
                    attributes['components'] = []
                attributes['components'].append('pallu')
                attributes['drape_style'] = 'traditional'
            else:
                attributes['drape_style'] = 'modern'
            
            # Default attributes for saree if not set
            if 'length' not in attributes:
                attributes['length'] = 'long'
            if 'fit' not in attributes:
                attributes['fit'] = 'draped'
            
            return attributes
        except Exception as e:
            print(f"Error in _analyze_saree: {str(e)}")
            return attributes

    def _analyze_kurta(self, img, attributes, segments_info=None):
        """
        Special analysis for kurtas
        """
        try:
            height, width = img.shape[:2]
            
            # Analyze length
            if height > width * 1.5:
                attributes['length'] = 'long'
            elif height > width:
                attributes['length'] = 'medium'
            else:
                attributes['length'] = 'short'
            
            # Analyze neckline - kurtas often have special necklines
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            top_section = gray[:height//6, width//4:3*width//4]
            edges = cv2.Canny(top_section, 50, 150)
            
            # Check for detailed neckline patterns
            neckline_detail = np.count_nonzero(edges) / edges.size
            
            if neckline_detail > 0.15:
                attributes['neck_style'] = 'embroidered'
            else:
                # Check shape
                _, binary = cv2.threshold(top_section, 10, 255, cv2.THRESH_BINARY)
                horizontal_profile = np.sum(binary, axis=0) / binary.shape[0]
                
                if len(horizontal_profile) > 0:
                    center_val = horizontal_profile[len(horizontal_profile)//2]
                    side_avg = (horizontal_profile[len(horizontal_profile)//4] + 
                               horizontal_profile[3*len(horizontal_profile)//4]) / 2
                    
                    if center_val < side_avg * 0.7:
                        attributes['neck_style'] = 'v-neck'
                    else:
                        attributes['neck_style'] = 'round'
                else:
                    attributes['neck_style'] = 'round'
            
            # Analyze sleeve length using the same method as for tops
            temp_top_attrs = {}
            self._analyze_top(img, temp_top_attrs, segments_info)
            
            if 'sleeve_length' in temp_top_attrs:
                attributes['sleeve_length'] = temp_top_attrs['sleeve_length']
            
            # Set components
            if 'components' not in attributes:
                attributes['components'] = []
            
            attributes['components'].append('main kurta body')
            
            # Check for side slits, common in kurtas
            bottom_side_edges = edges[2*height//3:, :width//4]
            bottom_side_density = np.count_nonzero(bottom_side_edges) / bottom_side_edges.size
            
            if bottom_side_density > 0.1:
                attributes['components'].append('side slits')
            
            return attributes
        except Exception as e:
            print(f"Error in _analyze_kurta: {str(e)}")
            return attributes

    def _create_labeled_visualization(self, original_img, attributes, segments_info):
        """
        Create a visualization with bounding boxes and labels from segmentation info.
        The clothing type label is drawn centered at the top edge of the image.
        """
        labeled_img = original_img.copy()
        
        # Label the main clothing body if available
        if segments_info.get('main_body'):
            x, y, w, h = segments_info['main_body']['bbox']
            cv2.rectangle(labeled_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(labeled_img, "Main Body", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Label each segmented region
        for region in segments_info.get('regions', []):
            rx, ry, rw, rh = region['bbox']
            cv2.rectangle(labeled_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            pos = region.get('position', '')
            cv2.putText(labeled_img, pos, (rx, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
        # Draw the clothing type label centered at the top
        clothing_type = attributes.get('type', 'unknown')
        label = f"Type: {clothing_type}"
        # Get text size to center it
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
        image_width = labeled_img.shape[1]
        x_coord = (image_width - text_width) // 2
        y_coord = text_height + 10  # some margin from top
        cv2.putText(labeled_img, label, (x_coord, y_coord),
                    font, scale, (0, 0, 255), thickness)
        
        # NEW: Updated design for detailed item type labeling at bottom
        detailed_label = f"Category: {clothing_type.capitalize()}"
        (det_width, det_height), _ = cv2.getTextSize(detailed_label, font, scale, thickness)
        bottom_x = (image_width - det_width) // 2
        bottom_y = labeled_img.shape[0] - 20  # margin from bottom
        # Draw a filled rectangle behind the text for better readability
        cv2.rectangle(labeled_img,
                      (bottom_x - 5, bottom_y - det_height - 5),
                      (bottom_x + det_width + 5, bottom_y + 5),
                      (255, 255, 255),  # white background
                      thickness=-1)
        cv2.putText(labeled_img, detailed_label, (bottom_x, bottom_y),
                    font, scale, (0, 0, 0), thickness)  # black text
        
        return labeled_img

    def _analyze_lehenga(self, img, attributes, segments_info=None):
        """
        Special analysis for lehengas
        """
        try:
            height, width = img.shape[:2]
            
            # Detect skirt style
            if height > width * 1.5:
                attributes['skirt_style'] = 'a-line'
            elif height > width:
                attributes['skirt_style'] = 'circular'
            else:
                attributes['skirt_style'] = 'straight'
            
            # Detect choli style
            top_img = img[:height//3, :]
            temp_top_attrs = {}
            self._analyze_top(top_img, temp_top_attrs, segments_info)
            
            if 'sleeve_length' in temp_top_attrs:
                attributes['choli_style'] = temp_top_attrs['sleeve_length']
            
            # Detect dupatta style
            dupatta_img = img[height//3:, :]
            temp_dupatta_attrs = {}
            self._analyze_skirt(dupatta_img, temp_dupatta_attrs, segments_info)
            
            if 'length' in temp_dupatta_attrs:
                attributes['dupatta_style'] = temp_dupatta_attrs['length']
            
            # Use segmentation info for better component detection if available
            if segments_info and 'regions' in segments_info:
                for region in segments_info['regions']:
                    position = region.get('position', '')
                    
                    if position == 'top':
                        if 'choli' not in attributes.get('components', []):
                            if 'components' not in attributes:
                                attributes['components'] = []
                            attributes['components'].append('choli')
                    
                    elif 'side' in position:
                        if 'dupatta' not in attributes.get('components', []):
                            if 'components' not in attributes:
                                attributes['components'] = []
                            attributes['components'].append('dupatta')
            
            return attributes
        except Exception as e:
            print(f"Error in _analyze_lehenga: {str(e)}")
            return attributes

    def _determine_pants_vs_dress(self, img, segments_info=None):
        """
        Specialized method to distinguish between pants and dresses
        using geometric analysis for clear classification
        """
        try:
            height, width = img.shape[:2]
            # Convert to grayscale and binary for shape analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            
            # Get the main body contour
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Feature 1: Height to width ratio (pants are typically taller relative to width)
            aspect_ratio = h / w if w > 0 else 0
            
            # Feature 2: Look for leg separation (pants typically have separation)
            # Focus on bottom half of the image
            bottom_half = binary[height//2:, :]
            midline = width // 2
            
            # Scan the bottom half to find gaps around the midline
            gap_count = 0
            leg_separation = False
            
            for i in range(height//2, height, height//20):  # sample at intervals
                row = binary[i, :]
                # Look for pattern of white-black-white around middle (possible leg separation)
                if i < height-10 and sum(row[midline-10:midline+10]) < 200*20:  # detect dark gap
                    gap_count += 1
            
            # If multiple gaps detected, likely pants
            leg_separation = gap_count > 2
            
            # Feature 3: Bottom edge analysis (pants often have two distinct bottom edges)
            bottom_section = binary[3*height//4:, :]
            
            # Apply edge detection to find bottom contours
            bottom_edges = cv2.Canny(bottom_section, 50, 150)
            _, bottom_contours, _ = cv2.findContours(bottom_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count contours that could be pants legs
            valid_bottom_contours = 0
            for cnt in bottom_contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Ignore tiny contours
                    valid_bottom_contours += 1
            
            # Make decision: Combine evidence
            pants_score = 0
            
            # High aspect ratio suggests pants
            if aspect_ratio > 2.5:
                pants_score += 3
            elif aspect_ratio > 2.0:
                pants_score += 2
            elif aspect_ratio > 1.5:
                pants_score += 1
            
            # Leg separation strongly suggests pants
            if leg_separation:
                pants_score += 3
            
            # Multiple bottom contours suggest pants
            if valid_bottom_contours >= 2:
                pants_score += 2
            
            # Add more weight if segments_info has a very tall main body
            if segments_info and segments_info.get('main_body'):
                seg_x, seg_y, seg_w, seg_h = segments_info['main_body']['bbox']
                seg_aspect = seg_h / seg_w if seg_w > 0 else 0
                if seg_aspect > 2.2:
                    pants_score += 2
            
            # Return decision
            if pants_score >= 3:
                return 'pants'
            elif pants_score <= 0 and aspect_ratio < 2.0:
                return 'dress'
            else:
                return None  # Uncertain, let other methods decide
                
        except Exception as e:
            print(f"Error in pants vs dress determination: {str(e)}")
            return None
