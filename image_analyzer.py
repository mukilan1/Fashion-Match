import cv2
import numpy as np
import os
import json
import time
import random
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class ClothingAnalyzer:
    def __init__(self, cache_dir='static/analysis_cache'):
        # Store analysis results in a cache directory
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define clothing attributes we want to analyze
        self.clothing_attributes = {
            'description': None,
            'color': None,
            'pattern': None,
            'sleeve_length': None,
            'neckline': None,
            'style': None,
            'fabric': None,
            'occasion': None
        }
        
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
            'brown': ([0, 40, 40], [25, 200, 200])
        }
        
        # Default values to use instead of "unknown" - defined AFTER self.colors
        self.default_values = {
            'description': "A stylish piece of clothing",
            'color': self._get_random_color(),
            'pattern': "solid",
            'sleeve_length': "medium",
            'neckline': "crew",
            'style': "casual",
            'fabric': "cotton",
            'occasion': "casual/everyday"
        }
        
        # Style classification with more detailed mappings
        self.style_classifiers = {
            'formal': {'colors': ['black', 'white', 'grey', 'blue'], 'patterns': ['solid', 'pinstripe']},
            'casual': {'colors': ['blue', 'green', 'red', 'orange', 'yellow', 'grey'], 'patterns': ['solid', 'simple pattern', 'plaid']},
            'athletic': {'colors': ['blue', 'red', 'black', 'white', 'grey'], 'patterns': ['solid', 'simple pattern']},
            'bohemian': {'colors': ['orange', 'purple', 'red', 'brown', 'green'], 'patterns': ['complex pattern', 'floral', 'paisley']},
            'vintage': {'colors': ['brown', 'grey', 'orange', 'blue', 'green'], 'patterns': ['simple pattern', 'solid', 'floral']},
            'elegant': {'colors': ['black', 'white', 'red', 'purple', 'blue'], 'patterns': ['solid', 'sequined']}
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
            
        elif attribute == 'sleeve_length':
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
            processed_img = self._preprocess_image(img, image_path)
            
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
            
            # Fill in any missing attributes with traditional CV methods on the processed image
            if 'color' not in attributes or not attributes['color']:
                attributes['color'] = self._detect_dominant_color(processed_img)
                
            if 'pattern' not in attributes or not attributes['pattern']:
                attributes['pattern'] = self._detect_pattern(processed_img)
                
            if 'style' not in attributes or not attributes['style']:
                attributes['style'] = self._classify_style(processed_img)
                
            if 'sleeve_length' not in attributes or not attributes['sleeve_length']:
                attributes['sleeve_length'] = self._estimate_sleeve_length(processed_img)
                
            if 'neckline' not in attributes or not attributes['neckline']:
                attributes['neckline'] = self._estimate_neckline(processed_img)
                
            if 'fabric' not in attributes or not attributes['fabric']:
                attributes['fabric'] = self._determine_fabric(processed_img)
                
            if 'occasion' not in attributes or not attributes['occasion']:
                attributes['occasion'] = self._determine_occasion(attributes)
            
            # Replace any "unknown" values with better guesses
            for attr in self.clothing_attributes:
                if attr not in attributes or attributes[attr] == "unknown" or not attributes[attr]:
                    attributes[attr] = self._get_best_guess_for_attribute(attr, processed_img, attributes)
            
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
    
    def _detect_dominant_color(self, img):
        """
        Detect the dominant color in the clothing with enhanced accuracy
        Uses color quantization for more accurate dominant color detection
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
            'sleeve_length': None,
            'neckline': None,
            'style': None,
            'fabric': None
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
                
        # Sleeve length
        sleeve_types = {
            'sleeveless': ['sleeveless', 'strapless', 'spaghetti strap'],
            'short': ['short sleeve', 'short-sleeve', 'cap sleeve', 't-shirt'],
            'medium': ['elbow length', 'half sleeve', 'three-quarter', '3/4 sleeve'],
            'long': ['long sleeve', 'long-sleeve', 'full sleeve']
        }
        
        for length, keywords in sleeve_types.items():
            if any(keyword in description.lower() for keyword in keywords):
                features['sleeve_length'] = length
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
    
    def _classify_style(self, img):
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
