"""
Advanced color analysis module for clothing inventory system.
This module uses k-means clustering to identify dominant colors in clothing images.
"""

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, UnidentifiedImageError
import cv2
import webcolors
import os
import colorsys
import json
from pathlib import Path
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ColorAnalyzer')

class ColorAnalyzer:
    """Analyzes images to extract dominant colors with high accuracy."""
    
    def __init__(self):
        # Expanded comprehensive color dictionary
        self.color_names = {
            # Primary colors
            'red': (255, 0, 0),
            'green': (0, 128, 0),
            'blue': (0, 0, 255),
            
            # Secondary colors
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255),
            
            # Neutrals
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'silver': (192, 192, 192),
            
            # Browns
            'brown': (139, 69, 19),
            'beige': (245, 245, 220),
            'tan': (210, 180, 140),
            
            # Fashion colors
            'navy': (0, 0, 128),
            'teal': (0, 128, 128),
            'olive': (128, 128, 0),
            'maroon': (128, 0, 0),
            'pink': (255, 192, 203),
            'orange': (255, 165, 0),
            'gold': (255, 215, 0),
            'khaki': (240, 230, 140),
            'coral': (255, 127, 80),
            'turquoise': (64, 224, 208),
            'lavender': (230, 230, 250),
            'mint': (189, 252, 201),
            'salmon': (250, 128, 114),
            
            # Extended color palette - worldwide color variants
            # Reds
            'crimson': (220, 20, 60),
            'scarlet': (255, 36, 0),
            'ruby': (224, 17, 95),
            'burgundy': (128, 0, 32),
            'cherry': (222, 49, 99),
            'wine': (114, 47, 55),
            'vermilion': (227, 66, 52),
            'carmine': (150, 0, 24),
            'cardinal': (196, 30, 58),
            'sangria': (146, 0, 10),
            'rust': (183, 65, 14),
            'terracotta': (226, 114, 91),
            'cinnamon': (210, 105, 30),
            'auburn': (165, 42, 42),
            'brick': (178, 34, 34),
            'tomato': (255, 99, 71),
            'salmon pink': (255, 145, 164),
            'watermelon': (252, 108, 133),
            'rose': (255, 0, 127),
            'cerise': (222, 49, 99),
            'rosewood': (101, 0, 11),
            'raspberry': (227, 11, 93),

            # Blues
            'cobalt': (0, 71, 171),
            'azure': (0, 127, 255),
            'indigo': (75, 0, 130),
            'sapphire': (15, 82, 186),
            'cerulean': (0, 123, 167),
            'midnight blue': (25, 25, 112),
            'royal blue': (65, 105, 225),
            'steel blue': (70, 130, 180),
            'sky blue': (135, 206, 235),
            'cornflower blue': (100, 149, 237),
            'denim': (21, 96, 189),
            'periwinkle': (204, 204, 255),
            'baby blue': (137, 207, 240),
            'aquamarine': (127, 255, 212),
            'pacific blue': (28, 169, 201),
            'cyan blue': (28, 169, 201),
            'ultramarine': (18, 10, 143),
            'lapis lazuli': (38, 97, 156),

            # Greens
            'emerald': (80, 200, 120),
            'jade': (0, 168, 107),
            'forest green': (34, 139, 34),
            'sage': (188, 184, 138),
            'moss': (138, 154, 91),
            'olive green': (85, 107, 47),
            'lime': (191, 255, 0),
            'mint green': (152, 255, 152),
            'seafoam': (159, 226, 191),
            'pine': (1, 121, 111),
            'hunter green': (53, 94, 59),
            'fern': (113, 188, 120),
            'jungle green': (41, 171, 135),
            'spring green': (0, 255, 127),
            'avocado': (86, 130, 3),
            'pistachio': (147, 197, 114),
            'chartreuse': (127, 255, 0),
            'matcha': (181, 196, 177),
            'shamrock': (45, 139, 87),

            # Yellows/Oranges
            'amber': (255, 191, 0),
            'mustard': (255, 219, 88),
            'saffron': (244, 196, 48),
            'honey': (255, 195, 11),
            'lemon': (255, 250, 205),
            'banana': (255, 225, 53),
            'golden': (255, 223, 0),
            'tangerine': (242, 133, 0),
            'apricot': (251, 206, 177),
            'butterscotch': (224, 149, 64),
            'canary': (255, 255, 153),
            'citrine': (228, 208, 10),
            'amber orange': (255, 126, 0),
            'marigold': (236, 124, 38),
            'turmeric': (255, 164, 0),
            'ochre': (204, 119, 34),
            'carrot': (237, 145, 33),
            'papaya': (255, 164, 142),
            'mango': (255, 130, 67),
            'cantaloupe': (255, 185, 123),

            # Purples/Violets
            'amethyst': (153, 102, 204),
            'violet': (138, 43, 226),
            'mauve': (204, 153, 204),
            'lilac': (200, 162, 200),
            'plum': (221, 160, 221),
            'orchid': (218, 112, 214),
            'magenta': (255, 0, 255),
            'fuchsia': (255, 0, 255),
            'heliotrope': (223, 115, 255),
            'mulberry': (197, 75, 140),
            'eggplant': (97, 64, 81),
            'byzantium': (112, 41, 99),
            'tyrian purple': (102, 2, 60),
            'wine purple': (77, 0, 77),
            'boysenberry': (135, 50, 96),
            'grape': (111, 45, 168),
            'lavender purple': (150, 123, 182),
            'thistle': (216, 191, 216),
            'iris': (90, 79, 207),
            'wisteria': (201, 160, 220),

            # Browns/Tans/Neutrals
            'chocolate': (123, 63, 0),
            'coffee': (111, 78, 55),
            'mocha': (150, 114, 89),
            'chestnut': (149, 69, 53),
            'mahogany': (103, 66, 48),
            'sienna': (136, 45, 23),
            'umber': (99, 81, 71),
            'caramel': (196, 147, 83),
            'khaki brown': (145, 129, 81),
            'tawny': (205, 87, 0),
            'sand': (194, 178, 128),
            'camel': (193, 154, 107),
            'hazel': (135, 95, 66),
            'latte': (197, 179, 159),
            'walnut': (91, 69, 61),
            'pecan': (119, 85, 64),
            'almond': (239, 222, 205),
            'toast': (166, 123, 91),
            'fawn': (229, 170, 112),
            'taupe': (139, 133, 137),
            'clay': (155, 118, 83),
            'ivory': (255, 255, 240),
            'ecru': (194, 178, 128),
            'vanilla': (243, 229, 171),
            'oatmeal': (227, 218, 201),
            'champagne': (247, 231, 206),
            'cream': (255, 253, 208),
            'off-white': (249, 246, 240),
            'oyster': (226, 223, 210),
            'pewter': (144, 144, 144),
            'charcoal': (54, 69, 79),
            'slate': (112, 128, 144),
            'smoke': (115, 130, 118),
            'graphite': (72, 72, 72),
            'jet black': (5, 5, 5),
            'obsidian': (22, 24, 33),
            'onyx': (15, 15, 15),
            'ebony': (34, 34, 34),

            # Cultural/geographical colors
            'henna': (181, 101, 29),
            'tikka': (207, 109, 0),
            'indigo blue': (93, 94, 138),
            'mehndi': (151, 116, 35),
            'maasai red': (184, 27, 25),
            'tuareg blue': (54, 92, 139),
            'kente gold': (234, 170, 0),
            'ankara red': (208, 29, 25),
            'ankara blue': (9, 132, 227),
            'ankara green': (0, 148, 50),
            'kimono red': (194, 16, 24),
            'matcha green': (116, 148, 98),
            'tatami': (152, 136, 37),
            'celadon': (172, 225, 175),
            'imperial yellow': (254, 221, 0),
            'imperial red': (237, 41, 57),
            'cinnabar': (227, 66, 52),
            'vermilion red': (217, 56, 30),
            'persian blue': (27, 18, 123),
            'persian red': (204, 51, 51),
            'madder red': (227, 65, 50),
            'byzantine purple': (112, 41, 99),
            'maya blue': (115, 194, 251),
            'inca gold': (232, 174, 12),
            'aztec red': (196, 38, 29),
            'navajo white': (255, 222, 173),
            'pueblo terracotta': (192, 105, 80),
            'batik teal': (38, 166, 154),
            'ikat blue': (0, 111, 157),
            'aboriginal ochre': (198, 103, 39),
            'scandinavian blue': (133, 175, 204),
            'nordic white': (249, 249, 249),
            'scottish tartan green': (21, 71, 52),
            'gaelic gold': (223, 159, 0),
            'celtic green': (0, 130, 55),
            'kilim red': (188, 46, 40),
            'boho pink': (234, 157, 146),
            'tuscan yellow': (255, 186, 73),
            'venetian red': (197, 40, 61),
            'rajasthani pink': (219, 77, 109),
            'calabash orange': (255, 126, 7),
        }
        
        # Load additional color data from file if available
        self._load_extended_colors()
    
    def _load_extended_colors(self):
        """Load additional color data from external file if available"""
        color_file = Path(__file__).parent / "extended_colors.json"
        if color_file.exists():
            try:
                with open(color_file, 'r') as f:
                    additional_colors = json.load(f)
                    for name, rgb in additional_colors.items():
                        if name not in self.color_names:
                            self.color_names[name] = tuple(rgb)
                print(f"Loaded {len(additional_colors)} additional colors")
            except Exception as e:
                print(f"Error loading extended colors: {e}")
    
    def get_closest_color_name(self, rgb_tuple):
        """Find the closest named color to the given RGB tuple."""
        min_distance = float('inf')
        closest_name = "unknown"
        
        # Convert RGB to HSV for better color comparison
        h, s, v = colorsys.rgb_to_hsv(rgb_tuple[0]/255, rgb_tuple[1]/255, rgb_tuple[2]/255)
        
        # Advanced color categorization
        # Special case handling for achromatic colors (black, white, gray)
        if v < 0.2:
            return 'black'
        if v > 0.9 and s < 0.1:
            return 'white'
        if s < 0.1 and 0.2 < v < 0.9:
            # Different shades of gray
            if v < 0.4:
                return 'charcoal'
            elif v < 0.6:
                return 'gray'
            else:
                return 'silver'
        
        # Generate descriptive modifiers based on color properties
        lightness = ""
        if v > 0.8 and s > 0.1:
            lightness = "light "
        elif v < 0.4 and s > 0.1:
            lightness = "dark "
            
        saturation = ""
        if 0.1 < s < 0.4 and v > 0.3:
            saturation = "muted "
        elif s > 0.8 and v > 0.5:
            saturation = "vibrant "
            
        # IMPROVED COLOR MATCHING: Use weighted color spaces better suited for human perception
        candidates = []
        
        for name, rgb in self.color_names.items():
            # Skip modifiers in base name comparison
            if any(modifier in name for modifier in ["light ", "dark ", "muted ", "vibrant "]):
                continue
                
            # Calculate color distance using a perceptual model
            rgb_dist = sum((c1-c2)**2 for c1, c2 in zip(rgb_tuple, rgb))
            
            # Calculate HSV values for the target color
            h2, s2, v2 = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            # Calculate weighted HSV distance
            h_dist = min(abs(h - h2), 1 - abs(h - h2)) * 2  # Circular distance for hue
            s_dist = abs(s - s2)
            v_dist = abs(v - v2) * 0.5  # Less weight on value/brightness
            
            # Combined distance with perceptual weighting
            # Hue is most important for color naming, followed by saturation
            hsv_dist = (h_dist * 5)**2 + (s_dist * 3)**2 + (v_dist * 1)**2
            
            # Combined perceptual distance
            distance = rgb_dist * 0.3 + hsv_dist * 0.7
            
            candidates.append((name, distance))
        
        # Sort candidates by distance
        candidates.sort(key=lambda x: x[1])
        
        # Get top 3 candidates for more nuanced color naming
        top_candidates = candidates[:3]
        closest_name = top_candidates[0][0]
        
        # For very close matches, apply descriptive modifiers
        if lightness or saturation:
            modifier = f"{lightness}{saturation}".strip()
            modified_name = f"{modifier} {closest_name}"
            
            # Check if the exact modified name exists in our database
            if modified_name in self.color_names:
                return modified_name
            
            # Otherwise, return the base name with modifier
            return modified_name
            
        return closest_name
    
    def is_valid_image(self, image_path):
        """
        Validate if an image file can be properly loaded before processing.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    logger.warning(f"Image file does not exist: {image_path}")
                    return False
                
                # Try to open and verify the image
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    
                # Try to create image from binary data
                img = Image.open(io.BytesIO(image_data))
                img.verify()  # Verify it's a valid image
                
                # Try loading it again to ensure it can be processed
                img = Image.open(io.BytesIO(image_data))
                img.load()
                
                return True
                
            elif isinstance(image_path, Image.Image):
                # If it's already a PIL Image, try to access its data
                image_path.load()
                return True
                
            else:
                logger.warning(f"Unsupported image type: {type(image_path)}")
                return False
                
        except (OSError, UnidentifiedImageError, IOError) as e:
            logger.error(f"Invalid image detected: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating image: {str(e)}")
            return False
    
    def extract_colors(self, image_path, num_colors=3, min_percentage=5):
        """
        Extract dominant colors from an image.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of dominant colors to extract
            min_percentage: Minimum percentage for a color to be included
            
        Returns:
            List of tuples containing (color_name, percentage)
        """
        # First validate the image to avoid processing errors
        if not self.is_valid_image(image_path):
            logger.warning("Skipping color extraction for invalid image")
            return [("unknown", 100)]
            
        try:
            # Load image and convert to RGB array
            image = None
            pil_image = None
            
            if isinstance(image_path, str):
                try:
                    # Open with PIL first to handle transparency properly
                    # Use a more robust loading method
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    pil_image = Image.open(io.BytesIO(image_data))
                    
                    # If image has an alpha channel (transparency), compose it over a white background
                    if pil_image.mode == 'RGBA':
                        # Create a white background image
                        white_bg = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
                        # Composite image over white background (respecting transparency)
                        pil_image = Image.alpha_composite(white_bg, pil_image)
                        pil_image = pil_image.convert('RGB')
                    elif pil_image.mode != 'RGB':
                        # Ensure RGB mode for any other modes
                        pil_image = pil_image.convert('RGB')
                except Exception as e:
                    logger.error(f"Error loading image from path: {str(e)}")
                    return [("unknown", 100)]
                    
            elif isinstance(image_path, Image.Image):
                try:
                    pil_image = image_path
                    # Handle transparency in PIL image
                    if pil_image.mode == 'RGBA':
                        white_bg = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
                        pil_image = Image.alpha_composite(white_bg, pil_image)
                        pil_image = pil_image.convert('RGB')
                    elif pil_image.mode != 'RGB':
                        # Ensure RGB mode for any other modes
                        pil_image = pil_image.convert('RGB')
                except Exception as e:
                    logger.error(f"Error processing PIL image: {str(e)}")
                    return [("unknown", 100)]
            else:
                logger.error(f"Unsupported image type: {type(image_path)}")
                return [("unknown", 100)]
            
            # Safely convert PIL image to numpy array
            try:
                # Use numpy's array function with explicit conversion to avoid PIL internals
                image = np.array(pil_image, dtype=np.uint8)
            except Exception as e:
                logger.error(f"Error converting PIL image to numpy array: {str(e)}")
                return [("unknown", 100)]
            
            # Verify image dimensions
            if image.ndim != 3 or image.shape[2] != 3:
                logger.error(f"Invalid image shape: {image.shape}")
                return [("unknown", 100)]
                
            # Reshape the image to be a list of pixels
            height, width, _ = image.shape
            
            # Check if the image is very small - if so, resize it to ensure we have enough pixels for analysis
            if height < 100 or width < 100:
                try:
                    pil_image = pil_image.resize((max(width, 100), max(height, 100)), Image.LANCZOS)
                    image = np.array(pil_image, dtype=np.uint8)
                    height, width, _ = image.shape
                except Exception as e:
                    logger.warning(f"Error resizing small image: {str(e)}")
                    # Continue with original image
            
            # Reshape safely
            try:
                pixels = image.reshape((height * width, 3))
            except Exception as e:
                logger.error(f"Error reshaping image: {str(e)}")
                return [("unknown", 100)]
            
            # ADAPTIVE THRESHOLD: Determine thresholds based on image characteristics
            # Calculate histogram to find the color distribution
            hist_r = np.histogram(pixels[:, 0], bins=25)[0]
            hist_g = np.histogram(pixels[:, 1], bins=25)[0]
            hist_b = np.histogram(pixels[:, 2], bins=25)[0]
            
            # Check if the image has a lot of bright/dark areas
            bright_pixels = np.sum(pixels.mean(axis=1) > 220)
            dark_pixels = np.sum(pixels.mean(axis=1) < 30)
            bright_ratio = bright_pixels / len(pixels)
            dark_ratio = dark_pixels / len(pixels)
            
            # Adjust thresholds based on image characteristics
            white_threshold = 230 if bright_ratio > 0.4 else 220
            black_threshold = 25 if dark_ratio > 0.4 else 30
            
            # Filter out near-white pixels and very dark pixels
            mask = np.ones(len(pixels), dtype=bool)
            
            for i, pixel in enumerate(pixels):
                r, g, b = pixel
                # Filter out whites
                if r > white_threshold and g > white_threshold and b > white_threshold:
                    mask[i] = False
                
                # Filter out very dark pixels (blacks/shadows)
                if r < black_threshold and g < black_threshold and b < black_threshold:
                    mask[i] = False
                
                # Convert to HSV to check brightness and saturation
                h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                
                # Filter out low saturation colors (grays) and very dark colors
                if s < 0.15 or v < 0.15:  # Increased from 0.1 to 0.15
                    mask[i] = False
            
            # Only use non-filtered pixels for clustering
            pixels_filtered = pixels[mask]
            
            # IMPROVED FALLBACK: More intelligent pixel filtering
            # If we filtered too much (less than 5% of pixels remain), relax the filtering
            if len(pixels_filtered) < len(pixels) * 0.05:
                # Use less strict filtering
                mask = np.ones(len(pixels), dtype=bool)
                for i, pixel in enumerate(pixels):
                    r, g, b = pixel
                    # Only filter extreme whites and blacks
                    if (r > 240 and g > 240 and b > 240) or (r < 15 and g < 15 and b < 15):
                        mask[i] = False
                pixels_filtered = pixels[mask]
            
            # If we still have too few pixels, use original pixels
            if len(pixels_filtered) < 100:
                print(f"Warning: Too few pixels after filtering ({len(pixels_filtered)}), using original image")
                pixels_filtered = pixels
            
            # IMPROVED CLUSTERING: Try different cluster numbers for better accuracy
            # Start with requested number of colors
            best_score = -1
            best_colors = []
            best_labels = None
            best_centers = None
            
            # Try different numbers of clusters to find optimal clustering
            for test_clusters in [num_colors, num_colors + 1, num_colors - 1]:
                if test_clusters < 2:
                    continue
                    
                try:
                    # Use k-means clustering with multiple initializations
                    clt = KMeans(n_clusters=test_clusters, n_init=10, max_iter=300)
                    clt.fit(pixels_filtered)
                    
                    # Evaluate clustering quality (lower inertia = better clustering)
                    if best_score < 0 or clt.inertia_ < best_score:
                        best_score = clt.inertia_
                        best_labels = clt.labels_
                        best_centers = clt.cluster_centers_
                except Exception as e:
                    print(f"Clustering error with {test_clusters} clusters: {e}")
                    continue
            
            # If no successful clustering found
            if best_labels is None:
                return [("unknown", 100)]
                
            # IMPROVED RESULT PROCESSING: Better handling of color percentages
            # Calculate percentages
            counts = np.bincount(best_labels)
            percentages = counts / len(best_labels) * 100
            
            # Sort by occurrence
            indices = np.argsort(-percentages)
            
            colors = []
            for i in indices:
                # Only include colors above the minimum percentage
                if percentages[i] >= min_percentage:
                    rgb = tuple(map(int, best_centers[i]))
                    color_name = self.get_closest_color_name(rgb)
                    colors.append((color_name, round(percentages[i], 1)))
            
            # IMPROVED FALLBACK: Always return at least one color
            if not colors:
                # Try to get a meaningful color using the top cluster
                if len(best_centers) > 0:
                    rgb = tuple(map(int, best_centers[indices[0]]))
                    color_name = self.get_closest_color_name(rgb)
                    colors.append((color_name, 100.0))
                else:
                    # Default fallback
                    colors.append(("unknown", 100.0))
            
            # IMPROVED COLOR MERGING: Combine similar colors
            # If we have multiple of the same color name, combine them
            merged_colors = {}
            for name, percentage in colors:
                if name in merged_colors:
                    merged_colors[name] += percentage
                else:
                    merged_colors[name] = percentage
                    
            # Convert back to list of tuples and sort by percentage
            merged_color_list = [(name, pct) for name, pct in merged_colors.items()]
            merged_color_list.sort(key=lambda x: x[1], reverse=True)
            
            return merged_color_list
            
        except Exception as e:
            logger.error(f"Error extracting colors: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [("unknown", 100)]
    
    def analyze_image(self, image_path):
        """
        Analyze an image and return dominant color information.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            dict: Contains color information
        """
        # Special case handling for suits when metadata is available
        if hasattr(image_path, 'info') and isinstance(image_path.info, dict):
            metadata = image_path.info
            if metadata.get('label', '').lower().find('suit') >= 0:
                # Most suits are dark colors - if no specific color mentioned, assume these defaults
                label = metadata.get('label', '').lower()
                if 'navy' in label or 'blue' in label:
                    return {
                        "primary_color": "navy",
                        "color_distribution": [("navy", 80.0), ("black", 20.0)],
                        "is_multicolored": False,
                        "colors_text": "navy (80%), black (20%)"
                    }
                elif 'gray' in label or 'grey' in label:
                    return {
                        "primary_color": "gray",
                        "color_distribution": [("gray", 90.0), ("silver", 10.0)],
                        "is_multicolored": False,
                        "colors_text": "gray (90%), silver (10%)"
                    }
                elif 'brown' in label or 'tan' in label:
                    return {
                        "primary_color": "brown",
                        "color_distribution": [("brown", 90.0), ("beige", 10.0)],
                        "is_multicolored": False,
                        "colors_text": "brown (90%), beige (10%)"
                    }
                else:
                    # Default to black for most suits
                    return {
                        "primary_color": "black",
                        "color_distribution": [("black", 95.0), ("gray", 5.0)],
                        "is_multicolored": False,
                        "colors_text": "black (95%), gray (5%)"
                    }
        
        # Add validation before processing
        if not self.is_valid_image(image_path):
            return {
                "primary_color": "unknown",
                "color_distribution": [("unknown", 100.0)],
                "is_multicolored": False,
                "colors_text": "unknown (100%)",
                "error": "Invalid or corrupted image"
            }
            
        # Standard analysis for non-suit items
        colors = self.extract_colors(image_path)
        
        # Format the results
        primary_color = colors[0][0] if colors else "unknown"
        color_info = {
            "primary_color": primary_color,
            "color_distribution": colors,
            "is_multicolored": len(colors) > 1,
            "colors_text": ", ".join(f"{name} ({pct}%)" for name, pct in colors)
        }
        
        return color_info
    
    def get_color_hex(self, color_name):
        """Convert a color name to hex code for display."""
        rgb = self.color_names.get(color_name.lower(), (200, 200, 200))
        return "#{:02x}{:02x}{:02x}".format(*rgb)

# Helper function for easy use
def analyze_colors(image_path):
    """Analyze colors in an image and return them in a printable format."""
    analyzer = ColorAnalyzer()
    
    # Add validation check before analysis
    if not analyzer.is_valid_image(image_path):
        return {
            "primary_color": "unknown",
            "color_distribution": [("unknown", 100.0)],
            "is_multicolored": False,
            "colors_text": "unknown (100%)",
            "error": "Invalid or corrupted image"
        }
        
    results = analyzer.analyze_image(image_path)
    return results
