"""
Advanced color analysis module for clothing inventory system.
This module uses k-means clustering to identify dominant colors in clothing images.
"""

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import webcolors
import os
import colorsys

class ColorAnalyzer:
    """Analyzes images to extract dominant colors with high accuracy."""
    
    def __init__(self):
        # Basic color names mapped to RGB values
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
        }
    
    def get_closest_color_name(self, rgb_tuple):
        """Find the closest named color to the given RGB tuple."""
        min_distance = float('inf')
        closest_name = "unknown"
        
        # Convert RGB to HSV for better color comparison
        h, s, v = colorsys.rgb_to_hsv(rgb_tuple[0]/255, rgb_tuple[1]/255, rgb_tuple[2]/255)
        
        # Special case handling
        if v < 0.25:  # Increased from 0.2 to better detect blacks
            return 'black'
        if v > 0.85 and s < 0.15:  # Adjusted thresholds
            return 'white'
        if s < 0.15 and 0.25 < v < 0.85:  # Better gray detection
            return 'gray'
        
        # IMPROVED COLOR MATCHING: Use color spaces better suited for human perception
        for name, rgb in self.color_names.items():
            # Skip black and white when saturation is reasonable
            if name in ['black', 'white', 'gray'] and s > 0.3:
                continue
            
            # Calculate color distance in a way that's closer to human perception
            # Use a weighted combination of RGB distance and HSV distance
            rgb_dist = sum((c1-c2)**2 for c1, c2 in zip(rgb_tuple, rgb))
            
            # Calculate HSV values for the target color
            h2, s2, v2 = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            # Calculate HSV distance, with more weight on hue
            h_dist = min(abs(h - h2), 1 - abs(h - h2)) * 2  # Circular distance for hue
            s_dist = abs(s - s2)
            v_dist = abs(v - v2) * 0.5  # Less weight on value/brightness
            hsv_dist = (h_dist**2 + s_dist**2 + v_dist**2) * 100
            
            # Combined distance (weighted)
            distance = rgb_dist * 0.5 + hsv_dist * 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name
    
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
        try:
            # Load image and convert to RGB array
            image = None
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    return [("unknown", 100)]
                
                # Open with PIL first to handle transparency properly
                pil_image = Image.open(image_path)
                
                # If image has an alpha channel (transparency), compose it over a white background
                if pil_image.mode == 'RGBA':
                    # Create a white background image
                    white_bg = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
                    # Composite image over white background (respecting transparency)
                    pil_image = Image.alpha_composite(white_bg, pil_image)
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array for processing
                image = np.array(pil_image)
            elif isinstance(image_path, Image.Image):
                pil_image = image_path
                # Handle transparency in PIL image
                if pil_image.mode == 'RGBA':
                    white_bg = Image.new('RGBA', pil_image.size, (255, 255, 255, 255))
                    pil_image = Image.alpha_composite(white_bg, pil_image)
                    pil_image = pil_image.convert('RGB')
                image = np.array(pil_image)
            else:
                return [("unknown", 100)]
            
            # Reshape the image to be a list of pixels
            height, width, _ = image.shape
            
            # Check if the image is very small - if so, resize it to ensure we have enough pixels for analysis
            if height < 100 or width < 100:
                pil_image = pil_image.resize((max(width, 100), max(height, 100)), Image.LANCZOS)
                image = np.array(pil_image)
                height, width, _ = image.shape
            
            pixels = image.reshape((height * width, 3))
            
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
            print(f"Error extracting colors: {str(e)}")
            import traceback
            traceback.print_exc()
            return [("unknown", 100)]
    
    def analyze_image(self, image_path):
        """
        Analyze an image and return dominant color information.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            dict: Contains color information
        """
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
    results = analyzer.analyze_image(image_path)
    return results
