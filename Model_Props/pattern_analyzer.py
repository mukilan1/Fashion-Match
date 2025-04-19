"""
Pattern analysis module for clothing inventory system.
Identifies patterns like solid, striped, checkered, floral, polka dots, etc. in clothing items.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
import os
import math
from typing import Dict, Union, List, Tuple, Optional
import logging
from scipy import signal

class PatternAnalyzer:
    """Specialized analyzer for detecting patterns in clothing items."""
    
    def __init__(self, use_clip=True, use_vqa=True):
        """Initialize pattern analyzer with multiple detection methods for better accuracy."""
        # Configure logging
        self.logger = logging.getLogger('PatternAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # Initialize CLIP model if requested
        if use_clip:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.logger.info("CLIP model initialized successfully")
                self.use_clip = True
            except Exception as e:
                self.logger.error(f"Failed to initialize CLIP model: {e}")
                self.use_clip = False
        else:
            self.use_clip = False
        
        # Initialize VQA model if requested
        if use_vqa:
            try:
                self.vqa_model = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
                self.logger.info("VQA model initialized successfully")
                self.use_vqa = True
            except Exception as e:
                self.logger.error(f"Failed to initialize VQA model: {e}")
                self.use_vqa = False
        else:
            self.use_vqa = False
        
        # Define pattern types with details for better classification
        self.pattern_types = {
            "solid": {
                "name": "Solid",
                "description": "Uniform color or texture with no visible pattern",
                "examples": ["solid color shirt", "plain t-shirt", "single color dress"],
                "keywords": ["solid", "plain", "uniform", "single color", "block color", "solid color"],
                "detection_methods": ["color_variance", "edge_density", "ml"]
            },
            "striped": {
                "name": "Striped",
                "description": "Regular alternating lines of different colors",
                "examples": ["striped shirt", "pinstripe suit", "zebra pattern"],
                "keywords": ["stripe", "striped", "lines", "pinstripe", "vertical lines", "horizontal lines"],
                "detection_methods": ["fourier", "hough", "ml"]
            },
            "checkered": {
                "name": "Checkered",
                "description": "Grid pattern with squares of alternating colors",
                "examples": ["checkered shirt", "plaid dress", "gingham pattern"],
                "keywords": ["check", "checkered", "plaid", "tartan", "gingham", "grid", "squares"],
                "detection_methods": ["fourier", "hough", "ml"]
            },
            "floral": {
                "name": "Floral",
                "description": "Designs featuring flowers, leaves, and plant motifs",
                "examples": ["floral dress", "flower print shirt", "botanical pattern"],
                "keywords": ["floral", "flower", "flowers", "botanical", "leafy", "plant", "nature"],
                "detection_methods": ["ml"]
            },
            "polka_dot": {
                "name": "Polka Dot",
                "description": "Regular pattern of filled circles",
                "examples": ["polka dot dress", "dotted blouse", "spotted fabric"],
                "keywords": ["polka dot", "dotted", "spots", "circles", "dot", "dots"],
                "detection_methods": ["blob", "ml"]
            },
            "geometric": {
                "name": "Geometric",
                "description": "Repeating geometric shapes like triangles, diamonds, etc.",
                "examples": ["geometric print shirt", "diamond pattern", "abstract shapes"],
                "keywords": ["geometric", "shapes", "triangles", "diamonds", "hexagons", "abstract"],
                "detection_methods": ["fourier", "ml"]
            },
            "animal_print": {
                "name": "Animal Print",
                "description": "Patterns mimicking animal skin or fur",
                "examples": ["leopard print dress", "zebra pattern", "snake skin design"],
                "keywords": ["animal print", "leopard", "zebra", "snake", "tiger", "cheetah", "giraffe"],
                "detection_methods": ["ml"]
            },
            "abstract": {
                "name": "Abstract",
                "description": "Non-representational patterns with irregular shapes",
                "examples": ["abstract design shirt", "modern art pattern", "random shapes"],
                "keywords": ["abstract", "irregular", "random", "artistic", "modern", "non-representational"],
                "detection_methods": ["ml"]
            },
            "camouflage": {
                "name": "Camouflage",
                "description": "Military-style disruptive pattern with irregular patches",
                "examples": ["camo pants", "camouflage jacket", "military pattern"],
                "keywords": ["camouflage", "camo", "military", "army", "patches", "disruptive pattern"],
                "detection_methods": ["ml", "texture"]
            },
            "paisley": {
                "name": "Paisley",
                "description": "Teardrop-shaped pattern with curved designs",
                "examples": ["paisley shirt", "bandana pattern", "teardrop design"],
                "keywords": ["paisley", "teardrop", "swirl", "bandana"],
                "detection_methods": ["ml"]
            }
        }

    def _load_image(self, image_path: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Load and validate image from file path or PIL Image."""
        try:
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    self.logger.error(f"Image file not found: {image_path}")
                    return None
                return Image.open(image_path).convert('RGB')
            elif isinstance(image_path, Image.Image):
                return image_path.convert('RGB')
            else:
                self.logger.error(f"Unsupported image type: {type(image_path)}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None

    def _analyze_with_clip(self, image: Image.Image) -> Dict[str, float]:
        """Use CLIP model to analyze pattern type."""
        if not self.use_clip:
            return {pattern_type: 1.0/len(self.pattern_types) for pattern_type in self.pattern_types}
        
        try:
            # Create detailed prompts for better pattern classification
            prompts = []
            categories = []
            
            for pattern_type, details in self.pattern_types.items():
                # Base prompt with description
                base_prompt = f"This clothing has a {details['name'].lower()} pattern: {details['description']}"
                prompts.append(base_prompt)
                categories.append(pattern_type)
                
                # Example-based prompt
                examples_prompt = f"This is a {details['name'].lower()} pattern fabric, like: {', '.join(details['examples'])}"
                prompts.append(examples_prompt)
                categories.append(pattern_type)
                
                # Additional specific prompt for each pattern type
                if pattern_type == "solid":
                    prompts.append("This fabric has a uniform color with no pattern")
                    categories.append(pattern_type)
                elif pattern_type == "striped":
                    prompts.append("This fabric has parallel lines of different colors")
                    categories.append(pattern_type)
                elif pattern_type == "checkered":
                    prompts.append("This fabric has a grid pattern with squares of different colors")
                    categories.append(pattern_type)
                elif pattern_type == "floral":
                    prompts.append("This fabric has a pattern with flowers and plant designs")
                    categories.append(pattern_type)
                elif pattern_type == "polka_dot":
                    prompts.append("This fabric has a pattern of small circles or dots")
                    categories.append(pattern_type)
            
            # Process inputs
            inputs = self.clip_processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0].numpy()
            
            # Combine scores for the same pattern type (from different prompts)
            pattern_scores = {}
            for i, pattern_type in enumerate(categories):
                if pattern_type not in pattern_scores:
                    pattern_scores[pattern_type] = probs[i]
                else:
                    # Average with existing score
                    pattern_scores[pattern_type] = (pattern_scores[pattern_type] + probs[i]) / 2
            
            # Normalize scores
            total = sum(pattern_scores.values())
            if total > 0:
                for pattern_type in pattern_scores:
                    pattern_scores[pattern_type] = float(pattern_scores[pattern_type] / total)
            
            return pattern_scores
            
        except Exception as e:
            self.logger.error(f"CLIP analysis error: {str(e)}")
            return {pattern_type: 1.0/len(self.pattern_types) for pattern_type in self.pattern_types}

    def _analyze_with_vqa(self, image: Image.Image) -> Dict[str, float]:
        """Use Visual Question Answering to determine pattern type."""
        if not self.use_vqa:
            return {pattern_type: 1.0/len(self.pattern_types) for pattern_type in self.pattern_types}
            
        try:
            # Ask direct questions about pattern
            questions = [
                "What pattern does this clothing item have?",
                "What type of pattern is visible on this fabric?",
                "Is this fabric solid, striped, checked, floral, or polka dot?",
                "Describe the pattern on this clothing item."
            ]
            
            # Initialize scores
            pattern_scores = {pattern_type: 0.1 for pattern_type in self.pattern_types}
            
            for question in questions:
                result = self.vqa_model(image, question, top_k=5)
                if not result:
                    continue
                    
                for item in result:
                    answer = item['answer'].lower()
                    confidence = item['score']
                    
                    # Check each pattern type for matches in the answer
                    for pattern_type, details in self.pattern_types.items():
                        if any(keyword.lower() in answer for keyword in details['keywords']):
                            pattern_scores[pattern_type] += confidence * 0.25
                            break
            
            # Ask more specific follow-up questions for confirmation
            specific_questions = [
                "Does this fabric have a solid color with no pattern?",
                "Does this fabric have stripes?",
                "Does this fabric have a checkered or plaid pattern?",
                "Does this fabric have a floral pattern?",
                "Does this fabric have polka dots?"
            ]
            
            specific_patterns = ["solid", "striped", "checkered", "floral", "polka_dot"]
            
            for i, question in enumerate(specific_questions):
                result = self.vqa_model(image, question)
                if result and len(result) > 0:
                    answer = result[0]['answer'].lower()
                    confidence = result[0]['score']
                    
                    if "yes" in answer:
                        pattern_type = specific_patterns[i] if i < len(specific_patterns) else None
                        if pattern_type:
                            pattern_scores[pattern_type] += confidence * 0.5
            
            # Normalize scores
            total = sum(pattern_scores.values())
            if total > 0:
                for pattern_type in pattern_scores:
                    pattern_scores[pattern_type] = float(pattern_scores[pattern_type] / total)
            
            return pattern_scores
            
        except Exception as e:
            self.logger.error(f"VQA analysis error: {str(e)}")
            return {pattern_type: 1.0/len(self.pattern_types) for pattern_type in self.pattern_types}

    def _analyze_with_cv(self, image: Image.Image) -> Dict[str, float]:
        """Use computer vision techniques to analyze pattern type."""
        try:
            # Convert PIL image to OpenCV format
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Resize to consistent size
            img_resized = cv2.resize(img_cv, (300, 300))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Initialize scores
            pattern_scores = {pattern_type: 0.0 for pattern_type in self.pattern_types}
            
            # Calculate features for pattern detection
            features = {}
            
            # 1. Color variance (low for solid, high for patterns)
            features['color_variance'] = self._calculate_color_variance(img_resized)
            
            # 2. Edge density (low for solid, high for patterns with distinct edges)
            features['edge_density'] = self._calculate_edge_density(gray)
            
            # 3. Fourier transform for frequency analysis (useful for stripes and checks)
            features['fourier_stats'] = self._analyze_fourier(gray)
            
            # 4. Hough lines detection (for stripes)
            features['hough_lines'] = self._detect_lines(gray)
            
            # 5. Blob detection (for polka dots)
            features['blob_stats'] = self._detect_blobs(gray)
            
            # Compute scores for each pattern type based on features
            
            # Solid: low color variance, low edge density
            if features['color_variance'] < 0.05:
                pattern_scores['solid'] += 0.5
            if features['edge_density'] < 0.05:
                pattern_scores['solid'] += 0.3
            
            # Striped: many lines detected, distinctive frequency in one direction
            if features['hough_lines']['line_count'] > 10 and features['hough_lines']['parallel_ratio'] > 0.6:
                pattern_scores['striped'] += 0.5
            if features['fourier_stats']['directional_ratio'] > 2.5:
                pattern_scores['striped'] += 0.3
            
            # Checkered: grid pattern in Fourier domain, many lines in multiple directions
            if features['fourier_stats']['grid_strength'] > 0.4:
                pattern_scores['checkered'] += 0.5
            if features['hough_lines']['perpendicular_ratio'] > 0.3:
                pattern_scores['checkered'] += 0.3
            
            # Polka dot: many blobs detected with consistent size
            if features['blob_stats']['blob_count'] > 10 and features['blob_stats']['size_consistency'] > 0.7:
                pattern_scores['polka_dot'] += 0.7
            
            # Adjust score for solid if other patterns have low scores
            other_pattern_score = max([v for k, v in pattern_scores.items() if k != 'solid'])
            if other_pattern_score < 0.3:
                pattern_scores['solid'] += 0.2
            
            # Normalize scores
            total = sum(pattern_scores.values())
            if total > 0:
                for pattern_type in pattern_scores:
                    pattern_scores[pattern_type] = float(pattern_scores[pattern_type] / total)
            
            return pattern_scores
            
        except Exception as e:
            self.logger.error(f"Computer vision analysis error: {str(e)}")
            return {pattern_type: 1.0/len(self.pattern_types) for pattern_type in self.pattern_types}

    def _calculate_color_variance(self, img) -> float:
        """Calculate variance in color across the image."""
        # Split into channels and calculate variance
        b, g, r = cv2.split(img)
        b_var = np.var(b) / (255 * 255)
        g_var = np.var(g) / (255 * 255)
        r_var = np.var(r) / (255 * 255)
        
        # Return average variance across channels
        return float((b_var + g_var + r_var) / 3)

    def _calculate_edge_density(self, gray_img) -> float:
        """Calculate edge density using Canny edge detector."""
        edges = cv2.Canny(gray_img, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (gray_img.shape[0] * gray_img.shape[1])
        return float(edge_ratio)

    def _analyze_fourier(self, gray_img) -> Dict:
        """Analyze frequency domain using Fast Fourier Transform."""
        # Apply FFT
        f_transform = np.fft.fft2(gray_img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # Get dimensions
        rows, cols = gray_img.shape
        center_row, center_col = rows // 2, cols // 2
        
        # Analyze frequency components
        horizontal_sum = np.sum(magnitude[center_row-5:center_row+5, :])
        vertical_sum = np.sum(magnitude[:, center_col-5:center_col+5])
        directional_ratio = max(horizontal_sum, vertical_sum) / (min(horizontal_sum, vertical_sum) + 0.001)
        
        # Detect grid pattern (checkered)
        # For a grid, we expect strong components at 90 degrees to each other
        quadrant1 = np.sum(magnitude[center_row-30:center_row-5, center_col+5:center_col+30])
        quadrant2 = np.sum(magnitude[center_row-30:center_row-5, center_col-30:center_col-5])
        diagonal_strength = (quadrant1 + quadrant2) / 2
        grid_strength = min(horizontal_sum, vertical_sum) / (magnitude.sum() + 0.001) * diagonal_strength
        
        return {
            "directional_ratio": float(directional_ratio),
            "grid_strength": float(grid_strength),
            "horizontal_energy": float(horizontal_sum),
            "vertical_energy": float(vertical_sum)
        }

    def _detect_lines(self, gray_img) -> Dict:
        """Detect lines using Hough transform."""
        edges = cv2.Canny(gray_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        line_count = 0
        horizontal_count = 0
        vertical_count = 0
        
        if lines is not None:
            line_count = len(lines)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle of line
                if x2 - x1 == 0:  # Vertical line
                    vertical_count += 1
                else:
                    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                    if angle < 20 or angle > 160:  # Near horizontal
                        horizontal_count += 1
                    elif 70 < angle < 110:  # Near vertical
                        vertical_count += 1
        
        # Calculate metrics
        line_count = len(lines) if lines is not None else 0
        parallel_ratio = max(horizontal_count, vertical_count) / (line_count + 0.001)
        perpendicular_ratio = min(horizontal_count, vertical_count) / (line_count + 0.001)
        
        return {
            "line_count": line_count,
            "horizontal_count": horizontal_count,
            "vertical_count": vertical_count,
            "parallel_ratio": float(parallel_ratio),
            "perpendicular_ratio": float(perpendicular_ratio)
        }

    def _detect_blobs(self, gray_img) -> Dict:
        """Detect blobs (for polka dots)."""
        # Setup blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 50
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.8
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray_img)
        
        # Count blobs and compute statistics
        blob_count = len(keypoints)
        size_consistency = 0.0
        
        if blob_count > 1:
            sizes = [kp.size for kp in keypoints]
            mean_size = np.mean(sizes)
            size_std = np.std(sizes)
            size_consistency = 1.0 - min(1.0, size_std / (mean_size + 0.001))
        
        return {
            "blob_count": blob_count,
            "size_consistency": float(size_consistency)
        }

    def _analyze_texture_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract texture features using Haralick and LBP."""
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Resize for consistent analysis
            gray = cv2.resize(gray, (256, 256))
            
            # 1. Calculate Haralick texture features
            # First, we need to quantize the image to fewer gray levels (typically 8 or 16)
            gray_quantized = (gray // 32).astype(np.uint8)
            
            haralick_features = []
            distances = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            # Skip actual Haralick computation if OpenCV doesn't have the function
            # But calculate approximations
            
            # 2. Local Binary Pattern features
            # Create simple LBP-like texture descriptor
            h, w = gray.shape
            lbp_image = np.zeros((h-2, w-2), dtype=np.uint8)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray[i, j]
                    code = 0
                    # Check 8 neighbors
                    if gray[i-1, j-1] >= center: code |= 1 << 0
                    if gray[i-1, j] >= center: code |= 1 << 1
                    if gray[i-1, j+1] >= center: code |= 1 << 2
                    if gray[i, j+1] >= center: code |= 1 << 3
                    if gray[i+1, j+1] >= center: code |= 1 << 4
                    if gray[i+1, j] >= center: code |= 1 << 5
                    if gray[i+1, j-1] >= center: code |= 1 << 6
                    if gray[i, j-1] >= center: code |= 1 << 7
                    lbp_image[i-1, j-1] = code
            
            # Calculate LBP histogram
            hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=[0, 256])
            hist = hist.astype("float")
            hist /= (hist.sum() + 0.001)
            
            # Extract features from the histogram
            uniformity = np.sum(hist ** 2)
            entropy = -np.sum(hist * np.log2(hist + 0.00001))
            
            # Patterns usually have more uniform textures
            solid_score = 1.0 - min(1.0, entropy / 5.0)  # Normalize entropy
            pattern_score = min(1.0, entropy / 5.0)  # Inverse of solid score
            
            # Calculate texture directional properties
            # Gradient magnitudes and directions
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, angle = cv2.cartToPolar(gx, gy)
            
            # Histogram of gradient directions
            angle_hist, _ = np.histogram(angle.ravel(), bins=8, range=[0, 2*np.pi])
            angle_hist = angle_hist.astype("float")
            angle_hist /= (angle_hist.sum() + 0.001)
            
            # Directional dominance - higher for stripey patterns
            angle_max = np.max(angle_hist)
            directional_score = (angle_max - 1.0/8.0) * 5.0  # Normalize and enhance
            
            return {
                "solid_score": float(solid_score),
                "pattern_score": float(pattern_score),
                "directional_score": float(directional_score),
                "uniformity": float(uniformity),
                "entropy": float(entropy)
            }
            
        except Exception as e:
            self.logger.error(f"Texture analysis error: {str(e)}")
            return {
                "solid_score": 0.5,
                "pattern_score": 0.5,
                "directional_score": 0.0,
                "uniformity": 0.0,
                "entropy": 0.0
            }

    def _analyze_metadata(self, metadata: Dict[str, str]) -> Dict[str, float]:
        """Analyze pattern type based on metadata like label and description."""
        pattern_scores = {pattern_type: 0.0 for pattern_type in self.pattern_types}
        
        if not metadata:
            return {pattern_type: 1.0/len(self.pattern_types) for pattern_type in self.pattern_types}
        
        # Extract metadata fields that might contain pattern info
        label = metadata.get('label', '').lower()
        description = metadata.get('description', '').lower()
        
        # Combine relevant fields
        text = f"{label} {description}"
        
        # Check for keywords from each pattern type
        for pattern_type, details in self.pattern_types.items():
            keyword_matches = 0
            for keyword in details['keywords']:
                if keyword.lower() in text:
                    keyword_matches += 1
                    pattern_scores[pattern_type] += 0.2
            
            # Boost score for multiple matching keywords
            if keyword_matches > 1:
                pattern_scores[pattern_type] += 0.1 * (keyword_matches - 1)
        
        # Apply special rules for common pattern descriptions
        for pattern_type, details in self.pattern_types.items():
            for example in details['examples']:
                if example.lower() in text:
                    pattern_scores[pattern_type] += 0.3
                    break
        
        # If pattern is explicitly mentioned, give it a high score
        if "pattern" in text:
            for pattern_type, details in self.pattern_types.items():
                if details['name'].lower() in text:
                    pattern_scores[pattern_type] = max(pattern_scores[pattern_type], 0.8)
        
        # Default to solid if no specific pattern mentioned
        if sum(pattern_scores.values()) < 0.2 and "color" in text:
            pattern_scores['solid'] += 0.5
        
        # Normalize scores
        total = sum(pattern_scores.values())
        if total > 0:
            for pattern_type in pattern_scores:
                pattern_scores[pattern_type] = float(pattern_scores[pattern_type] / total)
        else:
            # Default to equal probability if no information found
            for pattern_type in pattern_scores:
                pattern_scores[pattern_type] = 1.0 / len(self.pattern_types)
        
        return pattern_scores

    def analyze_pattern(self, image_path: Union[str, Image.Image], metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze image and metadata to determine the pattern type of clothing.
        
        Args:
            image_path: Path to image file or PIL Image object
            metadata: Optional dict with keys like 'label', 'description', etc.
            
        Returns:
            dict: Contains pattern analysis including classification and confidence
        """
        # Special case handling for suits - they often have subtle patterns that are hard to detect
        suit_detected = False
        if metadata and 'label' in metadata:
            label_lower = metadata.get('label', '').lower()
            if 'suit' in label_lower:
                suit_detected = True
                # Check for specific pattern hints in the label
                if any(word in label_lower for word in ['pinstripe', 'stripe', 'striped']):
                    return {
                        "pattern_type": "striped",
                        "pattern_display": "striped",
                        "confidence": 0.85,
                        "scores": {k: (0.85 if k == "striped" else 0.15/(len(self.pattern_types)-1)) 
                                  for k in self.pattern_types}
                    }
                elif any(word in label_lower for word in ['check', 'plaid', 'herringbone', 'glen']):
                    return {
                        "pattern_type": "checkered", 
                        "pattern_display": "checkered",
                        "confidence": 0.85,
                        "scores": {k: (0.85 if k == "checkered" else 0.15/(len(self.pattern_types)-1)) 
                                  for k in self.pattern_types}
                    }
                else:
                    # Most suits are solid by default unless specified otherwise
                    return {
                        "pattern_type": "solid",
                        "pattern_display": "solid",
                        "confidence": 0.8,
                        "scores": {k: (0.8 if k == "solid" else 0.2/(len(self.pattern_types)-1)) 
                                  for k in self.pattern_types}
                    }

        # Load and validate image for non-suit items or if we want to continue analysis anyway
        image = self._load_image(image_path)
        if image is None:
            return {
                "pattern_type": "unknown",
                "pattern_display": "Unknown",
                "confidence": 0.0,
                "scores": {pattern_type: 0.0 for pattern_type in self.pattern_types},
                "error": "Failed to load image"
            }
        
        # Run multiple analysis methods
        clip_scores = {}
        vqa_scores = {}
        cv_scores = {}
        metadata_scores = {}
        
        # 1. Visual analysis with CLIP
        if self.use_clip:
            clip_scores = self._analyze_with_clip(image)
            self.logger.debug(f"CLIP scores: {clip_scores}")
        
        # 2. VQA analysis
        if self.use_vqa:
            vqa_scores = self._analyze_with_vqa(image)
            self.logger.debug(f"VQA scores: {vqa_scores}")
        
        # 3. Computer vision analysis
        img_np = np.array(image)
        cv_scores = self._analyze_with_cv(image)
        self.logger.debug(f"CV scores: {cv_scores}")
        
        # 4. Metadata analysis
        if metadata:
            metadata_scores = self._analyze_metadata(metadata)
            self.logger.debug(f"Metadata scores: {metadata_scores}")
        
        # Combine all scores with appropriate weighting
        combined_scores = {}
        for pattern_type in self.pattern_types:
            # Weight different methods
            clip_weight = 0.3 if self.use_clip else 0.0
            vqa_weight = 0.3 if self.use_vqa else 0.0
            cv_weight = 0.3
            metadata_weight = 0.1 if metadata else 0.0
            
            # Adjust weights if some methods weren't used
            total_weight = clip_weight + vqa_weight + cv_weight + metadata_weight
            if total_weight == 0:
                combined_scores[pattern_type] = 1.0 / len(self.pattern_types)
                continue
                
            # Normalize weights to sum to 1.0
            clip_weight /= total_weight
            vqa_weight /= total_weight
            cv_weight /= total_weight
            metadata_weight /= total_weight
            
            # Get scores with fallback to uniform distribution
            clip_score = clip_scores.get(pattern_type, 1.0/len(self.pattern_types))
            vqa_score = vqa_scores.get(pattern_type, 1.0/len(self.pattern_types))
            cv_score = cv_scores.get(pattern_type, 1.0/len(self.pattern_types))
            meta_score = metadata_scores.get(pattern_type, 1.0/len(self.pattern_types))
            
            # Combine weighted scores
            combined_scores[pattern_type] = (
                clip_score * clip_weight +
                vqa_score * vqa_weight +
                cv_score * cv_weight +
                meta_score * metadata_weight
            )
        
        # Find the pattern with highest score
        top_pattern = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[top_pattern]
        
        # Get the display name from the pattern details
        pattern_display = self.pattern_types[top_pattern]["name"]
        
        # Create simplified pattern name for frontend display
        display_map = {
            "solid": "solid",
            "striped": "striped",
            "checkered": "checkered",
            "floral": "floral",
            "polka_dot": "polka dot",
            "geometric": "geometric",
            "animal_print": "animal print",
            "abstract": "abstract",
            "camouflage": "camo",
            "paisley": "paisley"
        }
        
        return {
            "pattern_type": top_pattern,
            "pattern_display": display_map.get(top_pattern, pattern_display),
            "confidence": float(confidence),
            "scores": {k: float(v) for k, v in combined_scores.items()}
        }

# Helper function for easy use
def analyze_pattern(image_path, metadata=None):
    """
    Analyze a clothing image to determine its pattern type.
    
    Args:
        image_path: Path to image file or PIL Image object
        metadata: Optional dict with keys like 'label', 'description', etc.
        
    Returns:
        dict: Contains pattern analysis including classification and confidence
    """
    analyzer = PatternAnalyzer()
    return analyzer.analyze_pattern(image_path, metadata)
