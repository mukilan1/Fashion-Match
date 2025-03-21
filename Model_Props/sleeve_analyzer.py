"""
Sleeve analysis module for clothing inventory system.
Identifies sleeve lengths (full sleeve, half sleeve, no sleeve) in clothing items.
"""

from transformers import CLIPProcessor, CLIPModel, pipeline
import torch
from PIL import Image
import os
import numpy as np
from typing import Dict, Union, List, Optional
import logging
import cv2
import io

class SleeveAnalyzer:
    """Specialized analyzer for detecting sleeve length in clothing items."""
    
    def __init__(self, use_clip=True, use_vqa=True):
        """Initialize sleeve analyzer with multiple model support for better accuracy."""
        self.logger = logging.getLogger('SleeveAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # Initialize CLIP model for zero-shot classification
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
        
        # Initialize VQA model for direct questioning
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
        
        # Define sleeve types and their descriptions for better classification
        self.sleeve_types = {
            "full_sleeve": {
                "name": "Full Sleeve",
                "description": "Long sleeves covering the entire arm to the wrist",
                "examples": ["long sleeve shirt", "full sleeve jacket", "winter coat"],
                "keywords": ["long sleeve", "full sleeve", "long-sleeve", "full-sleeve", 
                            "wrist-length", "winter", "coat", "sweater", "formal shirt"]
            },
            "half_sleeve": {
                "name": "Half Sleeve",
                "description": "Medium length sleeves extending to around the elbow",
                "examples": ["t-shirt", "polo shirt", "short sleeve button-up"],
                "keywords": ["half sleeve", "short sleeve", "elbow-length", "medium sleeve",
                            "t-shirt", "polo", "casual shirt"]
            },
            "no_sleeve": {
                "name": "No Sleeve",
                "description": "Sleeveless garment with no fabric covering the arms",
                "examples": ["tank top", "sleeveless dress", "cami", "vest"],
                "keywords": ["sleeveless", "no sleeve", "tank", "camisole", "vest", 
                            "strapless", "spaghetti strap", "muscle shirt", "halter"]
            }
        }
        
        # List of bottom wear items to ignore for sleeve analysis
        self.bottom_wear_items = [
            "pants", "jeans", "shorts", "skirt", "trousers", "leggings", "joggers", 
            "slacks", "chinos", "bottom", "sweatpants", "tights", "capris", "culottes",
            "briefs", "underwear", "boxers", "swim trunks", "bottoms"
        ]

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
        """Use CLIP model to analyze sleeve length."""
        if not self.use_clip:
            return {sleeve_type: 1.0/len(self.sleeve_types) for sleeve_type in self.sleeve_types}
        
        try:
            # Create detailed prompts for better classification
            prompts = []
            categories = []
            
            for sleeve_type, details in self.sleeve_types.items():
                # Base prompt using the description
                base_prompt = f"This is a {details['name'].lower()} garment: {details['description']}"
                prompts.append(base_prompt)
                categories.append(sleeve_type)
                
                # Additional prompt using examples for better context
                examples_prompt = f"This item has {details['name'].lower()}, like: {', '.join(details['examples'])}"
                prompts.append(examples_prompt)
                categories.append(sleeve_type)
                
                # Add image-specific prompts
                if sleeve_type == "full_sleeve":
                    prompts.append("This garment has sleeves that cover the entire arm")
                    categories.append(sleeve_type)
                elif sleeve_type == "half_sleeve":
                    prompts.append("This garment has sleeves that reach the elbow")
                    categories.append(sleeve_type)
                elif sleeve_type == "no_sleeve":
                    prompts.append("This garment has no sleeves and exposes the shoulders")
                    categories.append(sleeve_type)
            
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
            
            # Combine scores for the same sleeve type (from different prompts)
            sleeve_scores = {}
            for i, sleeve_type in enumerate(categories):
                if sleeve_type not in sleeve_scores:
                    sleeve_scores[sleeve_type] = probs[i]
                else:
                    # Average with existing score
                    sleeve_scores[sleeve_type] = (sleeve_scores[sleeve_type] + probs[i]) / 2
            
            # Normalize scores
            total = sum(sleeve_scores.values())
            if total > 0:
                for sleeve_type in sleeve_scores:
                    sleeve_scores[sleeve_type] = float(sleeve_scores[sleeve_type] / total)
            
            return sleeve_scores
            
        except Exception as e:
            self.logger.error(f"CLIP analysis error: {str(e)}")
            return {sleeve_type: 1.0/len(self.sleeve_types) for sleeve_type in self.sleeve_types}

    def _analyze_with_vqa(self, image: Image.Image) -> Dict[str, float]:
        """Use Visual Question Answering to determine sleeve length."""
        if not self.use_vqa:
            return {sleeve_type: 1.0/len(self.sleeve_types) for sleeve_type in self.sleeve_types}
            
        try:
            # Ask direct questions about sleeve length
            questions = [
                "What is the sleeve length of this garment?",
                "Does this item have full sleeves, half sleeves, or no sleeves?",
                "Are the sleeves long, short, or is it sleeveless?",
                "How long are the sleeves on this garment?"
            ]
            
            # Initialize default scores
            sleeve_scores = {sleeve_type: 0.1 for sleeve_type in self.sleeve_types}
            
            for question in questions:
                result = self.vqa_model(image, question, top_k=5)
                if not result:
                    continue
                    
                for item in result:
                    answer = item['answer'].lower()
                    confidence = item['score']
                    
                    # Match answer to sleeve types using keywords
                    if any(keyword in answer for keyword in ["long", "full", "wrist", "entire arm"]):
                        sleeve_scores["full_sleeve"] += confidence * 0.25
                    elif any(keyword in answer for keyword in ["short", "half", "elbow", "mid"]):
                        sleeve_scores["half_sleeve"] += confidence * 0.25
                    elif any(keyword in answer for keyword in ["no sleeve", "sleeveless", "tank", "strapless", "spaghetti"]):
                        sleeve_scores["no_sleeve"] += confidence * 0.25
            
            # Ask more specific follow-up questions for confirmation
            specific_questions = [
                "Do the sleeves reach the wrist?",
                "Do the sleeves reach the elbow?",
                "Is this a sleeveless garment?"
            ]
            
            for i, question in enumerate(specific_questions):
                result = self.vqa_model(image, question)
                if result and len(result) > 0:
                    answer = result[0]['answer'].lower()
                    confidence = result[0]['score']
                    
                    # First question is about full sleeves
                    if i == 0 and "yes" in answer:
                        sleeve_scores["full_sleeve"] += confidence * 0.5
                    # Second question is about half sleeves
                    elif i == 1 and "yes" in answer:
                        sleeve_scores["half_sleeve"] += confidence * 0.5
                    # Third question is about no sleeves
                    elif i == 2 and "yes" in answer:
                        sleeve_scores["no_sleeve"] += confidence * 0.5
            
            # Normalize scores
            total = sum(sleeve_scores.values())
            if total > 0:
                for sleeve_type in sleeve_scores:
                    sleeve_scores[sleeve_type] = float(sleeve_scores[sleeve_type] / total)
            
            return sleeve_scores
            
        except Exception as e:
            self.logger.error(f"VQA analysis error: {str(e)}")
            return {sleeve_type: 1.0/len(self.sleeve_types) for sleeve_type in self.sleeve_types}

    def _analyze_with_edge_detection(self, image: Image.Image) -> Dict[str, float]:
        """
        Use edge detection and contour analysis to determine sleeve presence and length.
        This computer vision approach complements the ML methods for more accurate results.
        """
        try:
            # Convert PIL image to OpenCV format
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Get image dimensions
            height, width = img_cv.shape[:2]
            
            # Preprocessing: resize to manageable size and convert to grayscale
            img_resized = cv2.resize(img_cv, (300, int(300 * height / width)))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours in the edge image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out small contours (noise)
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
            
            # Initial scores
            sleeve_scores = {
                "full_sleeve": 0.33,
                "half_sleeve": 0.33,
                "no_sleeve": 0.34
            }
            
            if not significant_contours:
                return sleeve_scores
            
            # Calculate the bounding rectangles for contours
            bounding_rects = [cv2.boundingRect(cnt) for cnt in significant_contours]
            
            # Extract features that might indicate sleeve presence and length
            top_edge_points = 0
            side_edge_points = 0
            
            # Check top third and sides of the image for edge points (potential sleeve indicators)
            for x, y, w, h in bounding_rects:
                # Check if contour is in the top third (shoulder area)
                if y < img_resized.shape[0] / 3:
                    top_edge_points += w * h
                
                # Check if contour is on the sides (potential sleeve area)
                if x < img_resized.shape[1] / 4 or x + w > img_resized.shape[1] * 3 / 4:
                    side_edge_points += w * h
            
            # Calculate percentages of edge points in these areas
            total_area = img_resized.shape[0] * img_resized.shape[1]
            top_ratio = min(1.0, top_edge_points / (total_area / 3))
            side_ratio = min(1.0, side_edge_points / (total_area / 2))
            
            # Adjust scores based on edge distribution
            # More edges on sides might indicate sleeves
            if side_ratio > 0.15:
                if side_ratio > 0.25:
                    # More side coverage indicates full sleeves
                    sleeve_scores["full_sleeve"] = 0.5
                    sleeve_scores["half_sleeve"] = 0.3
                    sleeve_scores["no_sleeve"] = 0.2
                else:
                    # Moderate side coverage suggests half sleeves
                    sleeve_scores["half_sleeve"] = 0.5
                    sleeve_scores["full_sleeve"] = 0.3
                    sleeve_scores["no_sleeve"] = 0.2
            else:
                # Limited side coverage suggests no sleeves
                sleeve_scores["no_sleeve"] = 0.5
                sleeve_scores["half_sleeve"] = 0.3
                sleeve_scores["full_sleeve"] = 0.2
            
            # Add influence from top edge analysis
            # Smoother top edges may indicate sleeveless garments
            if top_ratio < 0.1:
                sleeve_scores["no_sleeve"] += 0.1
            
            # Normalize final scores
            total = sum(sleeve_scores.values())
            for sleeve_type in sleeve_scores:
                sleeve_scores[sleeve_type] = sleeve_scores[sleeve_type] / total
            
            return sleeve_scores
            
        except Exception as e:
            self.logger.error(f"Edge detection analysis error: {str(e)}")
            return {sleeve_type: 1.0/len(self.sleeve_types) for sleeve_type in self.sleeve_types}

    def _analyze_metadata(self, metadata: Dict[str, str]) -> Dict[str, float]:
        """Analyze sleeve type based on metadata like label, description."""
        sleeve_scores = {sleeve_type: 0.33 for sleeve_type in self.sleeve_types}
        
        if not metadata:
            return sleeve_scores
        
        # Get relevant metadata fields
        label = metadata.get('label', '').lower()
        description = metadata.get('description', '').lower()
        
        # Combine text for analysis
        text = f"{label} {description}"
        
        # Check for keywords from each sleeve type
        for sleeve_type, details in self.sleeve_types.items():
            keyword_matches = 0
            for keyword in details['keywords']:
                if keyword.lower() in text:
                    keyword_matches += 1
                    sleeve_scores[sleeve_type] += 0.15
            
            # Bonus for multiple matching keywords
            if keyword_matches > 1:
                sleeve_scores[sleeve_type] += 0.1 * keyword_matches
        
        # Special rules for specific keywords
        if "long sleeve" in text or "full sleeve" in text:
            sleeve_scores["full_sleeve"] = max(sleeve_scores["full_sleeve"], 0.7)
        
        if "short sleeve" in text or "half sleeve" in text:
            sleeve_scores["half_sleeve"] = max(sleeve_scores["half_sleeve"], 0.7)
        
        if "sleeveless" in text or "tank" in text or "strapless" in text:
            sleeve_scores["no_sleeve"] = max(sleeve_scores["no_sleeve"], 0.7)
        
        # Normalize scores
        total = sum(sleeve_scores.values())
        if total > 0:
            for sleeve_type in sleeve_scores:
                sleeve_scores[sleeve_type] = float(sleeve_scores[sleeve_type] / total)
        
        return sleeve_scores

    def _is_bottom_wear(self, metadata: Dict[str, str]) -> bool:
        """Determine if the item is bottom wear (pants, skirt, etc.)."""
        if not metadata:
            return False
            
        # Check label and wearable position
        label = metadata.get('label', '').lower()
        wearable = metadata.get('wearable', '').lower()
        
        # Look for explicit bottom wear indicators
        if "bottom wearable" in wearable:
            return True
            
        # Check if label contains bottom wear items
        for item in self.bottom_wear_items:
            if item in label:
                return True
                
        return False

    def analyze_sleeve(self, image_path: Union[str, Image.Image], metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze image and metadata to determine sleeve length.
        
        Args:
            image_path: Path to image file or PIL Image object
            metadata: Optional dict with keys like 'label', 'description', etc.
            
        Returns:
            dict: Contains sleeve analysis including classification and confidence
        """
        # Check if this is bottom wear that should be ignored
        if metadata and self._is_bottom_wear(metadata):
            return {
                "sleeve_type": "not_applicable",
                "sleeve_display": "N/A (Bottom Wear)",
                "confidence": 1.0,
                "is_bottom_wear": True
            }
        
        # Load and validate image
        image = self._load_image(image_path)
        if image is None:
            return {
                "sleeve_type": "unknown",
                "sleeve_display": "Unknown",
                "confidence": 0.0,
                "scores": {sleeve_type: 0.0 for sleeve_type in self.sleeve_types},
                "error": "Failed to load image"
            }
        
        # Initialize empty scores
        clip_scores = {}
        vqa_scores = {}
        edge_scores = {}
        metadata_scores = {}
        
        # Visual analysis with CLIP
        if self.use_clip:
            clip_scores = self._analyze_with_clip(image)
            self.logger.debug(f"CLIP scores: {clip_scores}")
        
        # VQA analysis
        if self.use_vqa:
            vqa_scores = self._analyze_with_vqa(image)
            self.logger.debug(f"VQA scores: {vqa_scores}")
        
        # Edge detection analysis
        edge_scores = self._analyze_with_edge_detection(image)
        self.logger.debug(f"Edge detection scores: {edge_scores}")
        
        # Metadata analysis
        if metadata:
            metadata_scores = self._analyze_metadata(metadata)
            self.logger.debug(f"Metadata scores: {metadata_scores}")
        
        # Combine all scores with appropriate weighting
        combined_scores = {}
        for sleeve_type in self.sleeve_types:
            # Weight ML/AI methods more heavily than traditional CV
            clip_weight = 0.3 if self.use_clip else 0.0
            vqa_weight = 0.3 if self.use_vqa else 0.0
            edge_weight = 0.2
            metadata_weight = 0.2 if metadata else 0.0
            
            # Adjust weights if some methods weren't used
            total_weight = clip_weight + vqa_weight + edge_weight + metadata_weight
            if total_weight == 0:
                # Fallback to equal weights if no methods were used
                combined_scores[sleeve_type] = 1.0 / len(self.sleeve_types)
                continue
                
            # Normalize weights to sum to 1.0
            clip_weight /= total_weight
            vqa_weight /= total_weight
            edge_weight /= total_weight
            metadata_weight /= total_weight
            
            # Get scores with fallback to equal distribution
            clip_score = clip_scores.get(sleeve_type, 1.0/len(self.sleeve_types))
            vqa_score = vqa_scores.get(sleeve_type, 1.0/len(self.sleeve_types))
            edge_score = edge_scores.get(sleeve_type, 1.0/len(self.sleeve_types))
            meta_score = metadata_scores.get(sleeve_type, 1.0/len(self.sleeve_types))
            
            # Combine weighted scores
            combined_scores[sleeve_type] = (
                clip_score * clip_weight +
                vqa_score * vqa_weight +
                edge_score * edge_weight +
                meta_score * metadata_weight
            )
        
        # Find the highest scoring sleeve type
        top_sleeve_type = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[top_sleeve_type]
        
        # Get the display name from the sleeve type details
        sleeve_display = self.sleeve_types[top_sleeve_type]["name"]
        
        # Simplified output names for frontend compatibility
        display_map = {
            "full_sleeve": "full hand",
            "half_sleeve": "half hand",
            "no_sleeve": "no hand"
        }
        
        return {
            "sleeve_type": top_sleeve_type,
            "sleeve_display": display_map.get(top_sleeve_type, sleeve_display),
            "confidence": float(confidence),
            "scores": {k: float(v) for k, v in combined_scores.items()},
            "is_bottom_wear": False
        }

# Helper function for easy use
def analyze_sleeve(image_path, metadata=None):
    """Analyze sleeve length of clothing in an image with optional metadata context."""
    analyzer = SleeveAnalyzer()
    return analyzer.analyze_sleeve(image_path, metadata)
