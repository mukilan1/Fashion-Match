"""
Gender analysis module for clothing items.
Uses deep learning models to predict whether an item is designed for men, women, or is unisex.
"""

from transformers import CLIPProcessor, CLIPModel, pipeline
import torch
from PIL import Image
import os
import numpy as np
from typing import Dict, Union, List, Tuple, Optional
import logging

class GenderAnalyzer:
    """Specialized clothing gender analyzer with high accuracy."""
    
    def __init__(self, use_clip=True, use_vqa=True):
        """Initialize gender analyzer with multiple model support for better accuracy."""
        self.logger = logging.getLogger('GenderAnalyzer')
        
        # Initialize CLIP model for image understanding
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
            
        # Gender-specific clothing terms for better classification
        self.mens_terms = [
            "men's", "mens", "men", "masculine", "male", "boy", "gentleman", "gent",
            "suit", "tuxedo", "necktie", "bow tie", "boxer", "briefs",
        ]
        
        self.womens_terms = [
            "women's", "womens", "women", "feminine", "female", "girl", "lady", "ladies",
            "dress", "skirt", "blouse", "bra", "bikini", "gown", "maternity", "lehenga",
            "saree", "sari", "high heels", "pantyhose", "lipstick"
        ]
        
        # Special case items that can be ambiguous but have gender-specific designs
        self.analyze_context_items = [
            "shirt", "t-shirt", "pants", "jeans", "shorts", "jacket", "sweater",
            "hoodie", "coat", "socks", "shoes", "hat", "scarf", "watch"
        ]
        
        # Gender-specific design features
        self.feminine_design_features = [
            "floral", "pink", "purple", "crop top", "v-neck", "lace", "frills",
            "ruffles", "peplum", "pleated", "a-line", "fitted", "slim fit"
        ]
        
        self.masculine_design_features = [
            "straight cut", "loose fit", "dark tones", "plain", "button-down collar",
            "regular fit", "relaxed fit", "cargo", "geometric", "camo pattern"
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
        """Use CLIP model to analyze gender indicators in clothing."""
        if not self.use_clip:
            return {"men": 0.33, "women": 0.33, "unisex": 0.34}
        
        try:
            # Prepare prompts for gender classification
            prompts = [
                "This is men's clothing.",
                "This is women's clothing.",
                "This is unisex clothing.",
                "This clothing has masculine design elements.",
                "This clothing has feminine design elements.",
                "This is a gender-neutral clothing item."
            ]
            
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
            
            # Combine related probabilities
            men_score = (probs[0] + probs[3]) / 2
            women_score = (probs[1] + probs[4]) / 2
            unisex_score = (probs[2] + probs[5]) / 2
            
            # Normalize scores
            total = men_score + women_score + unisex_score
            if total > 0:
                men_score /= total
                women_score /= total
                unisex_score /= total
            
            return {
                "men": float(men_score),
                "women": float(women_score),
                "unisex": float(unisex_score)
            }
        except Exception as e:
            self.logger.error(f"CLIP analysis error: {str(e)}")
            return {"men": 0.33, "women": 0.33, "unisex": 0.34}

    def _analyze_with_vqa(self, image: Image.Image) -> Dict[str, float]:
        """Use Visual Question Answering to determine clothing gender."""
        if not self.use_vqa:
            return {"men": 0.33, "women": 0.33, "unisex": 0.34}
            
        try:
            # Ask a direct question about gender
            question = "Is this clothing designed for men, women, or is it unisex?"
            result = self.vqa_model(image, question, top_k=3)
            
            men_score, women_score, unisex_score = 0.33, 0.33, 0.34
            
            if result:
                # Process the results
                for item in result:
                    answer = item['answer'].lower()
                    score = item['score']
                    
                    # Map answers to gender categories
                    if any(term in answer for term in ["men", "man", "male", "masculine", "boy"]):
                        men_score = score
                    elif any(term in answer for term in ["women", "woman", "female", "feminine", "girl"]):
                        women_score = score
                    elif any(term in answer for term in ["unisex", "both", "neutral", "any", "all"]):
                        unisex_score = score
            
            # Ask follow-up questions for confirmation
            specific_q = "Does this clothing have gender-specific design elements?"
            specific_result = self.vqa_model(image, specific_q)
            
            if specific_result:
                answer = specific_result[0]['answer'].lower()
                # Adjust scores based on the answer
                if "yes" in answer:
                    # If gender-specific, reduce unisex score slightly
                    unisex_score = max(0.1, unisex_score * 0.7)
                elif "no" in answer:
                    # If not gender-specific, boost unisex score
                    unisex_score = min(0.8, unisex_score * 1.5)
            
            # Normalize scores
            total = men_score + women_score + unisex_score
            return {
                "men": float(men_score / total),
                "women": float(women_score / total),
                "unisex": float(unisex_score / total)
            }
        except Exception as e:
            self.logger.error(f"VQA analysis error: {str(e)}")
            return {"men": 0.33, "women": 0.33, "unisex": 0.34}

    def _analyze_label(self, label: str) -> Dict[str, float]:
        """Analyze clothing label text for gender indicators."""
        if not label or label.lower() == "unknown":
            return {"men": 0.33, "women": 0.33, "unisex": 0.34}
        
        label = label.lower()
        
        # Check for direct gender indicators
        men_indicators = sum(1 for term in self.mens_terms if term in label)
        women_indicators = sum(1 for term in self.womens_terms if term in label)
        
        # Check for design features that suggest gender
        men_design = sum(0.5 for feature in self.masculine_design_features if feature in label)
        women_design = sum(0.5 for feature in self.feminine_design_features if feature in label)
        
        # Combine indicators
        men_score = men_indicators + men_design
        women_score = women_indicators + women_design
        
        # If we have gender indicators
        if men_score > 0 or women_score > 0:
            # Calculate unisex score inversely proportional to gender-specific scores
            total_gender_score = men_score + women_score
            unisex_score = max(0.1, 1 - (total_gender_score / 10))
            
            # Normalize scores
            total = men_score + women_score + unisex_score
            return {
                "men": men_score / total,
                "women": women_score / total,
                "unisex": unisex_score / total
            }
        
        # For items that need context
        if any(item in label for item in self.analyze_context_items):
            # Default slightly toward unisex for these items
            return {"men": 0.3, "women": 0.3, "unisex": 0.4}
        
        # Default to balanced probabilities if no indicators
        return {"men": 0.33, "women": 0.33, "unisex": 0.34}

    def analyze_gender(self, image_path: Union[str, Image.Image], metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze image and metadata to determine the gender of clothing.
        
        Args:
            image_path: Path to image file or PIL Image object
            metadata: Optional dict with keys like 'label', 'color', etc.
            
        Returns:
            dict: Contains gender analysis including classification and confidence
        """
        # Load and validate image
        image = self._load_image(image_path)
        if image is None:
            return {
                "gender": "unknown",
                "confidence": 0.0,
                "probabilities": {"men": 0.33, "women": 0.33, "unisex": 0.34},
                "error": "Failed to load image"
            }
        
        # Initialize results storage
        clip_results = {"men": 0.33, "women": 0.33, "unisex": 0.34}
        vqa_results = {"men": 0.33, "women": 0.33, "unisex": 0.34}
        text_results = {"men": 0.33, "women": 0.33, "unisex": 0.34}
        
        # Visual analysis with CLIP
        if self.use_clip:
            clip_results = self._analyze_with_clip(image)
            self.logger.debug(f"CLIP results: {clip_results}")
        
        # VQA analysis
        if self.use_vqa:
            vqa_results = self._analyze_with_vqa(image)
            self.logger.debug(f"VQA results: {vqa_results}")
            
        # Text-based analysis from metadata
        if metadata:
            label = metadata.get("label", "")
            if label:
                text_results = self._analyze_label(label)
                self.logger.debug(f"Text analysis results: {text_results}")
                
            # If metadata has a specific gender field already
            if metadata.get("sex") in ["Men's", "men's", "male"]:
                text_results = {"men": 0.7, "women": 0.1, "unisex": 0.2}
            elif metadata.get("sex") in ["Women's", "women's", "female"]:
                text_results = {"men": 0.1, "women": 0.7, "unisex": 0.2}
        
        # Combine results with weighted averaging
        # Give more weight to VQA for direct questioning capability
        combined = {
            "men": clip_results["men"] * 0.3 + vqa_results["men"] * 0.4 + text_results["men"] * 0.3,
            "women": clip_results["women"] * 0.3 + vqa_results["women"] * 0.4 + text_results["women"] * 0.3,
            "unisex": clip_results["unisex"] * 0.3 + vqa_results["unisex"] * 0.4 + text_results["unisex"] * 0.3
        }
        
        # Special case handling for clothing types with strong gender associations
        if metadata and metadata.get("label"):
            label = metadata.get("label", "").lower()
            
            # Items that are strongly associated with women
            if any(term in label for term in ["dress", "skirt", "blouse", "bra"]):
                combined["women"] = max(combined["women"], 0.8)
                combined["men"] = min(combined["men"], 0.1)
            
            # Items strongly associated with men
            if any(term in label for term in ["tuxedo", "necktie", "bow tie"]):
                combined["men"] = max(combined["men"], 0.8)
                combined["women"] = min(combined["women"], 0.1)
        
        # Determine most likely gender
        max_gender = max(combined, key=combined.get)
        confidence = combined[max_gender]
        
        # Map to friendly names
        gender_map = {
            "men": "Men's",
            "women": "Women's",
            "unisex": "Unisex"
        }
        
        # If confidence is low, may be unisex
        if confidence < 0.45:
            gender = "Unisex"
            confidence = combined["unisex"]
        else:
            gender = gender_map[max_gender]
        
        return {
            "gender": gender,
            "confidence": float(confidence),
            "probabilities": {
                "men": float(combined["men"]), 
                "women": float(combined["women"]), 
                "unisex": float(combined["unisex"])
            }
        }

# Helper function for easy use
def analyze_gender(image_path, metadata=None):
    """Analyze gender of clothing in an image with optional metadata context."""
    analyzer = GenderAnalyzer()
    return analyzer.analyze_gender(image_path, metadata)
