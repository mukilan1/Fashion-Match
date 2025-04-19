"""
Costume and outfit style analyzer for clothing inventory system.
Identifies outfit types such as formal, casual, party, business, etc.
"""

import torch
from PIL import Image
import os
import numpy as np
from typing import Dict, Union, List, Optional
from transformers import CLIPProcessor, CLIPModel, pipeline
import logging

class CostumeAnalyzer:
    """Advanced analyzer for identifying clothing costume/style categories."""
    
    def __init__(self, use_clip=True, use_vqa=True):
        """Initialize costume analyzer with multiple model support for better accuracy."""
        self.logger = logging.getLogger('CostumeAnalyzer')
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
        
        # Define costume categories with detailed descriptions for better classification
        self.costume_categories = {
            "formal": {
                "name": "Formal",
                "description": "Elegant attire suitable for important ceremonies, black-tie events",
                "examples": ["tuxedo", "evening gown", "formal suit", "dress shirt with bow tie"],
                "keywords": ["tuxedo", "suit", "gown", "formal", "elegant", "black tie", "evening wear",
                            "dress shirt", "cocktail dress", "blazer", "tailored"]
            },
            "business": {
                "name": "Business",
                "description": "Professional attire worn in office or corporate settings",
                "examples": ["business suit", "blazer with slacks", "pencil skirt with blouse"],
                "keywords": ["business", "professional", "office", "corporate", "work", "blazer",
                            "slacks", "button-up", "tie", "blouse", "tailored"]
            },
            "business_casual": {
                "name": "Business Casual",
                "description": "Semi-professional attire that's relaxed but still presentable",
                "examples": ["chinos with button-down shirt", "blouse with dress pants"],
                "keywords": ["business casual", "smart casual", "semi-formal", "chinos", "khakis",
                            "polo shirt", "casual button-down", "loafers"]
            },
            "casual": {
                "name": "Casual",
                "description": "Everyday comfortable clothing for regular activities",
                "examples": ["jeans with t-shirt", "sundress", "shorts with tank top"],
                "keywords": ["casual", "everyday", "relaxed", "comfortable", "jeans", "t-shirt",
                            "shorts", "sneakers", "hoodie", "sweatshirt", "sweater"]
            },
            "party": {
                "name": "Party",
                "description": "Stylish and fun attire for social gatherings and celebrations",
                "examples": ["cocktail dress", "club wear", "sequin top with dress pants"],
                "keywords": ["party", "club", "night out", "sequin", "glitter", "cocktail", 
                            "dressy", "stylish", "trendy", "fashionable", "sexy"]
            },
            "sports": {
                "name": "Sports/Athletic",
                "description": "Performance clothing for athletic activities",
                "examples": ["tracksuit", "jersey", "running shorts with tank top"],
                "keywords": ["athletic", "sports", "gym", "workout", "exercise", "jersey",
                            "tracksuit", "running", "fitness", "performance"]
            },
            "lounge": {
                "name": "Lounge/Sleepwear",
                "description": "Comfortable clothing for relaxation or sleep",
                "examples": ["pajamas", "loungewear", "robe", "nightgown"],
                "keywords": ["lounge", "sleep", "pajamas", "pjs", "nightwear", "comfortable",
                            "relaxation", "robe", "slippers", "nightgown"]
            },
            "ethnic": {
                "name": "Ethnic/Cultural",
                "description": "Traditional clothing specific to cultures or regions",
                "examples": ["sari", "kimono", "dashiki", "hanbok"],
                "keywords": ["ethnic", "traditional", "cultural", "regional", "folk", "indigenous",
                            "sari", "kimono", "dashiki", "kurta", "hanbok", "cheongsam"]
            },
            "vintage": {
                "name": "Vintage/Retro",
                "description": "Clothing styled after previous eras",
                "examples": ["50s dress", "90s grunge", "vintage band t-shirt"],
                "keywords": ["vintage", "retro", "old school", "throwback", "classic", "antique",
                            "mid-century", "70s", "80s", "90s"]
            },
            "beach": {
                "name": "Beach/Swim",
                "description": "Clothing suited for beach or pool activities",
                "examples": ["swimsuit", "board shorts", "cover-up", "beach dress"],
                "keywords": ["beach", "swim", "swimwear", "bathing suit", "bikini", "board shorts",
                            "cover-up", "resortwear", "tropical", "summer"]
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
        """Use CLIP model to analyze costume/style categories."""
        if not self.use_clip:
            return {category: 1.0/len(self.costume_categories) for category in self.costume_categories}
        
        try:
            # Create detailed prompts for better classification
            prompts = []
            categories = []
            
            for category, details in self.costume_categories.items():
                # Base prompt using the description
                base_prompt = f"This is {details['name'].lower()} clothing: {details['description']}"
                prompts.append(base_prompt)
                categories.append(category)
                
                # Additional prompt using examples for better context
                examples_prompt = f"This outfit is {details['name'].lower()}, like: {', '.join(details['examples'])}"
                prompts.append(examples_prompt)
                categories.append(category)
            
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
            
            # Combine scores for the same category (from different prompts)
            category_scores = {}
            for i, category in enumerate(categories):
                if category not in category_scores:
                    category_scores[category] = probs[i]
                else:
                    # Average with existing score
                    category_scores[category] = (category_scores[category] + probs[i]) / 2
            
            # Normalize scores
            total = sum(category_scores.values())
            if total > 0:
                for category in category_scores:
                    category_scores[category] = float(category_scores[category] / total)
            
            return category_scores
            
        except Exception as e:
            self.logger.error(f"CLIP analysis error: {str(e)}")
            return {category: 1.0/len(self.costume_categories) for category in self.costume_categories}

    def _analyze_with_vqa(self, image: Image.Image) -> Dict[str, float]:
        """Use Visual Question Answering to determine clothing costume/style."""
        if not self.use_vqa:
            return {category: 1.0/len(self.costume_categories) for category in self.costume_categories}
            
        try:
            # Ask direct questions about the outfit's style/occasion
            questions = [
                "What style of clothing is this?",
                "What occasion is this outfit appropriate for?",
                "Is this formal, casual, business, or party attire?",
                "Describe the style of this clothing."
            ]
            
            # Initialize default scores
            category_scores = {category: 0.1 for category in self.costume_categories}
            
            for question in questions:
                result = self.vqa_model(image, question, top_k=5)
                if not result:
                    continue
                    
                for item in result:
                    answer = item['answer'].lower()
                    confidence = item['score']
                    
                    # Match answer to costume categories using keywords
                    for category, details in self.costume_categories.items():
                        # Check if any keywords for this category appear in the answer
                        for keyword in details['keywords']:
                            if keyword.lower() in answer:
                                # Add to the category score, weighted by the model's confidence
                                category_scores[category] += confidence * 0.2
            
            # Normalize scores
            total = sum(category_scores.values())
            if total > 0:
                for category in category_scores:
                    category_scores[category] = float(category_scores[category] / total)
            
            return category_scores
            
        except Exception as e:
            self.logger.error(f"VQA analysis error: {str(e)}")
            return {category: 1.0/len(self.costume_categories) for category in self.costume_categories}

    def _analyze_metadata(self, metadata: Dict[str, str]) -> Dict[str, float]:
        """Analyze costume/style based on metadata like label, color, pattern."""
        category_scores = {category: 0.1 for category in self.costume_categories}
        
        if not metadata:
            return {category: 1.0/len(self.costume_categories) for category in self.costume_categories}
        
        # Get relevant metadata fields
        label = metadata.get('label', '').lower()
        color = metadata.get('color', '').lower()
        pattern = metadata.get('pattern', '').lower()
        
        # Combine text for analysis
        text = f"{label} {color} {pattern}"
        
        # Check for keywords from each category
        for category, details in self.costume_categories.items():
            keyword_matches = 0
            for keyword in details['keywords']:
                if keyword.lower() in text:
                    keyword_matches += 1
                    category_scores[category] += 0.15
            
            # Bonus for multiple matching keywords (suggesting stronger category alignment)
            if keyword_matches > 1:
                category_scores[category] += 0.1 * keyword_matches
        
        # Special rules for certain types of clothing
        if "suit" in text or "blazer" in text:
            if "track" in text or "sport" in text:
                category_scores["sports"] += 0.3
            else:
                category_scores["formal"] += 0.2
                category_scores["business"] += 0.3
        
        if "dress" in text:
            if "evening" in text or "gown" in text:
                category_scores["formal"] += 0.4
            elif "cocktail" in text or "party" in text:
                category_scores["party"] += 0.4
            elif "sun" in text or "casual" in text:
                category_scores["casual"] += 0.3
        
        if "t-shirt" in text or "tshirt" in text or "tee" in text:
            category_scores["casual"] += 0.3
        
        if "swim" in text or "beach" in text or "bikini" in text:
            category_scores["beach"] += 0.5
        
        if "pajama" in text or "pj" in text or "sleep" in text:
            category_scores["lounge"] += 0.5
        
        # Color-based adjustments
        if "black" in color:
            category_scores["formal"] += 0.1
        if "neon" in color or "bright" in color:
            category_scores["party"] += 0.1
            category_scores["casual"] += 0.1
        
        # Pattern-based adjustments
        if "stripe" in pattern:
            if "pinstripe" in pattern:
                category_scores["business"] += 0.2
                category_scores["formal"] += 0.1
        if "floral" in pattern:
            category_scores["casual"] += 0.1
        
        # Normalize scores
        total = sum(category_scores.values())
        if total > 0:
            for category in category_scores:
                category_scores[category] = float(category_scores[category] / total)
        
        return category_scores

    def analyze_costume(self, image_path: Union[str, Image.Image], metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze image and metadata to determine the costume/style of clothing.
        
        Args:
            image_path: Path to image file or PIL Image object
            metadata: Optional dict with keys like 'label', 'color', etc.
            
        Returns:
            dict: Contains costume analysis including classification and confidence
        """
        # Load and validate image
        image = self._load_image(image_path)
        if image is None:
            return {
                "costume": "unknown",
                "confidence": 0.0,
                "costume_display_name": "Unknown",
                "scores": {category: 0.0 for category in self.costume_categories},
                "error": "Failed to load image"
            }
        
        # Special case handling for suits - directly check label before other analysis
        suit_detected = False
        if metadata and 'label' in metadata:
            label_lower = metadata['label'].lower()
            if 'suit' in label_lower:
                suit_detected = True
                # Check for specific suit types
                if 'court' in label_lower or 'formal' in label_lower or 'tuxedo' in label_lower:
                    return {
                        "costume": "formal",
                        "confidence": 0.9,
                        "costume_display_name": "Formal",
                        "scores": {k: (0.9 if k == "formal" else 0.1/(len(self.costume_categories)-1)) 
                                  for k in self.costume_categories},
                        "description": self.costume_categories["formal"]["description"]
                    }
                else:
                    # Default to business for most suits
                    return {
                        "costume": "business",
                        "confidence": 0.85,
                        "costume_display_name": "Business",
                        "scores": {k: (0.85 if k == "business" else 0.15/(len(self.costume_categories)-1)) 
                                  for k in self.costume_categories},
                        "description": self.costume_categories["business"]["description"]
                    }
                    
        # Continue with normal analysis for non-suit items
        
        # Initialize empty scores
        clip_scores = {}
        vqa_scores = {}
        metadata_scores = {}
        
        # Visual analysis with CLIP
        if self.use_clip:
            clip_scores = self._analyze_with_clip(image)
            self.logger.debug(f"CLIP scores: {clip_scores}")
        
        # VQA analysis
        if self.use_vqa:
            vqa_scores = self._analyze_with_vqa(image)
            self.logger.debug(f"VQA scores: {vqa_scores}")
        
        # Metadata analysis
        if metadata:
            metadata_scores = self._analyze_metadata(metadata)
            self.logger.debug(f"Metadata scores: {metadata_scores}")
        
        # Combine all scores with appropriate weighting
        combined_scores = {}
        for category in self.costume_categories:
            # Weight visual clues more heavily than metadata
            clip_weight = 0.4
            vqa_weight = 0.4
            metadata_weight = 0.2
            
            # Get scores with fallback to 0
            clip_score = clip_scores.get(category, 0.0)
            vqa_score = vqa_scores.get(category, 0.0)
            meta_score = metadata_scores.get(category, 0.0)
            
            # Adjust weights if some methods weren't used
            if not self.use_clip:
                vqa_weight += clip_weight / 2
                metadata_weight += clip_weight / 2
                clip_weight = 0
                
            if not self.use_vqa:
                clip_weight += vqa_weight / 2
                metadata_weight += vqa_weight / 2
                vqa_weight = 0
                
            if not metadata:
                clip_weight += metadata_weight / 2
                vqa_weight += metadata_weight / 2
                metadata_weight = 0
            
            # Combine weighted scores
            combined_scores[category] = (
                clip_score * clip_weight +
                vqa_score * vqa_weight +
                meta_score * metadata_weight
            )
        
        # Find the highest scoring category
        top_category = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[top_category]
        
        # Get the display name from the category details
        costume_display_name = self.costume_categories[top_category]["name"]
        
        return {
            "costume": top_category,
            "confidence": float(confidence),
            "costume_display_name": costume_display_name,
            "scores": {k: float(v) for k, v in combined_scores.items()},
            "description": self.costume_categories[top_category]["description"]
        }

# Helper function for easy use
def analyze_costume(image_path, metadata=None):
    """Analyze the costume/style of clothing in an image with optional metadata context."""
    analyzer = CostumeAnalyzer()
    return analyzer.analyze_costume(image_path, metadata)
