from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import os
import io
import torch
import numpy as np
from typing import Dict, Any, Union, List, Tuple

class ClothingImageAnalyzer:
    """Clothing analyzer with improved model performance"""
    
    def __init__(self, use_vqa=True, use_ml_model=True, model_dir=None, log_level=None):
        # Initialize both models for better performance
        self.vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
        print("Advanced VQA pipeline initialized")
        
        # Initialize CLIP model for more accurate image understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model initialized")
        
        # Clothing categories for better classification
        self.clothing_categories = [
            "top", "bottom", "shirt", "t-shirt", "blouse", "pants", "jeans", 
            "skirt", "dress", "jacket", "coat", "sweater", "hoodie", "shorts",
            "shoes", "sneakers", "boots", "hat", "cap", "accessory", "socks"
        ]
        
        self.body_locations = ["top", "bottom", "full body", "feet", "head", "hands"]
        
        # Extended lists for wearable position classification
        self.top_items = ["shirt", "t-shirt", "blouse", "jacket", "coat", "sweater", "hoodie", 
                         "top", "sweatshirt", "cardigan", "polo", "tank", "vest", "jersey", 
                         "tunic", "turtleneck", "button-down"]
        
        self.bottom_items = ["pants", "jeans", "shorts", "skirt", "trousers", "leggings", 
                            "joggers", "slacks", "chinos", "bottom", "sweatpants", "tights",
                            "capris", "culottes", "briefs"]
        
        # Added full body items list to correctly classify suits and full-body garments
        self.full_body_items = ["dress", "jumpsuit", "romper", "suit", "overall", "gown", 
                               "onesie", "bodysuit", "costume", "court suit"]
    
    def _analyze_with_clip(self, image) -> Tuple[str, float]:
        """Use CLIP model for better clothing classification"""
        try:
            # Prepare text prompts for classification
            text_inputs = self.clip_processor(
                text=["This is " + category for category in self.clothing_categories],
                return_tensors="pt",
                padding=True
            )
            
            # Process the image
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.clip_model(**{
                    **image_inputs,
                    **text_inputs
                })
                
                # Calculate similarity scores
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).numpy()[0]
                
                # Get the highest probability classification
                best_idx = np.argmax(probs)
                classification = self.clothing_categories[best_idx]
                confidence = float(probs[best_idx]) * 100
                
                return classification, confidence
        
        except Exception as e:
            print(f"CLIP analysis error: {str(e)}")
            return "unknown", 0.0
    
    def classify_wearable_position(self, label: str) -> str:
        """
        Classify a clothing item as 'top wearable', 'bottom wearable', or 'full body' based on its label.
        
        Args:
            label: String containing the clothing description/label
            
        Returns:
            str: 'top wearable', 'bottom wearable', 'full body', or 'unknown'
        """
        if not label or label == "unknown":
            return "unknown"
        
        # Convert to lowercase for case-insensitive matching
        label_lower = label.lower()
        
        # Special case handling for suits and coats - they should never be bottom wearables
        if "suit" in label_lower or "coat" in label_lower:
            # Court suits and formal suits are full body items
            if "court" in label_lower or "formal" in label_lower:
                return "full body"
            # Regular coats are top wearables
            return "top wearable"
        
        # Check for full body items first (most specific category)
        for item in self.full_body_items:
            if item in label_lower:
                return "full body"
        
        # Check for explicit matches with top items
        for item in self.top_items:
            if item in label_lower:
                return "top wearable"
        
        # Check for explicit matches with bottom items
        for item in self.bottom_items:
            if item in label_lower:
                return "bottom wearable"
        
        # Use CLIP-based similarity if no direct match is found and CLIP is available
        try:
            # Create text for comparison
            text_inputs = self.clip_processor(
                text=["This is a top garment worn on the upper body", 
                     "This is a bottom garment worn on the lower body",
                     "This is a full body garment covering both upper and lower body"],
                return_tensors="pt",
                padding=True
            )
            
            # Process the text description as an approximation
            label_inputs = self.clip_processor(
                text=[f"This is a {label}"],
                return_tensors="pt", 
                padding=True
            )
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**text_inputs)
                label_outputs = self.clip_model.get_text_features(**label_inputs)
                
            # Calculate similarity scores
            top_score = torch.cosine_similarity(label_outputs, outputs[0:1], dim=1).item()
            bottom_score = torch.cosine_similarity(label_outputs, outputs[1:2], dim=1).item()
            full_body_score = torch.cosine_similarity(label_outputs, outputs[2:3], dim=1).item()
            
            # Return the category with highest similarity
            if full_body_score > top_score and full_body_score > bottom_score:
                return "full body"
            elif top_score > bottom_score:
                return "top wearable"
            else:
                return "bottom wearable"
                
        except Exception as e:
            print(f"Error in wearable position classification: {str(e)}")
            # Default to top wearable if we can't determine
            return "top wearable"
    
    def analyze_image(self, image_data, question="What type of clothing item is this?") -> Dict[str, Union[str, float]]:
        """
        Analyze an image with a given question using multiple models for better accuracy.
        
        Args:
            image_data: PIL Image object or path to image file
            question: String containing the question to ask about the image
            
        Returns:
            dict: Contains 'classification', 'confidence', and 'wearable_position' keys
        """
        try:
            # Convert to PIL Image
            if isinstance(image_data, str):
                image = Image.open(image_data)
                print(f"Processing file: {os.path.basename(image_data)}")
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                return {"classification": "unknown", "confidence": 0.0}
            
            # First try specialized CLIP-based analysis for clothing
            clip_classification, clip_confidence = self._analyze_with_clip(image)
            
            # Then use VQA as a backup/verification
            vqa_result = self.vqa_pipeline(image, question, top_k=3)
            
            if vqa_result and len(vqa_result) > 0:
                vqa_answer = vqa_result[0].get('answer', 'unknown')
                vqa_confidence = vqa_result[0].get('score', 0) * 100
                
                # Use the VQA result if it contains location information
                if any(location in vqa_answer.lower() for location in self.body_locations):
                    final_classification = vqa_answer
                    final_confidence = vqa_confidence
                elif clip_confidence > 50:  # If CLIP is confident enough
                    final_classification = clip_classification
                    final_confidence = clip_confidence
                else:
                    final_classification = vqa_answer
                    final_confidence = vqa_confidence
                
                # Add wearable position classification based on the final classification
                wearable_position = self.classify_wearable_position(final_classification)
                
                # Special handling for suits and coats to ensure proper classification
                if "suit" in final_classification.lower() or "coat" in final_classification.lower():
                    # Ask a direct follow-up question to confirm if it's a full body suit
                    follow_up = "Is this a full body suit or just a coat?"
                    follow_result = self.vqa_pipeline(image, follow_up, top_k=1)
                    
                    if follow_result and "full" in follow_result[0]['answer'].lower():
                        wearable_position = "full body"
                    elif "coat" in final_classification.lower():
                        wearable_position = "top wearable"
                    else:
                        # Default to top wearable for suits if unsure
                        wearable_position = "top wearable"
                
                # If wearable position is still unknown, use a direct question
                if wearable_position == "unknown":
                    direct_question = "Is this a top, bottom, or full body garment?"
                    position_result = self.vqa_pipeline(image, direct_question, top_k=1)
                    
                    if position_result:
                        answer = position_result[0]['answer'].lower()
                        if "top" in answer:
                            wearable_position = "top wearable"
                        elif "bottom" in answer:
                            wearable_position = "bottom wearable"
                        elif "full" in answer or "body" in answer:
                            wearable_position = "full body"
                
                # If still unknown, try CLIP with the image directly
                if wearable_position == "unknown":
                    print("Trying direct image classification for position...")
                    try:
                        text_inputs = self.clip_processor(
                            text=["This is a top garment", "This is a bottom garment", "This is a full body garment"],
                            return_tensors="pt",
                            padding=True
                        )
                        
                        image_inputs = self.clip_processor(images=image, return_tensors="pt")
                        
                        with torch.no_grad():
                            outputs = self.clip_model(**{**image_inputs, **text_inputs})
                            probs = outputs.logits_per_image.softmax(dim=1)[0].numpy()
                            
                        if probs[0] > probs[1] and probs[0] > probs[2]:
                            wearable_position = "top wearable"
                        elif probs[1] > probs[0] and probs[1] > probs[2]:
                            wearable_position = "bottom wearable"
                        else:
                            wearable_position = "full body"
                        
                        print(f"Image-based position classification: {wearable_position} ({max(probs)*100:.1f}%)")
                    except Exception as e:
                        print(f"Error in direct image position classification: {str(e)}")
                
                # Ensure we have some position, default to a guess based on the label
                if wearable_position == "unknown":
                    # Use common words as fallback
                    label_lower = final_classification.lower()
                    if "suit" in label_lower or "coat" in label_lower:
                        wearable_position = "top wearable"  # Default suits/coats to top wearable
                    elif any(word in label_lower for word in ["shirt", "top", "jacket", "sweater"]):
                        wearable_position = "top wearable"
                    elif any(word in label_lower for word in ["pants", "jeans", "skirt", "shorts", "trousers"]):
                        wearable_position = "bottom wearable"
                    else:
                        # Make an educated guess
                        wearable_position = "top wearable"  # Default to top if truly can't tell
                
                print("=" * 50)
                print(f"CLOTHING DETECTION RESULT:")
                print(f"CLIP Classification: {clip_classification} ({clip_confidence:.2f}%)")
                print(f"VQA Answer: {vqa_answer} ({vqa_confidence:.2f}%)")
                print(f"Final Result: {final_classification}")
                print(f"Final Confidence: {final_confidence:.2f}%")
                print(f"Wearable Position: {wearable_position}")
                print("=" * 50)
                
                return {
                    "classification": final_classification,
                    "confidence": float(final_confidence),
                    "wearable_position": wearable_position
                }
            else:
                # Fall back to CLIP if VQA fails
                wearable_position = self.classify_wearable_position(clip_classification)
                
                return {
                    "classification": clip_classification,
                    "confidence": float(clip_confidence),
                    "wearable_position": wearable_position
                }
                
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return {"classification": "unknown", "confidence": 0.0, 
                    "wearable_position": "unknown", "error": str(e)}
