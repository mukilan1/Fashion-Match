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
    
    def analyze_image(self, image_data, question="What type of clothing item is this and where is it worn (top or bottom)?") -> Dict[str, Union[str, float]]:
        """
        Analyze an image with a given question using multiple models for better accuracy.
        
        Args:
            image_data: PIL Image object or path to image file
            question: String containing the question to ask about the image
            
        Returns:
            dict: Contains 'classification' and 'confidence' keys
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
                
                # Use the VQA result if it's more specific about location (top/bottom)
                if any(location in vqa_answer.lower() for location in self.body_locations):
                    final_classification = vqa_answer
                    final_confidence = vqa_confidence
                elif clip_confidence > 50:  # If CLIP is confident enough
                    final_classification = clip_classification
                    final_confidence = clip_confidence
                else:
                    final_classification = vqa_answer
                    final_confidence = vqa_confidence
                
                print("=" * 50)
                print(f"CLOTHING DETECTION RESULT:")
                print(f"CLIP Classification: {clip_classification} ({clip_confidence:.2f}%)")
                print(f"VQA Answer: {vqa_answer} ({vqa_confidence:.2f}%)")
                print(f"Final Result: {final_classification}")
                print(f"Final Confidence: {final_confidence:.2f}%")
                print("=" * 50)
                
                return {
                    "classification": final_classification,
                    "confidence": float(final_confidence)
                }
            else:
                # Fall back to CLIP if VQA fails
                return {
                    "classification": clip_classification,
                    "confidence": float(clip_confidence)
                }
                
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return {"classification": "unknown", "confidence": 0.0, "error": str(e)}
