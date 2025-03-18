import os
import logging
import torch
import numpy as np
from PIL import Image
import io
import sys
from typing import Dict, List, Tuple, Any, Optional, Union

# Add progress indicator function for image analyzer
def show_analyzer_progress(operation, percent=0, status="", final=False):
    """
    Display a progress indicator in the console for image analysis operations.
    
    Args:
        operation: String describing the current operation
        percent: Progress percentage (0-100)
        status: Additional status message
        final: Whether this is the final update for this operation
    """
    bar_length = 20
    filled_length = int(bar_length * percent / 100)
    bar = '■' * filled_length + '□' * (bar_length - filled_length)
    
    if final:
        sys.stdout.write(f"\r{operation}: [{bar}] {percent:.1f}% - {status} ✓ Completed\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r{operation}: [{bar}] {percent:.1f}% - {status}")
        sys.stdout.flush()

# Configure optional imports for the transformers models
try:
    from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers package not available. Some features will be disabled.")

class ClothingImageAnalyzer:
    """
    A class for analyzing clothing images to determine item type, color, pattern,
    and other characteristics using various AI techniques.
    """
    
    def __init__(self, use_vqa=True, use_ml_model=True, model_dir=None, log_level=logging.INFO):
        """
        Initialize the clothing image analyzer.
        
        Args:
            use_vqa: Whether to use Visual Question Answering
            use_ml_model: Whether to use ML classification models
            model_dir: Directory to store/load models (default: local_models in current directory)
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(level=log_level, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ClothingImageAnalyzer')
        
        # Initialize model variables
        self.model = None
        self.feature_extractor = None
        self.model_loaded = False
        self.vqa_pipeline = None
        self.vqa_loaded = False
        
        # Set model directory
        if model_dir is None:
            self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_models")
        else:
            self.model_dir = model_dir
            
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models if requested and available
        if TRANSFORMERS_AVAILABLE:
            if use_vqa:
                self.init_vqa()
            if use_ml_model:
                self.init_model()
        else:
            self.logger.warning("Transformers package not available. Using direct image analysis only.")
        
        # Initialize direct classifier
        self.direct_classifier = DirectTopBottomClassifier()
    
    def init_vqa(self):
        """Initialize Visual Question-Answering pipeline"""
        try:
            show_analyzer_progress("Loading VQA pipeline", 0, "Starting")
            self.logger.info("Loading Visual Question-Answering pipeline...")
            # Use a smaller VQA model that works well on CPU
            show_analyzer_progress("Loading VQA pipeline", 50, "Loading model")
            self.vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
            self.vqa_loaded = True
            show_analyzer_progress("Loading VQA pipeline", 100, "Model loaded", final=True)
            self.logger.info("VQA pipeline loaded successfully")
        except Exception as e:
            show_analyzer_progress("Loading VQA pipeline", 100, f"Error: {str(e)}", final=True)
            self.logger.error(f"Error loading VQA pipeline: {str(e)}")
            self.vqa_loaded = False

    def init_model(self):
        """Initialize ML model for clothing classification"""
        # Specifically target clothing classification models
        model_names = [
            "patrickjohncyh/fashion-mnist",  # Fashion MNIST model - good for clothing
            "noambechar/clothes-recognition", # Specialized clothing recognition model
            "microsoft/resnet-50"            # Reliable general purpose model
        ]
        
        model_loaded = False
        for i, model_name in enumerate(model_names):
            show_analyzer_progress("Loading ML model", (i / len(model_names)) * 50, f"Trying {model_name}")
            try:
                local_model_path = os.path.join(self.model_dir, model_name.split('/')[-1])
                
                # First check if model already exists locally
                if os.path.exists(local_model_path):
                    self.logger.info(f"Loading model from local storage: {local_model_path}")
                    show_analyzer_progress("Loading ML model", (i / len(model_names)) * 50 + 10, f"Loading from local storage")
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)
                    self.model = AutoModelForImageClassification.from_pretrained(local_model_path)
                else:
                    # Download and save model locally
                    self.logger.info(f"Downloading model {model_name} to {local_model_path}")
                    show_analyzer_progress("Loading ML model", (i / len(model_names)) * 50 + 10, f"Downloading model")
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                    show_analyzer_progress("Loading ML model", (i / len(model_names)) * 50 + 20, f"Downloading weights")
                    self.model = AutoModelForImageClassification.from_pretrained(model_name)
                    
                    # Save model locally for future use
                    show_analyzer_progress("Loading ML model", (i / len(model_names)) * 50 + 30, f"Saving model locally")
                    self.feature_extractor.save_pretrained(local_model_path)
                    self.model.save_pretrained(local_model_path)
                
                # Ensure model is on CPU
                self.model.to("cpu")
                self.model_loaded = True
                model_loaded = True
                show_analyzer_progress("Loading ML model", 100, f"Successfully loaded {model_name}", final=True)
                self.logger.info(f"Successfully loaded model: {model_name}")
                break  # Exit the loop if model loaded successfully
                
            except Exception as e:
                show_analyzer_progress("Loading ML model", (i / len(model_names)) * 50 + 40, f"Error: {str(e)}")
                self.logger.error(f"Error loading model {model_name}: {str(e)}")
                continue
        
        # Verify model loaded correctly
        if not model_loaded:
            show_analyzer_progress("Loading ML model", 100, "Failed to load any model, using direct classifier", final=True)
            self.logger.warning("Failed to load ML model - will use direct classifier")
            self.model_loaded = False

    def analyze_image(self, image_data: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Analyze a clothing image and return comprehensive information.
        
        Args:
            image_data: Can be a file path, image bytes, or PIL Image object
            
        Returns:
            Dictionary containing classification results and details
        """
        try:
            # Convert input to PIL Image
            show_analyzer_progress("Analyzing image", 10, "Loading image")
            image = self._ensure_pil_image(image_data)
            image = image.convert('RGB')  # Ensure consistent image format
            
            # First, get direct image analysis result
            show_analyzer_progress("Analyzing image", 30, "Analyzing features")
            direct_result, direct_confidence, direct_features = self.analyze_top_bottom_features(image)
            self.logger.info(f"Direct image analysis: {direct_result} ({direct_confidence:.2f}%)")
            
            # Try VQA-based classification (most accurate with details)
            vqa_result, vqa_confidence, vqa_details = None, None, None
            if self.vqa_loaded:
                show_analyzer_progress("Analyzing image", 50, "Running visual QA")
                vqa_result, vqa_confidence, vqa_details = self.classify_with_vqa(image)
            
            # Try ML model classification
            ml_result, ml_confidence, ml_prediction = None, None, None
            if self.model_loaded:
                show_analyzer_progress("Analyzing image", 70, "Running ML model")
                ml_result, ml_confidence, ml_prediction = self.classify_with_ml_model(image)
            
            # Determine the final result based on available methods
            show_analyzer_progress("Analyzing image", 90, "Finalizing results")
            if vqa_result:
                result = vqa_result
                confidence = vqa_confidence
                method = "VQA-based analysis"
                self.logger.info(f"Using VQA result: {result} ({confidence:.2f}%)")
            elif ml_result and ml_result not in ["Unknown (Could be accessory or other clothing item)", 
                                                "Unknown (No prediction available)"]:
                result = ml_result
                confidence = ml_confidence
                method = f"ML model: {ml_prediction}"
                self.logger.info(f"Using ML result: {result} ({confidence:.2f}%)")
            elif direct_result:
                result = direct_result
                confidence = direct_confidence
                method = "Direct image analysis"
                self.logger.info(f"Using direct analysis: {result} ({confidence:.2f}%)")
            else:
                result = "Unknown (Classification failed)"
                confidence = 0
                method = "All classification methods failed"
            
            # Compile the analysis results
            analysis_results = {
                "classification": result,
                "confidence": float(confidence),
                "method": method,
                "direct_analysis": {
                    "result": direct_result,
                    "confidence": float(direct_confidence),
                    "features": direct_features
                }
            }
            
            # Add ML results if available
            if ml_result:
                analysis_results["ml_analysis"] = {
                    "result": ml_result,
                    "confidence": float(ml_confidence),
                    "raw_prediction": ml_prediction
                }
            
            # Add VQA results if available
            if vqa_result:
                analysis_results["vqa_analysis"] = {
                    "result": vqa_result,
                    "confidence": float(vqa_confidence),
                    "qa_pairs": vqa_details
                }
            
            show_analyzer_progress("Analyzing image", 100, f"Result: {result} ({confidence:.1f}%)", final=True)
            return analysis_results
            
        except Exception as e:
            show_analyzer_progress("Analyzing image", 100, f"Error: {str(e)}", final=True)
            self.logger.error(f"Error analyzing image: {str(e)}")
            return {
                "classification": "Error analyzing image",
                "error": str(e),
                "success": False
            }
    
    def analyze_top_bottom_features(self, image) -> Tuple[str, float, Dict]:
        """Analyze image features specifically for top/bottom determination"""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Key distinguishing features
            features = {
                "height_to_width": height / width,
                "upper_lower_color_diff": 0,
                "vertical_structure": 0,
                "horizontal_structure": 0
            }
            
            # Check if image has typical pants/top shape
            is_likely_pants = height > width * 1.5
            is_likely_top = width >= height * 0.8
            
            # Analyze upper vs lower halves (color difference)
            upper_half = img_array[:height//2, :, :]
            lower_half = img_array[height//2:, :, :]
            color_diff = np.mean(abs(np.mean(upper_half, axis=(0,1)) - np.mean(lower_half, axis=(0,1))))
            features["upper_lower_color_diff"] = float(color_diff)
            
            # Check for vertical structures (typical in pants)
            vertical_edges = 0
            for x in range(1, width-1, max(1, width//50)):  # Sample points for efficiency
                for y in range(1, height-1, max(1, height//100)):
                    if np.sum(abs(img_array[y+1, x] - img_array[y-1, x])) > 100:
                        vertical_edges += 1
            features["vertical_structure"] = float(vertical_edges / (width * height/100) if width * height > 0 else 0)
            
            # Check for horizontal structures (typical in tops, collars, etc.)
            horizontal_edges = 0
            for y in range(1, min(height//2, 100)-1):  # Focus on upper part for collars/necklines
                for x in range(1, width-1, max(1, width//50)):
                    if np.sum(abs(img_array[y, x+1] - img_array[y, x-1])) > 100:
                        horizontal_edges += 1
            features["horizontal_structure"] = float(horizontal_edges / (width * height/200) if width * height > 0 else 0)
            
            # Make top/bottom determination based on features
            top_score = 0
            bottom_score = 0
            
            # Shape-based scores
            if is_likely_pants:
                bottom_score += 30
            if is_likely_top:
                top_score += 25
            
            # Vertical structure (pants have strong vertical lines)
            if features["vertical_structure"] > 0.1:
                bottom_score += 15
            
            # Horizontal structure (tops often have collars, patterns across chest)
            if features["horizontal_structure"] > 0.08:
                top_score += 15
                
            # More even color distribution often indicates tops
            if color_diff < 40:
                top_score += 10
            else:
                bottom_score += 5
                
            # Additional shape patterns
            if features["height_to_width"] < 0.9:
                top_score += 10  # Wide items are often tops
            elif features["height_to_width"] > 1.8:
                bottom_score += 15  # Tall items are often pants
                
            # Calculate confidence
            total = top_score + bottom_score
            confidence = max(top_score, bottom_score) / total * 100 if total > 0 else 50
            
            if top_score >= bottom_score:
                return "Top", confidence, features
            else:
                return "Bottom", confidence, features
                
        except Exception as e:
            self.logger.error(f"Error in direct image analysis: {e}")
            return None, 0, {}

    def classify_with_vqa(self, image) -> Tuple[Optional[str], Optional[float], Optional[List]]:
        """Use visual question-answering to determine clothing type with enhanced details"""
        if not self.vqa_loaded or self.vqa_pipeline is None:
            self.logger.warning("VQA pipeline not available, using fallback methods")
            return None, None, None
        
        try:
            # Enhanced set of questions
            questions = [
                "Is this men's clothing or women's clothing?",  # Gender detection
                "Is this clothing worn on the upper body or lower body?",  # Direct top/bottom question
                "Is this a formal suit or casual clothing?",  # Primary suit detection
                "Is this a complete suit or separate clothing item?",  # Stronger suit verification
                "Is this formal attire or a dress?",  # Extra verification for formal wear/dress
                "What type of clothing is this? Be specific.",
                "What color is this clothing item?",
                "What pattern does this clothing have? (solid, striped, checkered, etc.)",
                "Does this clothing have full sleeves, half sleeves, or is it sleeveless?",
                "Does this clothing have a collar or is it a turtleneck?",
            ]
            
            results = []
            for question in questions:
                result = self.vqa_pipeline(image, question, top_k=1)
                if result and isinstance(result, list) and len(result) > 0:
                    answer = result[0].get('answer', '').lower()
                    score = result[0].get('score', 0)
                    results.append((question, answer, score))
                    self.logger.info(f"VQA Q: '{question}' A: '{answer}' Score: {score:.4f}")
            
            if not results:
                return None, None, None
            
            # Extract all information from answers with suit/formal wear prioritization
            is_men = None
            is_women = None
            is_upper = False
            is_lower = False
            is_pants = False
            is_suit = False
            has_jacket = False
            is_complete_suit = False
            is_formal = False
            is_dress = False
            clothing_type = None
            color = None
            pattern = None
            sleeve_type = None
            
            # CRITICAL: First pass - check specifically for formal wear indicators
            formal_wear_count = 0
            for question, answer, _ in results:
                # Explicit formal wear detection
                if "formal suit or casual" in question:
                    is_suit = "suit" in answer or "formal" in answer
                    is_formal = "formal" in answer
                    if is_suit or is_formal:
                        formal_wear_count += 1
                        
                elif "jacket or blazer" in question:
                    has_jacket = "yes" in answer or "jacket" in answer or "blazer" in answer
                    if has_jacket:
                        formal_wear_count += 1
                        
                elif "complete suit or separate" in question:
                    is_complete_suit = "suit" in answer or "complete" in answer or "yes" in answer
                    if is_complete_suit:
                        formal_wear_count += 1
                        
                elif "formal attire or a dress" in question:
                    is_formal = "formal" in answer or "yes" in answer
                    is_dress = "dress" in answer
                    if is_formal or is_dress:
                        formal_wear_count += 1
                        
                elif "What type of clothing" in question:
                    clothing_type = answer
                    if any(term in answer.lower() for term in ["suit", "formal", "dress", "gown", "tuxedo"]):
                        formal_wear_count += 1
            
            # Strong formal wear detection - overrides all other classifications
            is_formal_wear = formal_wear_count >= 1 or is_suit or is_dress or has_jacket or is_complete_suit
            
            self.logger.info(f"Formal wear indicators: {formal_wear_count}, is_suit: {is_suit}, is_dress: {is_dress}")
            
            # If formal wear detected, override potential pants misclassification
            if is_formal_wear:
                is_suit = True
                is_upper = True
                is_lower = False
                is_pants = False  # Critical: Override any pants classification
                self.logger.info("Formal wear detected - overriding potential pants misclassification")
                
            # Second pass: process all other information without overriding formal wear
            for question, answer, _ in results:
                # Gender detection
                if "men's clothing or women's" in question:
                    is_men = "men" in answer
                    is_women = "women" in answer
                
                # Direct top/bottom detection (but don't override suit)
                if "upper body or lower body" in question and not is_suit:
                    is_upper = "upper" in answer
                    is_lower = "lower" in answer
                
                # Pants/shorts/top categorization (but don't override suit)
                if "pants, shorts, or a top" in question and not is_suit:
                    is_pants = "pants" in answer or "shorts" in answer
                    if not is_upper and not is_lower:
                        is_upper = "top" in answer
                        is_lower = is_pants
                
                # Clothing type
                if "What type of clothing" in question:
                    clothing_type = answer
                    
                    # If we still don't have upper/lower designation and not a suit, try to infer
                    if not is_suit and not is_upper and not is_lower:
                        tops = ['shirt', 'tshirt', 't-shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat', 'polo']
                        bottoms = ['pants', 'trousers', 'shorts', 'jeans', 'sweatpants', 'skirt']
                        
                        is_upper = any(item in answer for item in tops)
                        is_lower = any(item in answer for item in bottoms)
                
                # Color
                if "color" in question:
                    color = answer
                
                # Pattern
                if "pattern" in question:
                    if not any(generic in answer for generic in ["don't know", "cannot tell", "not sure"]):
                        if "solid" in answer or "plain" in answer or "no pattern" in answer:
                            pattern = "solid color"
                        elif "stripe" in answer:
                            pattern = "striped"
                        elif "check" in answer or "plaid" in answer:
                            pattern = "checkered"
                        elif "floral" in answer or "flower" in answer:
                            pattern = "floral"
                        elif "dot" in answer or "polka" in answer:
                            pattern = "polka dot"
                        else:
                            pattern = answer
                
                # Sleeve type
                if "sleeves" in question:
                    sleeve_type = answer
            
            # CRITICAL: Suit always overrides pants classification
            if is_suit:
                is_upper = True
                is_lower = False
                is_pants = False
            
            # Third verification attempt for suits - check combined answers for suit keywords
            if not is_suit:
                all_answers = " ".join([answer for _, answer, _ in results]).lower()
                suit_indicators = ["suit", "formal", "tuxedo", "blazer and pants", "jacket and pants"]
                if any(indicator in all_answers for indicator in suit_indicators):
                    is_suit = True
                    is_upper = True
                    is_lower = False
                    is_pants = False
                    self.logger.info("Suit detected from combined answer analysis")
            
            # If we still don't have a clear top/bottom designation, use direct image analysis
            if not is_upper and not is_lower and not is_suit:
                direct_result, direct_confidence, _ = self.analyze_top_bottom_features(image)
                if direct_result == "Top":
                    is_upper = True
                elif direct_result == "Bottom":
                    is_lower = True
            
            # Formulate gender part of classification
            gender_prefix = ""
            if is_men:
                gender_prefix = "Men's "
            elif is_women:
                gender_prefix = "Women's "
            
            # Determine main classification with formal wear priority
            if is_suit or is_formal_wear:
                if is_dress:
                    classification = f"{gender_prefix}Top (Full body garment: Dress)"
                else:
                    classification = f"{gender_prefix}Top (Full body garment: Suit)"
                self.logger.info(f"Final classification as formal wear: {classification}")
            elif is_upper:
                classification = f"{gender_prefix}Top"
                if clothing_type and not any(term in str(clothing_type).lower() for term in ["suit", "dress"]):
                    classification += f" ({clothing_type})"
            elif is_lower or is_pants:
                classification = f"{gender_prefix}Bottom"
                if clothing_type:
                    classification += f" ({clothing_type})"
            else:
                # Last attempt - analyze all answers
                all_answers = " ".join([answer for _, answer, _ in results])
                
                if "suit" in all_answers.lower():
                    classification = f"{gender_prefix}Top (Full body garment: Suit)"
                elif any(term in all_answers.lower() for term in ['top', 'shirt', 'jacket', 'sweater']):
                    classification = f"{gender_prefix}Top"
                    if clothing_type:
                        classification += f" ({clothing_type})"
                elif any(term in all_answers.lower() for term in ['bottom', 'pants', 'trousers', 'shorts']):
                    classification = f"{gender_prefix}Bottom"
                    if clothing_type:
                        classification += f" ({clothing_type})"
                else:
                    # Final resort: direct image analysis
                    direct_result, _, _ = self.analyze_top_bottom_features(image)
                    classification = f"{gender_prefix}{direct_result}"
            
            # Add additional details
            if color:
                classification += f", {color}"
            if pattern:
                classification += f", {pattern} pattern"
            if sleeve_type and ("Top" in classification or "suit" in classification.lower()):
                classification += f", {sleeve_type}"
            
            # Calculate confidence based on the highest relevant score
            top_bottom_scores = []
            for _, _, score in results[:6]:  # Consider the first 6 questions which focus on categorization
                top_bottom_scores.append(score)
            
            if top_bottom_scores:
                highest_confidence = max(top_bottom_scores) * 100
            else:
                highest_confidence = 70  # Default if no questions about top/bottom were answered
            
            return classification, highest_confidence, results
            
        except Exception as e:
            self.logger.error(f"Error in VQA classification: {str(e)}")
            return None, None, None

    def classify_with_ml_model(self, image) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Classify an image using the ML model"""
        if not self.model_loaded or self.feature_extractor is None or self.model is None:
            return None, None, None
            
        try:
            # Preprocess image for ML model
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
            
            # Get label and confidence
            if hasattr(self.model.config, 'id2label') and predicted_class_idx in self.model.config.id2label:
                ml_prediction = self.model.config.id2label[predicted_class_idx]
            else:
                ml_prediction = str(predicted_class_idx)  # Use class index if no label mapping
            
            ml_confidence = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_class_idx].item() * 100
            ml_result = self.map_to_top_bottom(ml_prediction)
            
            self.logger.info(f"ML model prediction: {ml_prediction} -> {ml_result} ({ml_confidence:.2f}%)")
            
            return ml_result, ml_confidence, ml_prediction
            
        except Exception as e:
            self.logger.error(f"Error with ML model classification: {str(e)}")
            return None, None, None

    def map_to_top_bottom(self, prediction) -> str:
        """Map generic clothing categories to top/bottom with expanded terms"""
        # Very comprehensive clothing terms list for better matching
        tops = [
            'shirt', 'tshirt', 't-shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat', 'polo', 
            'top', 'sweatshirt', 'tank', 'tank top', 'jersey', 'cardigan', 'pullover', 'vest',
            'blazer', 'jumper', 'tunic', 'dress shirt', 'button-up', 'button-down', 'tee', 
            'pullover', 'coat', 'anorak', 'shirt', 'tee', 'jumper', 'sweater', 'sweatshirt',
            'turtleneck', 'pullover', 'v-neck', 'top', 'henley', 'button', 'tshirt', 'polo',
            'long sleeve', 'dress-shirt', 'blouse', 'tee shirt', 'formal shirt', 'flannel'
        ]
        
        bottoms = [
            'pants', 'trousers', 'shorts', 'jeans', 'sweatpants', 'skirt', 'leggings',
            'joggers', 'chinos', 'slacks', 'khakis', 'cargo', 'cargo pants', 'capris',
            'culottes', 'bermudas', 'corduroys', 'sweat shorts', 'track pants', 'trouser',
            'pant', 'jean', 'denim', 'dress pant', 'jogger', 'sweat pant', 'trackpant',
            'boardshorts', 'pant', 'legwear', 'bottom', 'capri', 'trouser'
        ]
        
        # Terms for full body garments (special handling)
        full_body = [
            'suit', 'dress', 'jumpsuit', 'overall', 'onesie', 'romper', 'costume',
            'uniform', 'tuxedo', 'formal wear', 'gown', 'robe', 'kimono'
        ]
        
        # Handle empty predictions
        if not prediction:
            return "Unknown (No prediction available)"
        
        # Convert to string and lowercase for matching
        prediction_str = str(prediction).lower()
        
        self.logger.info(f"Mapping raw prediction: {prediction_str}")
        
        # Special case for suits - considering them primarily as tops
        if any(item in prediction_str for item in full_body):
            # Suits and formal wear in men's clothing are typically categorized by the jacket/top part
            self.logger.info(f"Full body garment detected: {prediction_str}")
            return "Top (Full body garment)"
        
        # Check for direct matches in our lists
        if any(item in prediction_str for item in tops):
            return "Top"
        elif any(item in prediction_str for item in bottoms):
            return "Bottom"
        
        # Check for fashion-mnist numeric classes
        try:
            if prediction_str.isdigit():
                prediction_num = int(prediction_str)
                # Common fashion datasets like Fashion-MNIST:
                # 0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat, 
                # 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot
                tops_indices = [0, 2, 3, 4, 6]  # T-shirt/top, Pullover, Dress, Coat, Shirt
                bottoms_indices = [1]  # Trouser
                if prediction_num in tops_indices:
                    return "Top"
                elif prediction_num in bottoms_indices:
                    return "Bottom"
        except:
            pass
        
        # Check if we can infer from partial matches
        for top_item in tops:
            if top_item in prediction_str or prediction_str in top_item:
                return "Top (inferred)"
        
        for bottom_item in bottoms:
            if bottom_item in prediction_str or prediction_str in bottom_item:
                return "Bottom (inferred)"
        
        # If all else fails, return the direct classifier result
        try:
            if isinstance(prediction, bytes):
                image = Image.open(io.BytesIO(prediction))
            elif isinstance(prediction, str) and os.path.exists(prediction):
                image = Image.open(prediction)
            else:
                return "Unknown (direct classification failed)"
                
            result, _ = self.direct_classifier.classify(image)
            return result
        except:
            return "Unknown (direct classification failed)"

    def _ensure_pil_image(self, image_data) -> Image.Image:
        """Convert various image inputs to PIL Image object"""
        if isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, str):
            if os.path.isfile(image_data):
                return Image.open(image_data)
            else:
                raise ValueError(f"File not found: {image_data}")
        else:
            raise TypeError(f"Unsupported image data type: {type(image_data)}")


class DirectTopBottomClassifier:
    """A direct classifier for distinguishing tops from bottoms based on image features"""
    
    def __init__(self):
        # Pre-trained simple classifier
        self.tops_features = {
            "avg_height_ratio": 0.7,  # tops usually have smaller height/width ratio
            "color_variance_upper": 0.6,  # tops often have more details in upper part
        }
        # Fix the incomplete line
        self.bottoms_features = {
            "avg_height_ratio": 1.5,  # bottoms usually have larger height/width ratio
            "color_variance_lower": 0.4,  # bottoms often have more uniform color
        }
        
    def classify(self, image):
        """Classify an image as top or bottom based on shape and color distribution"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Basic shape analysis
            height_ratio = height / width
            
            # Simple rule-based classification
            if height_ratio > 1.2:  # Taller than wide - likely bottoms
                return "Bottom", 70.0
            else:  # Wider than tall or square - likely tops
                return "Top", 65.0
                
        except Exception as e:
            print(f"Error in direct classification: {e}")
            return "Unknown", 0.0