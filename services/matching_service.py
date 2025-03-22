"""
Clothing matching service
"""
import re
import ollama
from sentence_transformers import SentenceTransformer, util

from config import SBERT_MODEL_NAME, OLLAMA_MODEL_NAME, COLOR_PAIRS, MIN_MATCH_THRESHOLD

# Initialize SentenceTransformer model globally
try:
    print("Initializing SentenceTransformer model...")
    SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
    print("SentenceTransformer model initialized successfully")
except Exception as e:
    print(f"Error initializing SentenceTransformer: {str(e)}")
    SBERT_MODEL = None

def generate_match_reason(new_item, match_item, similarity_score):
    """Generate a concise explanation of why two clothing items match well together"""
    reasons = []
    
    # Check color compatibility
    if new_item['color'] == match_item['color']:
        reasons.append(f"matching {new_item['color']} color")
    else:
        reasons.append(f"complementary {new_item['color']} and {match_item['color']} colors")
    
    # Check pattern compatibility
    if new_item['pattern'] == match_item['pattern']:
        if new_item['pattern'] != "unknown" and new_item['pattern'] != "plain":
            reasons.append(f"coordinated {new_item['pattern']} patterns")
    
    # Check style consistency
    if new_item['costume'] == match_item['costume'] and new_item['costume'] != "unknown":
        reasons.append(f"consistent {new_item['costume']} style")
    
    # Check gender appropriateness
    if new_item['sex'] == match_item['sex'] and new_item['sex'] != "unknown":
        reasons.append(f"suitable for {new_item['sex']}")
    
    # Create final reason text based on similarity score
    if similarity_score > 90:
        quality = "perfect"
    elif similarity_score > 75:
        quality = "excellent"
    elif similarity_score > 60:
        quality = "good"
    else:
        quality = "decent"
    
    # Combine all reasons into a single, concise line
    if reasons:
        joined_reasons = ", ".join(reasons[:2])  # Limit to top 2 reasons for brevity
        return f"A {quality} match with {joined_reasons}."
    else:
        return f"A {quality} match based on overall compatibility."

def generate_detailed_match_reason(top_item, bottom_item, score, details):
    """
    Generate a detailed reason for why two items match well, with fallback if Ollama is unavailable.
    This function provides more specific matching reasons.
    """
    # First try to use dynamic reason generation via Ollama
    try:
        prompt = (
            f"Why do these clothes match well? Top: {top_item['label']} ({top_item['costume']}, {top_item['pattern']}, "
            f"{top_item['color']}). Bottom: {bottom_item['label']} ({bottom_item['costume']}, {bottom_item['pattern']}, "
            f"{bottom_item['color']}). Match score: {score:.2f}. Explain in one sentence what makes them a good outfit combination."
        )
        
        response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        result_text = response.get('message', {}).get('content', "").strip()
        
        # Clean up the response
        cleaned_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()
        
        if not cleaned_text or len(cleaned_text) < 10:
            raise ValueError("Generated text too short")
            
        return cleaned_text.split("\n")[0]
        
    except Exception as e:
        print(f"Ollama match reason generation failed: {str(e)}")
        
        # More detailed fallback logic based on specific attributes
        reasons = []
        
        # Check for color compatibility
        if top_item['color'] == bottom_item['color'] and top_item['color'] != "unknown":
            reasons.append(f"matching {top_item['color']} color")
        elif top_item['color'] != "unknown" and bottom_item['color'] != "unknown":
            reasons.append(f"{top_item['color']} top complements {bottom_item['color']} bottom")
        
        # Check for style compatibility
        if top_item['costume'] == bottom_item['costume'] and top_item['costume'] != "unknown":
            reasons.append(f"consistent {top_item['costume']} style")
        
        # Check for pattern compatibility
        if top_item['pattern'] == "solid" and bottom_item['pattern'] != "solid":
            reasons.append(f"solid top balances {bottom_item['pattern']} bottom")
        elif bottom_item['pattern'] == "solid" and top_item['pattern'] != "solid":
            reasons.append(f"{top_item['pattern']} top balances solid bottom")
        
        # If we found specific reasons, use them
        if reasons:
            return f"These items pair well with {' and '.join(reasons)} (score: {score:.2f})"
        else:
            return f"This {top_item['label']} and {bottom_item['label']} create a coordinated outfit (score: {score:.2f})"

def find_best_match(new_item, stored_items):
    """
    Find the best matching item for a given item.
    
    Args:
        new_item: Dictionary containing new item data
        stored_items: Dictionary of stored items to check against
    
    Returns:
        dict: Best match information
    """
    # Get properties of the uploaded item
    wearable_position = new_item.get("wearable", "unknown")
    primary_color = new_item.get("color", "unknown")
    costume = new_item.get("costume", "unknown")
    pattern = new_item.get("pattern", "unknown")
    gender = new_item.get("sex", "unknown")
    label = new_item.get("label", "unknown")
    
    # Debug print item info
    print(f"Finding match for: {label} ({wearable_position}), color: {primary_color}, pattern: {pattern}")
    
    # Determine what type of item we need to match with
    target_wearable_type = "bottom wearable" if wearable_position == "top wearable" else "top wearable"
    print(f"Looking for {target_wearable_type} to match with {wearable_position}")
    
    # Create the embedding for the new item
    new_text = f"{label} {wearable_position} {costume} {pattern} {primary_color} {gender}"
    
    # Use global model if available, otherwise create a temporary one
    if SBERT_MODEL is None:
        print("Warning: Using one-time SentenceTransformer model - this may cause performance issues")
        try:
            temp_model = SentenceTransformer(SBERT_MODEL_NAME)
            new_embedding = temp_model.encode(new_text, convert_to_tensor=True)
        except Exception as e:
            print(f"Error creating temporary model: {e}")
            # Fallback to simple matching if model fails
            return fallback_match(new_item, stored_items, target_wearable_type)
    else:
        new_embedding = SBERT_MODEL.encode(new_text, convert_to_tensor=True)
    
    candidates = []
    for filename, data in stored_items.items():
        if filename == new_item.get("filename"):  # Skip matching with self
            continue
            
        # Only consider items of the complementary wearable type
        cand_wearable = data.get("wearable", "").lower()
        if cand_wearable != target_wearable_type:
            continue  # Skip items that aren't the right type (top or bottom)
            
        cand_text = f"{data.get('label','')} {data.get('wearable','')} {data.get('costume','')} {data.get('pattern','')} {data.get('color','')} {data.get('sex','')}"
        candidates.append((filename, cand_text, data))
    
    print(f"Found {len(candidates)} potential matching candidates")
    
    if not candidates:
        return None, f"No {target_wearable_type} items found to match with your {wearable_position}"
        
    best_candidate, best_score = None, -1
    
    # Get properties of the item
    item_color = primary_color.lower()
    item_style = costume.lower()
    item_gender = gender.lower()
    
    for filename, cand_text, data in candidates:
        # Calculate semantic similarity using SBERT
        if SBERT_MODEL is None:
            cand_embedding = temp_model.encode(cand_text, convert_to_tensor=True)
        else:
            cand_embedding = SBERT_MODEL.encode(cand_text, convert_to_tensor=True)
            
        # Get base similarity score
        base_score = util.cos_sim(new_embedding, cand_embedding).item()
        
        # Apply fashion-specific matching bonuses
        # Extract candidate attributes
        cand_color = data.get("color", "").lower()
        cand_style = data.get("costume", "").lower()
        cand_gender = data.get("sex", "").lower()
        cand_pattern = data.get("pattern", "").lower()
        
        # Color compatibility bonus
        color_bonus = 0
        if item_color in COLOR_PAIRS and cand_color in COLOR_PAIRS.get(item_color, []):
            color_bonus = 0.15
            
        # Style consistency bonus
        style_bonus = 0
        if item_style == cand_style and item_style != "unknown":
            style_bonus = 0.1
        
        # Gender consistency bonus
        gender_bonus = 0
        if item_gender == cand_gender or item_gender == "unisex" or cand_gender == "unisex":
            gender_bonus = 0.05
            
        # Pattern compatibility - solid with pattern often works well
        pattern_bonus = 0
        if (pattern == "solid" and cand_pattern != "solid" and cand_pattern != "unknown") or \
           (cand_pattern == "solid" and pattern != "solid" and pattern != "unknown"):
            pattern_bonus = 0.05
        
        # Calculate final score with bonuses
        total_score = base_score + color_bonus + style_bonus + gender_bonus + pattern_bonus
        
        # Track details for explanation
        match_details = {
            "base_score": base_score,
            "color_bonus": color_bonus,
            "style_bonus": style_bonus,
            "gender_bonus": gender_bonus,
            "pattern_bonus": pattern_bonus,
            "total_score": total_score,
            "explanation": []
        }
        
        # Add explanations for bonuses
        if color_bonus > 0:
            match_details["explanation"].append(f"{item_color} works well with {cand_color}")
        if style_bonus > 0:
            match_details["explanation"].append(f"both are {item_style} style")
        if gender_bonus > 0:
            match_details["explanation"].append(f"gender compatible")
        if pattern_bonus > 0:
            match_details["explanation"].append(f"{pattern} pairs nicely with {cand_pattern}")
        
        print(f"Candidate: {filename}, Score: {total_score}")
        
        # Track best match with all details
        if total_score > best_score:
            best_score = total_score
            best_candidate = {
                "filename": filename,
                "data": data,
                "score": total_score * 100,  # Convert to percentage
                "match_details": match_details
            }
    
    if best_candidate:
        print(f"Best match found: {best_candidate['filename']} with score {best_candidate['score']}")
        return best_candidate, None
    else:
        return None, "No suitable match found for your item"

# Add a simple fallback matching function when models fail
def fallback_match(new_item, stored_items, target_wearable_type):
    """Simple fallback matching when ML models fail"""
    print("Using fallback matching logic")
    candidates = []
    
    for filename, data in stored_items.items():
        if filename == new_item.get("filename"):  # Skip matching with self
            continue
            
        # Only consider items of the complementary wearable type
        cand_wearable = data.get("wearable", "").lower()
        if cand_wearable != target_wearable_type:
            continue
            
        # Simple scoring - match color, pattern, etc.
        score = 0
        if data.get("color") == new_item.get("color"):
            score += 40
        if data.get("costume") == new_item.get("costume"):
            score += 20
        if data.get("sex") == new_item.get("sex") or data.get("sex") == "unisex":
            score += 10
            
        candidates.append({
            "filename": filename,
            "data": data,
            "score": score,
            "match_details": {"explanation": ["Simple matching used due to model failure"]}
        })
    
    # Sort by score and take best
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    if candidates:
        return candidates[0], None
    else:
        return None, f"No {target_wearable_type} items found to match with your item"

def auto_match(labels):
    """
    Auto-match tops with bottoms, ensuring appropriate matching.
    
    Args:
        labels: Dictionary of labels
    
    Returns:
        list: Auto-generated matches
    """
    tops = []
    bottoms = []
    
    # Filter and categorize clothing items
    for filename, data in labels.items():
        wearable_type = data.get("wearable", "").lower()
        
        # Create richer text representation that includes all meaningful attributes
        text = f"{data.get('label','')} {data.get('costume','')} {data.get('pattern','')} {data.get('color','')} {data.get('sex','')}"
        
        # Strict categorization to ensure proper matching
        if wearable_type == "top wearable":
            tops.append({"filename": filename, "text": text, "data": data})
        elif wearable_type == "bottom wearable":
            bottoms.append({"filename": filename, "text": text, "data": data})
    
    # Check if we have sufficient items for matching
    if not tops or not bottoms:
        return []
        
    auto_matches = []
    
    # Use global model if available, otherwise initialize a temporary one
    if SBERT_MODEL is not None:
        embedding_model = SBERT_MODEL
    else:
        embedding_model = SentenceTransformer(SBERT_MODEL_NAME)
    
    for top in tops:
        # Pre-encode top embedding for efficiency
        top_emb = embedding_model.encode(top["text"], convert_to_tensor=True)
        best_match = None
        best_score = -1
        
        top_color = top["data"].get("color", "").lower()
        top_style = top["data"].get("costume", "").lower()
        top_gender = top["data"].get("sex", "").lower()
        
        for bottom in bottoms:
            # Apply business logic for compatibility
            bottom_color = bottom["data"].get("color", "").lower()
            bottom_style = bottom["data"].get("costume", "").lower()
            bottom_gender = bottom["data"].get("sex", "").lower()
            
            # Basic color compatibility check - apply a bonus for compatible colors
            color_bonus = 0
            if top_color in COLOR_PAIRS and bottom_color in COLOR_PAIRS.get(top_color, []):
                color_bonus = 0.15
                
            # Style consistency check - formal with formal, casual with casual
            style_bonus = 0
            if top_style == bottom_style and top_style != "unknown":
                style_bonus = 0.1
            
            # Gender consistency check
            gender_bonus = 0
            if top_gender == bottom_gender or top_gender == "unisex" or bottom_gender == "unisex":
                gender_bonus = 0.05
            
            # Encode bottom and calculate semantic similarity
            bottom_emb = embedding_model.encode(bottom["text"], convert_to_tensor=True)
            base_score = util.cos_sim(top_emb, bottom_emb).item()
            
            # Add bonuses to semantic score for final match score
            total_score = base_score + color_bonus + style_bonus + gender_bonus
            
            # Track the best match
            if total_score > best_score:
                best_score = total_score
                best_match = {
                    "bottom": bottom,
                    "score": total_score,
                    "semantic_score": base_score,
                    "color_bonus": color_bonus,
                    "style_bonus": style_bonus,
                    "gender_bonus": gender_bonus
                }
        
        # Only include matches that meet the minimum threshold
        if best_match and best_match["score"] >= MIN_MATCH_THRESHOLD:
            # Generate a specific reason for why these items match
            try:
                match_details = f"(Base:{best_match['semantic_score']:.2f}, Color:{best_match['color_bonus']:.2f}, Style:{best_match['style_bonus']:.2f}, Gender:{best_match['gender_bonus']:.2f})"
                reason_text = generate_detailed_match_reason(top["data"], best_match["bottom"]["data"], best_match["score"], match_details)
            except Exception as e:
                print(f"Error generating match reason: {str(e)}")
                reason_text = f"These items have a match score of {best_match['score']:.2f} {match_details}"
                
            auto_matches.append({
                "top": top,
                "bottom": best_match["bottom"],
                "score": best_match["score"],
                "reason": reason_text
            })
    
    # Sort matches by score, highest first
    auto_matches.sort(key=lambda x: x["score"], reverse=True)
    
    return auto_matches
