"""
Metadata extraction and parsing service
"""

def parse_clothing_analysis(classification):
    """
    Parse a detailed clothing classification string into separate components.
    
    For example:
    "Men's Top (Full body garment: Suit), gray, solid color pattern, long sleeve"
    becomes:
    {
        "wearable": "Top",
        "sex": "Men's",
        "color": "gray",
        "pattern": "solid color",
        "hand": "long sleeve",
        "costume": "formal"
    }
    """
    result = {
        "wearable": "unknown",
        "sex": "unknown",
        "color": "unknown",
        "pattern": "unknown",
        "hand": "unknown",
        "costume": "unknown"
    }
    
    if not classification or classification == "unknown":
        return result
    
    # Extract gender/sex information
    if classification.startswith("Men's"):
        result["sex"] = "Men's"
        classification = classification[6:].strip()  # Remove "Men's " prefix
    elif classification.startswith("Women's"):
        result["sex"] = "Women's"
        classification = classification[8:].strip()  # Remove "Women's " prefix
    
    # Extract wearable type (Top or Bottom)
    if classification.startswith("Top"):
        result["wearable"] = "top wearable"
    elif classification.startswith("Bottom"):
        result["wearable"] = "bottom wearable"
    
    # Extract full body garment type (usually in parentheses)
    if "Full body garment:" in classification:
        result["costume"] = "formal"
        if "Dress" in classification:
            result["wearable"] = "dress"  # Special category for dresses
    
    # Extract color, pattern, and sleeve information
    parts = classification.split(',')
    for part in parts:
        part = part.strip().lower()
        
        # Check for colors
        colors = ["black", "white", "red", "blue", "green", "yellow", "purple", "pink", "orange", 
                 "brown", "gray", "grey", "navy", "teal", "maroon", "olive", "tan"]
        for color in colors:
            if color in part:
                result["color"] = color
                break
        
        # Check for patterns
        if "pattern" in part or any(p in part for p in ["solid", "striped", "checkered", "plaid", "floral", "polka dot", "dotted"]):
            if "solid" in part:
                result["pattern"] = "solid"
            elif "striped" in part:
                result["pattern"] = "striped"
            elif "checkered" in part or "plaid" in part:
                result["pattern"] = "checkered"
            elif "floral" in part:
                result["pattern"] = "floral"
            elif "polka dot" in part or "dotted" in part:
                result["pattern"] = "polka dot"
            else:
                result["pattern"] = part
        
        # Check for sleeve type
        if "sleeve" in part:
            if "long" in part:
                result["hand"] = "full hand"
            elif "short" in part or "half" in part:
                result["hand"] = "half hand"
            elif "no" in part or "sleeveless" in part:
                result["hand"] = "no hand"
            else:
                result["hand"] = part
    
    return result
