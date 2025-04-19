"""
Utility module for classifying clothing items as top or bottom wearables.
This can be used independently of the main image analyzer.
"""

class WearableClassifier:
    """Simple classifier for determining if a clothing item is a top or bottom wearable"""
    
    def __init__(self):
        self.top_items = ["shirt", "t-shirt", "blouse", "jacket", "coat", "sweater", "hoodie", 
                         "top", "sweatshirt", "cardigan", "polo", "tank", "vest", "jersey", 
                         "tunic", "turtleneck", "button-down"]
        
        self.bottom_items = ["pants", "jeans", "shorts", "skirt", "trousers", "leggings", 
                            "joggers", "slacks", "chinos", "bottom", "sweatpants", "tights",
                            "capris", "culottes", "briefs"]
        
        self.full_body_items = ["dress", "jumpsuit", "romper", "suit", "overall", "gown", 
                               "onesie", "bodysuit", "co stume", "court suit"]
    
    def classify(self, label: str) -> str:
        """
        Classify a clothing item based on its label.
        
        Args:
            label: String containing the clothing description/label
            
        Returns:
            str: 'top wearable', 'bottom wearable', 'full body', or 'unknown'
        """
        if not label or label == "unknown":
            return "unknown"
        
        # Convert to lowercase for case-insensitive matching
        label_lower = label.lower()
        
        # Special case handling for suits - they should never be bottom wearables
        if "suit" in label_lower:
            return "full body"
            
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
        
        # If no match found
        return "unknown"

# Simple function to use directly
def classify_wearable(label: str) -> str:
    """
    Quick classify function that can be used without instantiating the class.
    
    Args:
        label: String containing the clothing description
        
    Returns:
        str: Wearable position classification
    """
    classifier = WearableClassifier()
    return classifier.classify(label)

# Usage example
if __name__ == "__main__":
    # Test with some examples
    test_items = [
        "Blue T-Shirt",
        "Black Jeans",
        "Summer Dress",
        "Red Hoodie",
        "Pleated Skirt",
        "Evening Gown",
        "Unknown Item"
    ]
    
    for item in test_items:
        position = classify_wearable(item)
        print(f"Item: {item} → Position: {position}")
