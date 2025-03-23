import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ClothingItem:
    """Data class to represent a clothing item"""
    filename: str
    label: str
    wearable: str
    costume: str = "No"
    sex: str = "Unisex"
    color: str = "Unknown"
    pattern: str = "Solid"
    color_detail: Optional[str] = None
    is_valid: bool = True

def get_all_images() -> List[Dict[str, Any]]:
    """
    Get all clothing items from the existing labels.json file
    
    Returns:
        List[Dict]: A list of clothing item dictionaries
    """
    # Load the labels.json file from the project root
    labels_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'labels.json')
    
    try:
        with open(labels_path, 'r') as f:
            data = json.load(f)
            
        # Convert the dictionary to a list of items with filenames included
        items = []
        for filename, item_data in data.items():
            item = dict(item_data)  # Create a copy of the item data
            item['filename'] = filename  # Add the filename to the item
            items.append(item)
            
        return items
    except Exception as e:
        print(f"Error loading clothing data from labels.json: {str(e)}")
        return []

def create_clothing_item(item_data: Dict[str, Any]) -> ClothingItem:
    """Create a ClothingItem from a dictionary of data"""
    return ClothingItem(
        filename=item_data.get('filename', ''),
        label=item_data.get('label', 'Unknown'),
        wearable=item_data.get('wearable', 'Unknown'),
        costume=item_data.get('costume', 'No'),
        sex=item_data.get('sex', 'Unisex'),
        color=item_data.get('color', 'Unknown'),
        pattern=item_data.get('pattern', 'Solid'),
        color_detail=item_data.get('color_detail', None),
        is_valid=item_data.get('is_valid', True)
    )

def classify_wearable(wearable: str) -> str:
    """
    Classify wearable type as 'top', 'bottom', or 'other'
    
    Args:
        wearable: The wearable type string
        
    Returns:
        str: 'top', 'bottom', or 'other'
    """
    wearable = wearable.lower()
    
    # Top categories
    top_keywords = ['top', 'shirt', 'blouse', 'tshirt', 't-shirt', 'sweater', 
                    'jacket', 'hoodie', 'blazer', 'tank', 'sweatshirt', 'cardigan']
    
    # Bottom categories
    bottom_keywords = ['bottom', 'pants', 'skirt', 'shorts', 'jeans', 'trouser', 
                       'leggings', 'sweatpants', 'jogger']
    
    for keyword in top_keywords:
        if keyword in wearable:
            return 'top'
            
    for keyword in bottom_keywords:
        if keyword in wearable:
            return 'bottom'
            
    return 'other'

def is_top_wearable(wearable: str) -> bool:
    """Check if a wearable is a top item based on its description"""
    if not wearable:
        return False
    
    wearable = wearable.lower()
    return 'top' in wearable or any(item in wearable for item in 
                                   ['shirt', 'blouse', 't-shirt', 'tshirt', 'sweater', 'jacket', 'hoodie'])

def is_bottom_wearable(wearable: str) -> bool:
    """Check if a wearable is a bottom item based on its description"""
    if not wearable:
        return False
    
    wearable = wearable.lower()
    return 'bottom' in wearable or any(item in wearable for item in 
                                      ['pant', 'jean', 'trouser', 'skirt', 'short'])
