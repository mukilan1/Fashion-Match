"""
Helper functions for templates
"""

def get_nav_links():
    """Return navigation links for templates"""
    return {
        'home': '/',
        'match': '/match',
        'pose_rigging': '/pose_rigging',
        'mediapipe_gallery': '/mediapipe_gallery',
    }

def format_metadata_for_display(item):
    """Format metadata for display in templates"""
    if not item:
        return {}
        
    # Format and clean up metadata for display
    return {
        'label': item.get('label', 'Unknown Item'),
        'wearable': item.get('wearable', 'Unknown').replace('_', ' ').title(),
        'costume': item.get('costume_display', item.get('costume', 'Unknown')).title(),
        'color': item.get('color', '#cccccc'),
        'pattern': item.get('pattern', 'Unknown').title(),
        'sex': item.get('sex', 'Unknown'),
        'sleeve': item.get('hand', 'Unknown').replace('_', ' ').title()
    }
