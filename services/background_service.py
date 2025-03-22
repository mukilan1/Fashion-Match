"""
Background removal service
"""
import os
import io
from PIL import Image
from rembg import remove

from utils.progress import show_progress

def remove_background(input_image):
    """
    Removes the background from an image and returns the processed image.
    
    Args:
        input_image: PIL Image or file-like object
    Returns:
        PIL Image with transparent background
    """
    try:
        show_progress("Removing background", 10, "Preparing image")
        # Convert PIL Image to bytes if needed
        if isinstance(input_image, Image.Image):
            img_byte_arr = io.BytesIO()
            input_image.save(img_byte_arr, format='PNG')
            img_data = img_byte_arr.getvalue()
            show_progress("Removing background", 30, "Image converted")
        else:
            # Assume input is a file path
            with open(input_image, 'rb') as img_file:
                img_data = img_file.read()
                show_progress("Removing background", 30, "Image loaded")
        
        # Process the image to remove background
        show_progress("Removing background", 50, "Processing")
        output_data = remove(img_data)
        show_progress("Removing background", 80, "Background removed")
        
        # Convert back to PIL Image
        result_image = Image.open(io.BytesIO(output_data))
        show_progress("Removing background", 100, "Complete", final=True)
        return result_image
    except Exception as e:
        show_progress("Removing background", 100, f"Failed: {str(e)}", final=True)
        # Return the original image if background removal fails
        if isinstance(input_image, str) and os.path.exists(input_image):
            return Image.open(input_image)
        return input_image
