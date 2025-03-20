from PIL import Image
from transformers import pipeline

# Initialize the VQA pipeline
vqa_pipeline = pipeline("visual-question-answering")

def analyze_image(image, question="What is this?"):
    """
    Analyze an image with a given question using VQA.
    
    Args:
        image: PIL Image object or path to image file
        question: String containing the question to ask about the image
        
    Returns:
        dict: Contains 'answer', 'confidence' and 'success' keys
    """
    try:
        # If image is a string (path), open it
        if isinstance(image, str):
            image = Image.open(image)
            
        # Use the VQA pipeline
        result = vqa_pipeline(image, question, top_k=1)
        
        # Extract the answer and confidence
        if result and len(result) > 0:
            answer = result[0].get('answer', 'unknown')
            confidence = result[0].get('score', 0) * 100
            return {
                'success': True,
                'answer': answer,
                'confidence': confidence
            }
        else:
            return {
                'success': False,
                'error': 'No answer found'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Example usage
if __name__ == "__main__":
    # This will run when the script is executed directly
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        question = sys.argv[2] if len(sys.argv) > 2 else "What is this?"
        
        print(f"Analyzing image: {image_path}")
        print(f"Question: {question}")
        
        result = analyze_image(image_path, question)
        
        if result['success']:
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}%")
        else:
            print(f"Error: {result['error']}")
    else:
        print("Usage: python vqa_core.py <image_path> [question]")
