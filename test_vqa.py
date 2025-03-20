from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import pipeline
import os

app = Flask(__name__)

# Initialize the VQA pipeline exactly as in the example
vqa_pipeline = pipeline("visual-question-answering")

# Simple form for uploading an image and entering a question
@app.route('/', methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>VQA Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            .result { margin-top: 20px; padding: 10px; background-color: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>Visual Question Answering Test</h1>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <div>
                <label for="image">Upload Image:</label>
                <input type="file" name="image" accept="image/*" required>
            </div>
            <div style="margin-top: 10px;">
                <label for="question">Question:</label>
                <input type="text" name="question" value="What is this?" required>
            </div>
            <div style="margin-top: 10px;">
                <button type="submit">Analyze</button>
            </div>
        </form>
    </body>
    </html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Get the image file
    image_file = request.files['image']
    
    # Get the question (default to clothing type if not provided)
    question = request.form.get('question', 'What type of clothing is this?')
    
    try:
        # Open the image with PIL
        image = Image.open(image_file)
        
        # Use the VQA pipeline exactly as in the example
        result = vqa_pipeline(image, question, top_k=1)
        
        # Extract the answer and confidence
        if result and len(result) > 0:
            answer = result[0].get('answer', 'unknown')
            confidence = result[0].get('score', 0) * 100
            
            # Return a simple result page
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>VQA Result</title>
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .result {{ margin-top: 20px; padding: 10px; background-color: #f0f0f0; }}
                </style>
            </head>
            <body>
                <h1>Analysis Result</h1>
                <div class="result">
                    <p><strong>Question:</strong> {question}</p>
                    <p><strong>Answer:</strong> {answer}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                <p><a href="/">Try another image</a></p>
            </body>
            </html>
            '''
        else:
            return jsonify({'error': 'No answer found'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting VQA test server at http://127.0.0.1:5000/")
    app.run(debug=True)
