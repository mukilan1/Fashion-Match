from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import json
from werkzeug.utils import secure_filename
from image_analyzer import ClothingAnalyzer

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ANALYSIS_CACHE = 'static/analysis_cache'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create necessary directories
for directory in [UPLOAD_FOLDER, ANALYSIS_CACHE]:
    os.makedirs(directory, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANALYSIS_CACHE'] = ANALYSIS_CACHE
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize the clothing analyzer
clothing_analyzer = ClothingAnalyzer(cache_dir=ANALYSIS_CACHE)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_all_analyses():
    """Load all cached analysis results"""
    analyses = {}
    try:
        for filename in os.listdir(app.config['ANALYSIS_CACHE']):
            if filename.endswith('.json'):
                image_name = filename[:-5]  # Remove .json extension
                with open(os.path.join(app.config['ANALYSIS_CACHE'], filename), 'r') as f:
                    analyses[image_name] = json.load(f)
    except Exception as e:
        print(f"Error loading analyses: {str(e)}")
    return analyses

@app.route('/')
def index():
    # Get all uploaded images to display
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    images = [img for img in images if img.split('.')[-1].lower() in ALLOWED_EXTENSIONS]
    
    # Pre-analyze any images that haven't been analyzed yet
    auto_analyze_images(images)
    
    # Load all analysis results
    analysis_results = load_all_analyses()
    
    return render_template('index.html', images=images, analysis_results=analysis_results)

def auto_analyze_images(images):
    """Automatically analyze any images that don't have cached results"""
    for image in images:
        cache_file = os.path.join(app.config['ANALYSIS_CACHE'], image + ".json")
        if not os.path.exists(cache_file):
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], image)
                clothing_analyzer.analyze_image(file_path)
                print(f"Auto-analyzed: {image}")
            except Exception as e:
                print(f"Error auto-analyzing {image}: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'files[]' not in request.files:
        flash('No files selected')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('index'))
    
    file_count = 0
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)
            file_count += 1
            
            # Automatically analyze the image after upload
            try:
                clothing_analyzer.analyze_image(file_path)
            except Exception as e:
                print(f"Error analyzing uploaded image {filename}: {str(e)}")
    
    if file_count > 0:
        flash(f'Successfully uploaded and analyzed {file_count} image(s)')
    else:
        flash('No valid images uploaded')
    
    return redirect(url_for('index'))

@app.route('/analyze/<filename>')
def analyze_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash(f"Image {filename} not found")
        return redirect(url_for('index'))
    
    try:
        # Force reanalysis of the image
        analysis_result = clothing_analyzer.analyze_image(file_path)
        flash(f"Successfully analyzed {filename}")
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error analyzing image: {str(e)}")
        return redirect(url_for('index'))

@app.route('/analyze_all')
def analyze_all_images():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    images = [img for img in images if img.split('.')[-1].lower() in ALLOWED_EXTENSIONS]
    
    count = clothing_analyzer.preanalyze_images(app.config['UPLOAD_FOLDER'], ALLOWED_EXTENSIONS)
    
    flash(f"Analyzed {count} images")
    return redirect(url_for('index'))

@app.route('/clear_analysis')
def clear_analysis():
    try:
        # Delete all JSON files in the analysis cache directory
        for file in os.listdir(app.config['ANALYSIS_CACHE']):
            if file.endswith('.json'):
                os.remove(os.path.join(app.config['ANALYSIS_CACHE'], file))
        
        flash("Analysis results cleared")
    except Exception as e:
        flash(f"Error clearing analysis: {str(e)}")
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Pre-analyze existing images on startup
    clothing_analyzer.preanalyze_images(UPLOAD_FOLDER, ALLOWED_EXTENSIONS)
    app.run(debug=True)
