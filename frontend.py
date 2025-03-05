from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get all uploaded images to display
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    images = [img for img in images if img.split('.')[-1].lower() in ALLOWED_EXTENSIONS]
    return render_template('index.html', images=images)

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'files[]' not in request.files:
        flash('No files selected')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(request.url)
    
    file_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_count += 1
    
    if file_count > 0:
        flash(f'Successfully uploaded {file_count} image(s)')
    else:
        flash('No valid images uploaded')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
