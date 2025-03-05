# Multiple Image Upload Flask Application

A simple Flask web application that allows users to upload and view multiple images.

## Features

- Upload multiple images at once
- Display uploaded images in a responsive gallery
- Flash messages for user feedback
- File type validation

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd F_M
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Execute the following command to start the application:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/`

## Project Structure

```
F_M/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css       # CSS styling
│   └── uploads/            # Uploaded images are stored here
└── templates/
    └── index.html          # Main HTML template
```
