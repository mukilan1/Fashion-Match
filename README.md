# Fashion Matching Application with Pose Rigging

This application combines fashion item matching with MediaPipe pose rigging visualization in a single integrated web application.

## Features

### Fashion Matching
- Upload clothing images
- Automatic classification of garments
- Matches tops and bottoms based on style compatibility
- Color analysis and pattern recognition
- Match history tracking

### Pose Rigging Visualization
- Real-time pose detection using webcam
- Enhanced multi-colored skeletal visualization
- Thicker lines for better visibility
- Highlighted joint points
- Color-coded body segments

## Setup

1. Make sure you have installed all required packages:
   ```
   pip install flask opencv-python mediapipe numpy pillow scikit-learn sentence-transformers transformers
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the navigation menu to switch between:
   - Home/Upload page
   - Clothing Match page
   - Pose Rigging Visualization page

## Pose Rigging Usage

1. Navigate to the Pose Rigging page
2. Click the "Start Camera" button
3. Stand in front of the webcam so your body is visible
4. The visualization will show your pose with enhanced colored rigging
5. Click "Stop Camera" when finished

## Color Code for Pose Rigging

- Blue: Right arm
- Green: Left arm
- Red: Right leg
- Cyan: Left leg
- Magenta: Torso
- Yellow: Face

## Project Structure

- `app.py`: Main Flask application with all routes
- `templates/`: HTML templates for all pages
- `static/`: CSS, JavaScript, and image files
- `uploads/`: Directory for uploaded clothing images
