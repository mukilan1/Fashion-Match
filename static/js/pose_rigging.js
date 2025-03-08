const videoElement = document.getElementsByClassName('input-video')[0];
const canvasElement = document.getElementsByClassName('output-canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const statusElement = document.createElement('div'); // Create status element for debugging

// Add status element to the page
document.querySelector('.controls').appendChild(statusElement);
statusElement.style.marginTop = '10px';
statusElement.style.padding = '5px';
statusElement.style.backgroundColor = '#f8f9fa';
statusElement.style.borderRadius = '4px';
statusElement.style.fontSize = '14px';

// Performance tracking
let frameCount = 0;
let lastFrameTime = 0;
let fps = 0;
let camera = null;
let poseRunning = false;

// Updated color palette with modern, visually appealing colors (in RGB format)
const colors = {
  'right_arm': 'rgb(52, 152, 219)',     // Blue
  'left_arm': 'rgb(46, 204, 113)',      // Green
  'right_leg': 'rgb(231, 76, 60)',      // Red
  'left_leg': 'rgb(155, 89, 182)',      // Purple
  'torso': 'rgb(26, 188, 156)',         // Teal
  'face': 'rgb(241, 196, 15)'           // Yellow
};

// Define body part connections for custom drawing
const bodyParts = {
  'right_arm': [
    ['RIGHT_SHOULDER', 'RIGHT_ELBOW'],
    ['RIGHT_ELBOW', 'RIGHT_WRIST']
  ],
  'left_arm': [
    ['LEFT_SHOULDER', 'LEFT_ELBOW'],
    ['LEFT_ELBOW', 'LEFT_WRIST']
  ],
  'right_leg': [
    ['RIGHT_HIP', 'RIGHT_KNEE'],
    ['RIGHT_KNEE', 'RIGHT_ANKLE']
  ],
  'left_leg': [
    ['LEFT_HIP', 'LEFT_KNEE'],
    ['LEFT_KNEE', 'LEFT_ANKLE']
  ],
  'torso': [
    ['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
    ['RIGHT_SHOULDER', 'RIGHT_HIP'],
    ['RIGHT_HIP', 'LEFT_HIP'],
    ['LEFT_HIP', 'LEFT_SHOULDER']
  ],
  'face': [
    ['NOSE', 'LEFT_EYE_INNER'],
    ['LEFT_EYE_INNER', 'LEFT_EYE'],
    ['LEFT_EYE', 'LEFT_EYE_OUTER'],
    ['NOSE', 'RIGHT_EYE_INNER'],
    ['RIGHT_EYE_INNER', 'RIGHT_EYE'],
    ['RIGHT_EYE', 'RIGHT_EYE_OUTER'],
    ['MOUTH_LEFT', 'MOUTH_RIGHT']
  ]
};

// Key points to draw larger circles
const keyPoints = [
  'LEFT_SHOULDER', 'RIGHT_SHOULDER',
  'LEFT_ELBOW', 'RIGHT_ELBOW',
  'LEFT_WRIST', 'RIGHT_WRIST',
  'LEFT_HIP', 'RIGHT_HIP',
  'LEFT_KNEE', 'RIGHT_KNEE',
  'LEFT_ANKLE', 'RIGHT_ANKLE',
  'NOSE'
];

function onResults(results) {
  // Increment frame counter for debugging
  frameCount++;
  statusElement.textContent = `Processed frames: ${frameCount}`;
  
  // Clear the canvas
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  // Draw the video frame
  canvasCtx.drawImage(
    results.image, 0, 0, canvasElement.width, canvasElement.height);
  
  if (results.poseLandmarks) {
    const landmarks = results.poseLandmarks;
    
    // Draw custom colored body parts with thicker lines
    for (const [part, connections] of Object.entries(bodyParts)) {
      const color = colors[part];
      
      for (const connection of connections) {
        const start = landmarks[window.POSE_LANDMARKS[connection[0]]];
        const end = landmarks[window.POSE_LANDMARKS[connection[1]]];
        
        if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
          // Draw thicker lines
          canvasCtx.beginPath();
          canvasCtx.moveTo(start.x * canvasElement.width, start.y * canvasElement.height);
          canvasCtx.lineTo(end.x * canvasElement.width, end.y * canvasElement.height);
          canvasCtx.lineWidth = 6;
          canvasCtx.strokeStyle = color;
          canvasCtx.stroke();
        }
      }
    }
    
    // Draw larger circles at key joints
    for (const point of keyPoints) {
      const landmark = landmarks[window.POSE_LANDMARKS[point]];
      
      if (landmark && landmark.visibility > 0.5) {
        const x = landmark.x * canvasElement.width;
        const y = landmark.y * canvasElement.height;
        
        // Draw a larger filled circle at each key point
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 8, 0, 2 * Math.PI);
        canvasCtx.fillStyle = 'white';
        canvasCtx.fill();
        
        // Draw black outline
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'black';
        canvasCtx.stroke();
      }
    }
  }
  
  canvasCtx.restore();
}

// Initialize the MediaPipe Pose solution
const pose = new Pose({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
  }
});

pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  smoothSegmentation: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

pose.onResults(onResults);

// Start button event listener
startButton.addEventListener('click', () => {
  if (!poseRunning) {
    startCamera();
  }
});

// Stop button event listener
stopButton.addEventListener('click', () => {
  if (poseRunning && camera) {
    stopCamera();
  }
});

// Function to start the camera with proper error handling
function startCamera() {
  try {
    statusElement.textContent = "Starting camera...";
    statusElement.style.color = "blue";
    
    // Reset frame counter
    frameCount = 0;
    
    // Create camera instance if it doesn't exist
    if (!camera) {
      camera = new Camera(videoElement, {
        onFrame: async () => {
          try {
            await pose.send({image: videoElement});
          } catch (error) {
            console.error("Error in pose detection:", error);
            statusElement.textContent = `Error: ${error.message}`;
            statusElement.style.color = "red";
          }
        },
        width: 640,
        height: 480
      });
    }
    
    // Start the camera with promise handling
    camera.start()
      .then(() => {
        console.log("Camera started successfully");
        poseRunning = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        statusElement.textContent = "Camera running - processing frames...";
        statusElement.style.color = "green";
      })
      .catch(error => {
        console.error("Failed to start camera:", error);
        statusElement.textContent = `Camera error: ${error.message}`;
        statusElement.style.color = "red";
      });
  } catch (error) {
    console.error("Error starting camera:", error);
    statusElement.textContent = `Setup error: ${error.message}`;
    statusElement.style.color = "red";
  }
}

// Function to stop the camera
function stopCamera() {
  try {
    camera.stop();
    poseRunning = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    statusElement.textContent = `Camera stopped. Processed ${frameCount} frames`;
    statusElement.style.color = "blue";
  } catch (error) {
    console.error("Error stopping camera:", error);
    statusElement.textContent = `Stop error: ${error.message}`;
    statusElement.style.color = "red";
  }
}

// Add event listener for when the page is hidden/shown to handle camera correctly
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden' && poseRunning) {
    // Page is hidden, pause camera to save resources
    if (camera) {
      try {
        camera.stop();
        statusElement.textContent = "Camera paused (page hidden)";
        statusElement.style.color = "orange";
      } catch (e) {
        console.log("Error pausing camera:", e);
      }
    }
  } else if (document.visibilityState === 'visible' && poseRunning) {
    // Page is visible again, resume camera if it was running
    if (camera) {
      try {
        camera.start()
          .then(() => {
            statusElement.textContent = "Camera resumed - processing frames...";
            statusElement.style.color = "green";
          })
          .catch(e => {
            console.error("Failed to resume camera:", e);
            statusElement.textContent = `Resume error: ${e.message}`;
            statusElement.style.color = "red";
          });
      } catch (e) {
        console.error("Error resuming camera:", e);
      }
    }
  }
});
