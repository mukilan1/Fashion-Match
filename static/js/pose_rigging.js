const videoElement = document.getElementsByClassName('input-video')[0];
const canvasElement = document.getElementsByClassName('output-canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const statusContainer = document.getElementById('status-container');

let camera = null;
let poseRunning = false;
let frameCount = 0;
let lastFrameTime = 0;
let fps = 0;

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

// Define colors for different body parts (in RGB format)
const colors = {
  'right_arm': 'rgb(0, 0, 255)',      // Blue
  'left_arm': 'rgb(0, 255, 0)',       // Green
  'right_leg': 'rgb(255, 0, 0)',      // Red
  'left_leg': 'rgb(0, 255, 255)',     // Cyan
  'torso': 'rgb(255, 0, 255)',        // Magenta
  'face': 'rgb(255, 255, 0)'          // Yellow
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

// Update the status display with better UI
function updateStatus(message, type = 'info') {
  if (!statusContainer) return;
  
  // Remove any existing icon classes
  statusContainer.innerHTML = '';
  
  // Create icon element
  const icon = document.createElement('i');
  icon.className = 'fas me-2 ';
  
  // Add appropriate icon and color based on message type
  switch (type) {
    case 'success':
      icon.className += 'fa-check-circle';
      statusContainer.style.color = '#28a745';
      statusContainer.style.backgroundColor = '#d4edda';
      break;
    case 'error':
      icon.className += 'fa-exclamation-circle';
      statusContainer.style.color = '#721c24';
      statusContainer.style.backgroundColor = '#f8d7da';
      break;
    case 'warning':
      icon.className += 'fa-exclamation-triangle';
      statusContainer.style.color = '#856404';
      statusContainer.style.backgroundColor = '#fff3cd';
      break;
    case 'info':
    default:
      icon.className += 'fa-info-circle';
      statusContainer.style.color = '#0c5460';
      statusContainer.style.backgroundColor = '#d1ecf1';
  }
  
  statusContainer.appendChild(icon);
  
  // Add message text
  const textSpan = document.createElement('span');
  textSpan.textContent = message;
  statusContainer.appendChild(textSpan);
  
  // Add FPS counter if running
  if (poseRunning && type === 'success' && fps > 0) {
    const fpsSpan = document.createElement('span');
    fpsSpan.className = 'badge bg-secondary ms-2';
    fpsSpan.textContent = `${fps.toFixed(1)} FPS`;
    statusContainer.appendChild(fpsSpan);
  }
}

function onResults(results) {
  // Calculate FPS
  const now = performance.now();
  if (lastFrameTime > 0) {
    const delta = (now - lastFrameTime) / 1000;
    fps = 0.9 * fps + 0.1 * (1 / delta); // Smooth the FPS calculation
  }
  lastFrameTime = now;
  
  // Increment frame counter for debugging
  frameCount++;
  if (frameCount % 30 === 0) {  // Update status every 30 frames for better performance
    updateStatus(`Processing pose detection in real-time`, 'success');
  }
  
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
    updateStatus("Starting camera and initializing pose detection...", 'info');
    
    // Reset counters
    frameCount = 0;
    fps = 0;
    lastFrameTime = 0;
    
    // Create camera instance if it doesn't exist
    if (!camera) {
      camera = new Camera(videoElement, {
        onFrame: async () => {
          try {
            await pose.send({image: videoElement});
          } catch (error) {
            console.error("Error in pose detection:", error);
            updateStatus(`Error: ${error.message}`, 'error');
          }
        },
        width: 640,
        height: 480
      });
    }
    
    // Add loading animation
    startButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';
    startButton.disabled = true;
    
    // Start the camera with promise handling
    camera.start()
      .then(() => {
        console.log("Camera started successfully");
        poseRunning = true;
        startButton.innerHTML = '<i class="fas fa-play me-2"></i> Start Camera';
        startButton.disabled = true;
        stopButton.disabled = false;
        updateStatus("Camera running - Move around to see the pose tracking", 'success');
      })
      .catch(error => {
        console.error("Failed to start camera:", error);
        startButton.innerHTML = '<i class="fas fa-play me-2"></i> Start Camera';
        startButton.disabled = false;
        updateStatus(`Camera error: ${error.message}`, 'error');
        
        // Show troubleshooting tips
        document.getElementById('troubleshooting').style.display = 'block';
      });
  } catch (error) {
    console.error("Error starting camera:", error);
    startButton.innerHTML = '<i class="fas fa-play me-2"></i> Start Camera';
    startButton.disabled = false;
    updateStatus(`Setup error: ${error.message}`, 'error');
  }
}

// Function to stop the camera
function stopCamera() {
  try {
    camera.stop();
    poseRunning = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    updateStatus(`Camera stopped. Processed ${frameCount} frames`, 'info');
  } catch (error) {
    console.error("Error stopping camera:", error);
    updateStatus(`Stop error: ${error.message}`, 'error');
  }
}

// Add event listener for when the page is hidden/shown to handle camera correctly
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden' && poseRunning) {
    // Page is hidden, pause camera to save resources
    if (camera) {
      try {
        camera.stop();
        updateStatus("Camera paused (page hidden)", 'warning');
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
            updateStatus("Camera resumed - processing frames...", 'success');
          })
          .catch(e => {
            console.error("Failed to resume camera:", e);
            updateStatus(`Resume error: ${e.message}`, 'error');
          });
      } catch (e) {
        console.error("Error resuming camera:", e);
      }
    }
  }
});

// Initialize the status display
document.addEventListener('DOMContentLoaded', function() {
  updateStatus('Ready to start pose detection', 'info');
});
