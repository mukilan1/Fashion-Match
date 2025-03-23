// Complete pose detection with MediaPipe integration

let videoElement, canvasElement, canvasCtx, tryonCanvasElement, tryonCanvasCtx;
let camera = null;
let poseDetectionActive = false;

// FPS calculation variables
let frameCount = 0;
let lastTime = 0;
let fps = 0;

// Essential MediaPipe pose connections for visualizing the skeleton
const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5],
  [5, 6], [6, 8], [9, 10], [11, 12], [11, 13],
  [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
  [18, 20], [11, 23], [12, 24], [23, 24], [23, 25],
  [24, 26], [25, 27], [26, 28], [27, 29], [28, 30],
  [29, 31], [30, 32], [27, 31], [28, 32]
];

// Initialize the pose detection
function initializePose() {
  console.log('Initializing pose detection...');

  // DOM elements
  videoElement = document.querySelector('.input-video');
  canvasElement = document.querySelector('.output-canvas');
  canvasCtx = canvasElement.getContext('2d');
  tryonCanvasElement = document.querySelector('.tryon-canvas');
  tryonCanvasCtx = tryonCanvasElement.getContext('2d');
  
  // Set initial canvas size
  canvasElement.width = 640;
  canvasElement.height = 480;
  tryonCanvasElement.width = 640;
  tryonCanvasElement.height = 480;

  // Initialize MediaPipe Pose
  const pose = new Pose({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
    }
  });

  // Configure pose options
  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  // Set up pose detection callback
  pose.onResults(onResults);

  // Create camera object
  camera = new Camera(videoElement, {
    onFrame: async () => {
      if (poseDetectionActive) {
        await pose.send({image: videoElement});
      }
    },
    width: 640,
    height: 480
  });

  return pose;
}

// Add this function to apply the color legend to MediaPipe visualization
function drawConnectors(landmarks, ctx, width, height) {
  if (!landmarks) return;
  
  // Define color scheme based on the color legend
  const colorScheme = {
    rightArm: '#3498db',   // Blue
    leftArm: '#2ecc71',    // Green
    rightLeg: '#e74c3c',   // Red
    leftLeg: '#9b59b6',    // Purple
    torso: '#1abc9c',      // Teal
    face: '#f1c40f'        // Yellow
  };
  
  // Set line styles
  ctx.lineWidth = 5;
  
  // Define connections with their body part regions
  const connections = [
    // Face connections
    { start: 0, end: 1, color: colorScheme.face },
    { start: 1, end: 2, color: colorScheme.face },
    { start: 2, end: 3, color: colorScheme.face },
    { start: 3, end: 7, color: colorScheme.face },
    { start: 0, end: 4, color: colorScheme.face },
    { start: 4, end: 5, color: colorScheme.face },
    { start: 5, end: 6, color: colorScheme.face },
    { start: 6, end: 8, color: colorScheme.face },
    
    // Torso connections
    { start: 11, end: 12, color: colorScheme.torso },
    { start: 12, end: 24, color: colorScheme.torso },
    { start: 24, end: 23, color: colorScheme.torso },
    { start: 23, end: 11, color: colorScheme.torso },
    
    // Left arm connections
    { start: 11, end: 13, color: colorScheme.leftArm },
    { start: 13, end: 15, color: colorScheme.leftArm },
    { start: 15, end: 17, color: colorScheme.leftArm },
    { start: 17, end: 19, color: colorScheme.leftArm },
    { start: 19, end: 21, color: colorScheme.leftArm },
    { start: 15, end: 19, color: colorScheme.leftArm },
    
    // Right arm connections
    { start: 12, end: 14, color: colorScheme.rightArm },
    { start: 14, end: 16, color: colorScheme.rightArm },
    { start: 16, end: 18, color: colorScheme.rightArm },
    { start: 18, end: 20, color: colorScheme.rightArm },
    { start: 20, end: 22, color: colorScheme.rightArm },
    { start: 16, end: 20, color: colorScheme.rightArm },
    
    // Left leg connections
    { start: 23, end: 25, color: colorScheme.leftLeg },
    { start: 25, end: 27, color: colorScheme.leftLeg },
    { start: 27, end: 29, color: colorScheme.leftLeg },
    { start: 29, end: 31, color: colorScheme.leftLeg },
    
    // Right leg connections
    { start: 24, end: 26, color: colorScheme.rightLeg },
    { start: 26, end: 28, color: colorScheme.rightLeg },
    { start: 28, end: 30, color: colorScheme.rightLeg },
    { start: 30, end: 32, color: colorScheme.rightLeg }
  ];

  // Draw the connections with their respective colors
  connections.forEach(connection => {
    const start = landmarks[connection.start];
    const end = landmarks[connection.end];
    
    // Only draw if both points are visible
    if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
      ctx.beginPath();
      ctx.moveTo(start.x * width, start.y * height);
      ctx.lineTo(end.x * width, end.y * height);
      ctx.strokeStyle = connection.color;
      ctx.stroke();
    }
  });
  
  // Draw the landmark points
  landmarks.forEach((landmark, index) => {
    if (landmark.visibility > 0.5) {
      let color;
      
      // Determine color based on landmark index
      if (index >= 0 && index <= 10) {
        color = colorScheme.face;
      } else if (index >= 11 && index <= 22) {
        color = (index % 2 === 0) ? colorScheme.rightArm : colorScheme.leftArm;
      } else if (index >= 23 && index <= 32) {
        color = (index % 2 === 0) ? colorScheme.rightLeg : colorScheme.leftLeg;
      } else {
        color = '#ffffff';  // Default white
      }
      
      // Draw circle
      ctx.beginPath();
      ctx.arc(landmark.x * width, landmark.y * height, 6, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    }
  });
}

// Process pose detection results - THIS IS THE KEY FUNCTION FOR BOTH SCREENS
function onResults(results) {
  // Update FPS calculation
  frameCount++;
  const now = performance.now();
  const elapsed = now - lastTime;
  
  if (elapsed > 1000) {
    fps = (frameCount * 1000) / elapsed;
    frameCount = 0;
    lastTime = now;
    
    // Update UI with FPS if PoseUI exists
    if (window.PoseUI) {
      window.PoseUI.updateFPS(fps);
    }
  }
  
  // Ensure canvas is properly sized based on video
  if (videoElement.videoWidth && videoElement.videoHeight) {
    if (canvasElement.width !== videoElement.videoWidth) {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
    }
    
    if (tryonCanvasElement.width !== videoElement.videoWidth) {
      tryonCanvasElement.width = videoElement.videoWidth;
      tryonCanvasElement.height = videoElement.videoHeight;
    }
  }

  // Clear the canvas and draw the video frame
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  // Draw pose landmarks on left screen if available
  if (results.poseLandmarks) {
    // Draw connectors (skeleton lines)
    drawConnectors(results.poseLandmarks, canvasCtx, canvasElement.width, canvasElement.height);
    
    // Draw the landmark points
    drawLandmarks(canvasCtx, results.poseLandmarks, {
      color: '#ffffff',
      lineWidth: 2,
      radius: (landmark) => {
        return landmark.visibility > 0.5 ? 3 : 1;
      }
    });
  }
  
  canvasCtx.restore();
  
  // Update the virtual try-on screen with pose data
  if (window.updateVirtualTryOn && tryonCanvasCtx) {
    window.updateVirtualTryOn(tryonCanvasCtx, results);
  }
  
  // Also store the results for animation loop
  window.lastPoseResults = results;
}

// Start camera function - This is called when Start Camera button is clicked
function startCamera() {
  if (!camera) {
    console.error('Camera not initialized');
    return;
  }
  
  console.log('Starting camera...');
  
  // Update UI
  if (window.PoseUI) {
    window.PoseUI.setStatus('Starting camera...', 'info');
  }
  
  // Start the camera stream
  camera.start()
    .then(() => {
      console.log('Camera started successfully');
      poseDetectionActive = true;
      
      // Update UI
      if (window.PoseUI) {
        window.PoseUI.hideLoading();
        window.PoseUI.setStatus('Camera active. Move around to see pose detection.', 'success');
      }
      
      // Update button states
      const startButton = document.getElementById('start-button');
      const stopButton = document.getElementById('stop-button');
      if (startButton) startButton.disabled = true;
      if (stopButton) stopButton.disabled = false;
      
      // Initialize the try-on screen
      initializeTryOn();
    })
    .catch(error => {
      console.error('Error starting camera:', error);
      
      // Update UI with error message
      if (window.PoseUI) {
        window.PoseUI.setStatus('Camera error: ' + error.message, 'error');
      }
      
      // Show troubleshooting tips
      document.getElementById('troubleshooting').style.display = 'block';
    });
}

// Stop camera function
function stopCamera() {
  if (!camera) {
    console.error('Camera not initialized');
    return;
  }
  
  console.log('Stopping camera...');
  
  try {
    poseDetectionActive = false;
    camera.stop();
    
    // Update UI
    if (window.PoseUI) {
      window.PoseUI.setStatus('Camera stopped. Click Start Camera to begin again.', 'info');
    }
    
    // Update button states
    const startButton = document.getElementById('start-button');
    const stopButton = document.getElementById('stop-button');
    if (startButton) startButton.disabled = false;
    if (stopButton) stopButton.disabled = true;
    
    console.log('Camera stopped successfully');
  } catch (error) {
    console.error('Error stopping camera:', error);
  }
}

// Initialize try-on screen
function initializeTryOn() {
  const tryonCanvas = document.querySelector('.tryon-canvas');
  if (!tryonCanvas) {
    console.error('Try-on canvas not found');
    return;
  }
  
  const ctx = tryonCanvas.getContext('2d');
  
  // Mirror the video feed to the try-on canvas initially
  const videoElement = document.querySelector('.input-video');
  if (videoElement && videoElement.videoWidth && videoElement.videoHeight) {
    tryonCanvas.width = videoElement.videoWidth;
    tryonCanvas.height = videoElement.videoHeight;
    
    // Draw black background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, tryonCanvas.width, tryonCanvas.height);
    
    console.log('Try-on canvas initialized with size:', tryonCanvas.width, 'x', tryonCanvas.height);
  } else {
    // Use default size if video dimensions aren't available
    tryonCanvas.width = 640;
    tryonCanvas.height = 480;
    
    // Draw black background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, tryonCanvas.width, tryonCanvas.height);
    
    console.log('Try-on canvas initialized with default size');
  }
  
  // Mark as initialized
  window.tryonInitialized = true;
  
  // Start the animation loop for the try-on canvas
  requestAnimationFrame(tryonAnimationLoop);
}

// Animation loop for try-on canvas - keeps updating even without new pose data
function tryonAnimationLoop() {
  if (!window.tryonInitialized) return;
  
  const tryonCanvas = document.querySelector('.tryon-canvas');
  if (!tryonCanvas) return;
  
  const ctx = tryonCanvas.getContext('2d');
  
  // If we have recent pose data, use it
  if (window.lastPoseResults) {
    window.updateVirtualTryOn(ctx, window.lastPoseResults);
  } else {
    // Otherwise just draw the video
    const videoElement = document.querySelector('.input-video');
    if (videoElement && videoElement.readyState >= 2) {
      // Draw black background
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, tryonCanvas.width, tryonCanvas.height);
      
      // Draw the video with mirroring
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-tryonCanvas.width, 0);
      ctx.drawImage(videoElement, 0, 0, tryonCanvas.width, tryonCanvas.height);
      ctx.restore();
    }
  }
  
  // Continue animation loop
  requestAnimationFrame(tryonAnimationLoop);
}

// Initial setup when document is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Initialize pose detection system
  const pose = initializePose();
  
  // Add event listeners to buttons
  const startButton = document.getElementById('start-button');
  const stopButton = document.getElementById('stop-button');
  
  if (startButton) {
    startButton.addEventListener('click', startCamera);
  }
  
  if (stopButton) {
    stopButton.addEventListener('click', stopCamera);
  }
  
  console.log('Pose detection system initialized and ready');
});

// Make functions available globally for external script access
window.onResults = onResults;
window.startCamera = startCamera;
window.stopCamera = stopCamera;
window.initializeTryOn = initializeTryOn;
