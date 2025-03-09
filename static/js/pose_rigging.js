/**
 * MediaPipe Pose Detection integration for fashion visualization
 */

document.addEventListener('DOMContentLoaded', function() {
  // DOM elements
  const videoElement = document.querySelector('.input-video');
  const canvasElement = document.querySelector('.output-canvas');
  const canvasCtx = canvasElement.getContext('2d');
  const startButton = document.getElementById('start-button');
  const stopButton = document.getElementById('stop-button');
  
  // Pose detection variables
  let pose;
  let camera;
  let lastFrameTime = 0;
  let poseActive = false;
  
  // Initialize pose model
  function initializePose() {
    pose = new Pose({
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
    
    if (window.PoseUI) {
      window.PoseUI.hideLoading();
    }
  }
  
  // Handle pose detection results
  function onResults(results) {
    // Calculate FPS
    const now = performance.now();
    const deltaTime = now - lastFrameTime;
    const fps = 1000 / deltaTime;
    lastFrameTime = now;
    
    if (window.PoseUI && window.PoseUI.updateFPS) {
      window.PoseUI.updateFPS(fps);
    }
    
    // Draw results on canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw the video feed
    canvasCtx.globalCompositeOperation = 'source-over';
    canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
      
    // Draw pose landmarks with our custom styling
    if (results.poseLandmarks) {
      // Draw the pose landmarks with custom styling
      drawLandmarks(canvasCtx, results.poseLandmarks);
      
      // Draw the connections between landmarks
      drawConnections(canvasCtx, results.poseLandmarks);
    }
    
    canvasCtx.restore();
  }
  
  // Custom landmark drawing function with fashion-oriented styling
  function drawLandmarks(ctx, landmarks) {
    if (!landmarks) return;
    
    for (const landmark of landmarks) {
      const x = landmark.x * canvasElement.width;
      const y = landmark.y * canvasElement.height;
      const z = landmark.z; // Use z for depth-based styling
      
      // Calculate visibility-based opacity
      const opacity = landmark.visibility ? Math.max(0.2, landmark.visibility) : 0.2;
      
      // Draw landmark
      ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
      ctx.beginPath();
      
      // Size based on z-depth (closer = larger)
      const size = Math.max(4, 8 - (z * 20));
      
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fill();
      
      // Add a highlight effect
      ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.5})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  
  // Custom connections drawing with fashion color coding
  function drawConnections(ctx, landmarks) {
    if (!landmarks) return;
    
    // Color coding for body parts
    const colorMap = {
      rightArm: '#3498db', // Blue
      leftArm: '#2ecc71',  // Green
      rightLeg: '#e74c3c', // Red
      leftLeg: '#9b59b6',  // Purple
      torso: '#1abc9c',    // Teal
      face: '#f1c40f'      // Yellow
    };
    
    // Define connections between landmarks with body part classification
    const connections = [
      // Face
      {from: 0, to: 1, part: 'face'},
      {from: 1, to: 2, part: 'face'},
      {from: 2, to: 3, part: 'face'},
      {from: 3, to: 7, part: 'face'},
      {from: 0, to: 4, part: 'face'},
      {from: 4, to: 5, part: 'face'},
      {from: 5, to: 6, part: 'face'},
      {from: 6, to: 8, part: 'face'},
      
      // Torso
      {from: 11, to: 12, part: 'torso'},
      {from: 12, to: 24, part: 'torso'},
      {from: 24, to: 23, part: 'torso'},
      {from: 23, to: 11, part: 'torso'},
      
      // Left arm
      {from: 11, to: 13, part: 'leftArm'},
      {from: 13, to: 15, part: 'leftArm'},
      {from: 15, to: 17, part: 'leftArm'},
      {from: 17, to: 19, part: 'leftArm'},
      {from: 19, to: 15, part: 'leftArm'},
      
      // Right arm
      {from: 12, to: 14, part: 'rightArm'},
      {from: 14, to: 16, part: 'rightArm'},
      {from: 16, to: 18, part: 'rightArm'},
      {from: 18, to: 20, part: 'rightArm'},
      {from: 20, to: 16, part: 'rightArm'},
      
      // Left leg
      {from: 23, to: 25, part: 'leftLeg'},
      {from: 25, to: 27, part: 'leftLeg'},
      {from: 27, to: 29, part: 'leftLeg'},
      {from: 29, to: 31, part: 'leftLeg'},
      {from: 27, to: 31, part: 'leftLeg'},
      
      // Right leg
      {from: 24, to: 26, part: 'rightLeg'},
      {from: 26, to: 28, part: 'rightLeg'},
      {from: 28, to: 30, part: 'rightLeg'},
      {from: 30, to: 32, part: 'rightLeg'},
      {from: 28, to: 32, part: 'rightLeg'}
    ];
    
    // Draw each connection with its assigned color
    for (const connection of connections) {
      const from = landmarks[connection.from];
      const to = landmarks[connection.to];
      
      // Skip if landmarks not visible enough
      if (from.visibility < 0.3 || to.visibility < 0.3) continue;
      
      // Calculate coordinates
      const fromX = from.x * canvasElement.width;
      const fromY = from.y * canvasElement.height;
      const toX = to.x * canvasElement.width;
      const toY = to.y * canvasElement.height;
      
      // Get color for this body part
      const color = colorMap[connection.part];
      
      // Average visibility for this connection
      const visibility = (from.visibility + to.visibility) / 2;
      
      // Draw connection with gradient based on body part
      const gradient = ctx.createLinearGradient(fromX, fromY, toX, toY);
      gradient.addColorStop(0, `${color}99`); // Semi-transparent
      gradient.addColorStop(1, `${color}${Math.round(visibility * 255).toString(16).padStart(2, '0')}`);
      
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.stroke();
      
      // Add highlight effect
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.strokeStyle = `rgba(255, 255, 255, ${visibility * 0.3})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  
  // Start camera
  function startCamera() {
    if (poseActive) return;
    
    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      }
    };
    
    if (window.PoseUI) {
      window.PoseUI.setStatus('Starting camera...', 'warning');
    }
    
    navigator.mediaDevices.getUserMedia(constraints)
      .then((stream) => {
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {
          videoElement.play();
          startPoseDetection();
        };
      })
      .catch((err) => {
        console.error('Error accessing camera: ', err);
        if (window.PoseUI) {
          window.PoseUI.setStatus(`Camera error: ${err.message}`, 'error');
        }
        document.getElementById('troubleshooting').style.display = 'block';
      });
  }
  
  // Start pose detection
  function startPoseDetection() {
    if (!pose) {
      initializePose();
    }
    
    // Set up camera
    camera = new Camera(videoElement, {
      onFrame: async () => {
        await pose.send({image: videoElement});
      },
      width: 1280,
      height: 720
    });
    
    // Start camera
    camera.start()
      .then(() => {
        console.log('Camera started successfully');
        poseActive = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        
        if (window.PoseUI) {
          window.PoseUI.setStatus('Pose detection active', 'success');
        }
      })
      .catch((err) => {
        console.error('Error starting camera: ', err);
        if (window.PoseUI) {
          window.PoseUI.setStatus(`Error: ${err.message}`, 'error');
        }
      });
  }
  
  // Stop camera and pose detection
  function stopCamera() {
    if (!poseActive) return;
    
    if (camera) {
      camera.stop();
    }
    
    if (videoElement.srcObject) {
      videoElement.srcObject.getTracks().forEach(track => track.stop());
      videoElement.srcObject = null;
    }
    
    poseActive = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    
    // Clear the canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (window.PoseUI) {
      window.PoseUI.setStatus('Camera stopped', 'info');
    }
  }
  
  // Event listeners for buttons
  startButton.addEventListener('click', startCamera);
  stopButton.addEventListener('click', stopCamera);
  
  // Canvas setup
  function setupCanvas() {
    // Set canvas to display size
    const displayWidth = canvasElement.clientWidth;
    const displayHeight = canvasElement.clientHeight;
    
    // Check if the canvas is not the same size
    if (canvasElement.width !== displayWidth || 
        canvasElement.height !== displayHeight) {
      canvasElement.width = displayWidth;
      canvasElement.height = displayHeight;
    }
  }
  
  // Handle resize events
  window.addEventListener('resize', setupCanvas);
  
  // Initial setup
  setupCanvas();
  initializePose();
  
  // Expose functions to global scope for debugging
  window.startPoseCamera = startCamera;
  window.stopPoseCamera = stopCamera;
});
