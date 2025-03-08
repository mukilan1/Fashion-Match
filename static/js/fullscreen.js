document.addEventListener('DOMContentLoaded', function() {
  const videoContainer = document.getElementById('video-container');
  const fullscreenBtn = document.getElementById('fullscreen-btn');
  const exitFullscreenBtn = document.getElementById('exit-fullscreen-btn');
  const fullscreenOverlay = document.getElementById('fullscreen-overlay');
  
  // Function to enter fullscreen
  function enterFullscreen() {
    if (videoContainer.requestFullscreen) {
      videoContainer.requestFullscreen();
    } else if (videoContainer.webkitRequestFullscreen) { /* Safari */
      videoContainer.webkitRequestFullscreen();
    } else if (videoContainer.msRequestFullscreen) { /* IE11 */
      videoContainer.msRequestFullscreen();
    }
    
    videoContainer.classList.add('fullscreen');
  }
  
  // Function to exit fullscreen
  function exitFullscreen() {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) { /* Safari */
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) { /* IE11 */
      document.msExitFullscreen();
    }
    
    videoContainer.classList.remove('fullscreen');
  }
  
  // Toggle fullscreen when the button is clicked
  fullscreenBtn.addEventListener('click', function() {
    enterFullscreen();
  });
  
  // Exit fullscreen when the exit button is clicked
  exitFullscreenBtn.addEventListener('click', function() {
    exitFullscreen();
  });
  
  // Also handle the case when user presses ESC to exit fullscreen
  document.addEventListener('fullscreenchange', function() {
    if (!document.fullscreenElement) {
      videoContainer.classList.remove('fullscreen');
    }
  });
  
  // For Safari
  document.addEventListener('webkitfullscreenchange', function() {
    if (!document.webkitFullscreenElement) {
      videoContainer.classList.remove('fullscreen');
    }
  });
  
  // For IE11
  document.addEventListener('MSFullscreenChange', function() {
    if (!document.msFullscreenElement) {
      videoContainer.classList.remove('fullscreen');
    }
  });
});
