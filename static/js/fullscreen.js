/**
 * Fullscreen functionality for pose visualization
 */

document.addEventListener('DOMContentLoaded', function() {
  const videoContainer = document.getElementById('video-container');
  const fullscreenBtn = document.getElementById('fullscreen-btn');
  const exitFullscreenBtn = document.getElementById('exit-fullscreen-btn');
  const fullscreenOverlay = document.getElementById('fullscreen-overlay');
  
  if (!videoContainer || !fullscreenBtn || !exitFullscreenBtn) {
    console.error('Required fullscreen elements not found');
    return;
  }
  
  // Toggle fullscreen mode when clicking the fullscreen button
  fullscreenBtn.addEventListener('click', function() {
    enterFullscreen();
  });
  
  // Exit fullscreen when clicking the exit button
  exitFullscreenBtn.addEventListener('click', function() {
    exitFullscreen();
  });
  
  // Also exit fullscreen when pressing Escape
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && videoContainer.classList.contains('fullscreen')) {
      exitFullscreen();
    }
  });
  
  // Enter fullscreen mode
  function enterFullscreen() {
    videoContainer.classList.add('fullscreen');
    document.body.style.overflow = 'hidden'; // Prevent scrolling
    
    // Show temporary instruction overlay
    fullscreenOverlay.style.display = 'block';
    setTimeout(() => {
      fullscreenOverlay.style.display = 'none';
    }, 3000);
    
    // Resize canvas in fullscreen mode
    const canvasElement = videoContainer.querySelector('.output-canvas');
    if (canvasElement) {
      setTimeout(() => {
        canvasElement.width = window.innerWidth;
        canvasElement.height = window.innerHeight;
      }, 100);
    }
    
    // Try to request actual browser fullscreen for better experience
    if (videoContainer.requestFullscreen) {
      videoContainer.requestFullscreen().catch(err => {
        console.warn('Error attempting to enable fullscreen mode:', err);
      });
    } else if (videoContainer.webkitRequestFullscreen) { /* Safari */
      videoContainer.webkitRequestFullscreen();
    } else if (videoContainer.msRequestFullscreen) { /* IE11 */
      videoContainer.msRequestFullscreen();
    }
  }
  
  // Exit fullscreen mode
  function exitFullscreen() {
    videoContainer.classList.remove('fullscreen');
    document.body.style.overflow = ''; // Restore scrolling
    
    // Exit actual browser fullscreen if active
    if (document.fullscreenElement) {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) { /* Safari */
        document.webkitExitFullscreen();
      } else if (document.msExitFullscreen) { /* IE11 */
        document.msExitFullscreen();
      }
    }
    
    // Resize canvas back to original size
    const canvasElement = videoContainer.querySelector('.output-canvas');
    if (canvasElement) {
      setTimeout(() => {
        canvasElement.width = canvasElement.clientWidth;
        canvasElement.height = canvasElement.clientHeight;
      }, 100);
    }
  }
  
  // Handle browser fullscreen change event
  document.addEventListener('fullscreenchange', handleFullscreenChange);
  document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
  document.addEventListener('mozfullscreenchange', handleFullscreenChange);
  document.addEventListener('MSFullscreenChange', handleFullscreenChange);
  
  function handleFullscreenChange() {
    if (!document.fullscreenElement && !document.webkitFullscreenElement &&
        !document.mozFullScreenElement && !document.msFullscreenElement) {
      // Browser's fullscreen mode was exited, ensure our UI reflects this
      videoContainer.classList.remove('fullscreen');
      document.body.style.overflow = '';
    }
  }
});
