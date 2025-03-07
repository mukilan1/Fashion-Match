/**
 * Improved image handling functions
 */

// Handle image loading with fallbacks
function setupImageHandling() {
  // Find all images that need handling
  const images = document.querySelectorAll('.managed-image');
  
  images.forEach(img => {
    // Save original source
    const originalSrc = img.getAttribute('src');
    const placeholderId = img.getAttribute('data-placeholder');
    const placeholder = document.getElementById(placeholderId);
    
    // Show placeholder initially
    if (placeholder) {
      placeholder.style.display = 'flex';
    }
    
    // Handle successful load
    img.onload = function() {
      if (placeholder) {
        placeholder.style.display = 'none';
      }
      img.classList.add('loaded');
    };
    
    // Handle image errors
    img.onerror = function() {
      // Try with cache busting
      const cacheBuster = '?cb=' + new Date().getTime();
      img.src = originalSrc + cacheBuster;
      
      // Set second error handler for final fallback
      img.onerror = function() {
        if (placeholder) {
          placeholder.innerHTML = '<i class="fas fa-exclamation-triangle text-danger fa-2x mb-2"></i><p>Image failed to load</p>';
          placeholder.style.display = 'flex';
        }
        img.style.display = 'none';
        
        // Show error message if available
        const errorMsgId = img.getAttribute('data-error-message');
        if (errorMsgId) {
          const errorMsg = document.getElementById(errorMsgId);
          if (errorMsg) {
            errorMsg.style.display = 'block';
          }
        }
      };
    };
  });
}

// Lazy load images
function setupLazyLoading() {
  if ('IntersectionObserver' in window) {
    const lazyImages = document.querySelectorAll('.lazy-image');
    
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.classList.add('loading');
          observer.unobserve(img);
        }
      });
    });
    
    lazyImages.forEach(img => {
      imageObserver.observe(img);
    });
  } else {
    // Fallback for browsers without IntersectionObserver
    const lazyImages = document.querySelectorAll('.lazy-image');
    lazyImages.forEach(img => {
      img.src = img.dataset.src;
    });
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  setupImageHandling();
  setupLazyLoading();
});
