/**
 * Background Removal Animation Effects
 * Provides visual feedback during the background removal process
 */

class BackgroundRemovalAnimation {
    constructor() {
        this.animationQueue = [];
        this.isProcessing = false;
    }
    
    /**
     * Add an animation job to the queue
     * @param {Object} element - The DOM element to animate
     * @param {Function} onComplete - Callback when animation is complete
     */
    queueAnimation(element, onComplete) {
        this.animationQueue.push({ element, onComplete });
        
        if (!this.isProcessing) {
            this.processQueue();
        }
    }
    
    /**
     * Process the animation queue
     */
    async processQueue() {
        if (this.animationQueue.length === 0) {
            this.isProcessing = false;
            return;
        }
        
        this.isProcessing = true;
        const job = this.animationQueue.shift();
        
        try {
            await this.runAnimation(job.element);
            if (job.onComplete) job.onComplete();
        } catch (error) {
            console.error('Animation error:', error);
        }
        
        // Process next item in queue
        this.processQueue();
    }
    
    /**
     * Run the background removal animation
     * @param {Object} element - The element to animate
     * @returns {Promise} - Resolves when animation completes
     */
    runAnimation(element) {
        return new Promise((resolve) => {
            // Save the original image
            const originalImg = element.querySelector('img');
            if (!originalImg) {
                resolve();
                return;
            }
            
            // Create animation overlay
            const overlay = document.createElement('div');
            overlay.className = 'bg-removal-animation-overlay';
            overlay.innerHTML = `
                <div class="removal-effect">
                    <div class="magic-wand">
                        <i class="fas fa-magic"></i>
                    </div>
                    <div class="removal-progress">
                        <div class="progress-track">
                            <div class="progress-fill"></div>
                        </div>
                    </div>
                    <div class="removal-text">Removing Background...</div>
                </div>
            `;
            
            // Add overlay
            element.appendChild(overlay);
            
            // Animate the background removal process
            const progressFill = overlay.querySelector('.progress-fill');
            let progress = 0;
            
            const interval = setInterval(() => {
                progress += 2;
                progressFill.style.width = `${Math.min(progress, 100)}%`;
                
                if (progress >= 100) {
                    clearInterval(interval);
                    
                    // Show completion effect and resolve
                    this.showCompletionEffect(element, overlay, resolve);
                }
            }, 50);
        });
    }
    
    /**
     * Show completion animation when background removal finishes
     * @param {Object} element - The container element
     * @param {Object} overlay - The animation overlay
     * @param {Function} resolve - Promise resolve function
     */
    showCompletionEffect(element, overlay, resolve) {
        overlay.querySelector('.removal-text').textContent = 'Background Removed!';
        
        // Add completion particles
        this.addCompletionParticles(element);
        
        // Add "completed" class for CSS animations
        overlay.classList.add('completed');
        
        // Remove overlay after animation
        setTimeout(() => {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.remove();
                resolve();
            }, 500);
        }, 1000);
    }
    
    /**
     * Add particle effect when background removal completes
     * @param {Object} element - The container element
     */
    addCompletionParticles(element) {
        const colors = ['#4a90e2', '#2575fc', '#6a11cb', '#ffffff'];
        const particleCount = 20;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'removal-particle';
            particle.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            particle.style.left = `${50 + (Math.random() - 0.5) * 20}%`;
            particle.style.top = `${50 + (Math.random() - 0.5) * 20}%`;
            
            element.appendChild(particle);
            
            // Animate particle
            setTimeout(() => {
                const angle = Math.random() * Math.PI * 2;
                const distance = 30 + Math.random() * 50;
                
                particle.style.transform = `translate(
                    ${Math.cos(angle) * distance}px, 
                    ${Math.sin(angle) * distance}px
                ) scale(0)`;
                
                // Remove particle after animation
                setTimeout(() => {
                    particle.remove();
                }, 1000);
            }, 10);
        }
    }
}

// Create global instance
window.bgRemovalAnimation = new BackgroundRemovalAnimation();
