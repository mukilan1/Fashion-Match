/**
 * Enhanced chatbot formatter with improved alignment handling
 */

// Process chatbot response text with better alignment handling
function formatChatbotResponse(text) {
    if (!text) return '';
    
    // Normalize line endings and clean up any excessive whitespace
    text = text.replace(/\r\n/g, '\n')
               .replace(/\n{3,}/g, '\n\n')
               .replace(/ {2,}/g, ' ')
               .trim();
    
    // Remove any markdown syntax indicators that could cause issues
    text = text.replace(/```/g, '');
    
    // Format bold text consistently
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Fix common headline patterns
    text = text.replace(/^(#+)\s+(.+)$/gm, '<div class="response-headline">$2</div>');
    text = text.replace(/^([A-Z][A-Za-z\s]+:)(\s*)$/gm, '<div class="response-headline">$1</div>');
    text = text.replace(/^([A-Z][A-Za-z\s]+:)(\s+)(?!<)/gm, '<div class="response-headline">$1</div>$2');
    
    // Process bullet points consistently even with different markers
    const processLists = (input) => {
        let sections = input.split(/\n\n|\r\n\r\n/);
        
        for (let i = 0; i < sections.length; i++) {
            // Check for bullet point pattern (• or - or *)
            if (/^[•\-*]\s+/m.test(sections[i])) {
                // Split by list markers
                let listItems = sections[i].split(/\n[•\-*]\s+/);
                let beforeList = listItems[0];
                
                // If first item isn't preceded by a bullet, it's not part of list
                if (!/^[•\-*]\s+/.test(beforeList)) {
                    listItems = listItems.slice(1);
                } else {
                    // Otherwise remove the bullet from first item
                    beforeList = beforeList.replace(/^[•\-*]\s+/, '');
                }
                
                // Build proper HTML list
                let listHTML = '';
                if (beforeList.trim()) {
                    listHTML += `${beforeList.trim()}\n`;
                }
                
                if (listItems.length > 0) {
                    const itemsHTML = listItems
                        .map(item => item.trim())
                        .filter(item => item.length > 0)
                        .map(item => `<li>${item}</li>`)
                        .join('');
                    
                    if (itemsHTML) {
                        listHTML += `<ul>${itemsHTML}</ul>`;
                    }
                }
                
                sections[i] = listHTML;
            }
        }
        
        return sections.join('\n\n');
    };
    
    // Apply list processing
    text = processLists(text);
    
    // Format key-value pairs more consistently with better alignment
    const keyValueProcessors = [
        // "Key: Value" format - fashion-specific terms
        {
            pattern: /(Top|Bottom|Footwear|Accessories|Shoes):\s*([^\n]+)/g,
            replacement: '<div class="key-value outfit-item"><div class="key">$1</div><div class="value">$2</div></div>'
        },
        // "Color: Value" format
        {
            pattern: /(Primary|Secondary|Accent|Base|Main|Color)(\s+[Cc]olor)?:\s*([^\n]+)/g,
            replacement: (match, key, _, value) => {
                return `<div class="key-value color-item"><div class="key">${key}${_ || ''}</div><div class="value">${value}</div></div>`;
            }
        },
        // Style related key-values
        {
            pattern: /(Style|Occasion|Season|Material|Fabric):\s*([^\n]+)/g,
            replacement: '<div class="key-value style-item"><div class="key">$1</div><div class="value">$2</div></div>'
        }
    ];
    
    // Apply each key-value pattern matcher
    keyValueProcessors.forEach(processor => {
        text = text.replace(processor.pattern, processor.replacement);
    });
    
    // Ensure good paragraph structure but don't add unnecessary tags
    if (!text.includes('<p>')) {
        let parts = text.split('\n\n');
        parts = parts.map(part => {
            // Don't wrap these elements in <p> tags
            if (part.includes('<div class="key-value') ||
                part.includes('<ul') || 
                part.includes('<ol') ||
                part.includes('<div class="response-headline">')) {
                return part;
            }
            return `<p>${part}</p>`;
        });
        text = parts.join('');
    }
    
    return text;
}

// Enhanced message display with alignment fixes
function addFormattedMessage(message, isUser = false) {
    const chatBody = document.getElementById('chat-body');
    const messageDiv = document.createElement('div');
    
    messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
    
    if (!isUser) {
        // Apply enhanced formatting for bot messages
        messageDiv.innerHTML = formatChatbotResponse(message);
        
        // Post-process to fix any remaining alignment issues
        setTimeout(() => {
            // Ensure lists are properly aligned
            const lists = messageDiv.querySelectorAll('ul, ol');
            lists.forEach(list => {
                list.style.paddingLeft = '22px';
                list.style.marginTop = '8px';
                list.style.marginBottom = '8px';
            });
            
            // Ensure key-value pairs are aligned
            const keyValuePairs = messageDiv.querySelectorAll('.key-value');
            keyValuePairs.forEach(pair => {
                const key = pair.querySelector('.key');
                const value = pair.querySelector('.value');
                if (key && value) {
                    key.style.minWidth = '80px';
                    key.style.marginRight = '10px';
                }
            });
        }, 0);
    } else {
        // User messages are simpler
        messageDiv.textContent = message;
    }
    
    chatBody.appendChild(messageDiv);
    smoothScrollToBottom(chatBody);
    
    return messageDiv;
}

// Apply entry animation based on message type
function applyEntryAnimation(element, isUser) {
    // Clear any existing animations
    element.style.animation = '';
    
    // Force browser reflow to ensure animation triggers
    void element.offsetWidth;
    
    // Apply appropriate animation
    if (isUser) {
        element.style.animation = 'fadeInUp 0.3s ease-out forwards, slide-in-right 0.3s ease-out forwards';
    } else {
        element.style.animation = 'fadeInUp 0.3s ease-out forwards, slide-in-left 0.3s ease-out forwards';
    }
}

// Smooth scroll to bottom of chat container
function smoothScrollToBottom(element) {
    const start = element.scrollTop;
    const end = element.scrollHeight - element.clientHeight;
    const change = end - start;
    const duration = 300;
    let startTime = null;
    
    function animateScroll(timestamp) {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1);
        element.scrollTop = start + change * ease(progress);
        
        if (progress < 1) {
            window.requestAnimationFrame(animateScroll);
        }
    }
    
    function ease(t) {
        // Easing function for smooth animation
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }
    
    window.requestAnimationFrame(animateScroll);
}

// Enhanced typing indicator with improved animation
function showTypingIndicator() {
    const chatBody = document.getElementById('chat-body');
    const existingIndicator = document.getElementById('typing-indicator');
    
    // Remove existing indicator if present
    if (existingIndicator) {
        existingIndicator.remove();
    }
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing';
    typingDiv.id = 'typing-indicator';
    
    // Create animated dots with staggered delays
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'dot';
        dot.style.animationDelay = `${i * 0.15}s`;
        typingDiv.appendChild(dot);
    }
    
    chatBody.appendChild(typingDiv);
    smoothScrollToBottom(chatBody);
    
    // Apply animation to the typing indicator
    applyEntryAnimation(typingDiv, false);
}

// Initialize the enhanced chatbot UI
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing enhanced chatbot UI with improved animations');
    
    // Replace the default message handler with our enhanced version
    window.addFormattedMessage = addFormattedMessage;
    window.showTypingIndicator = showTypingIndicator;
    
    // Set animation delays for suggestion chips to create staggered appearance
    document.querySelectorAll('.chip').forEach((chip, index) => {
        chip.style.setProperty('--index', index + 1);
    });
});
