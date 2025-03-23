"""
Streamlined fashion chatbot service using Ollama models exclusively
"""
import os
import subprocess

# Configure path to fashion knowledge
FASHION_KNOWLEDGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fashion_knowledge.txt")

# Load fashion knowledge as context
try:
    with open(FASHION_KNOWLEDGE_PATH, 'r') as f:
        FASHION_KNOWLEDGE = f.read()
    print(f"✓ Loaded fashion knowledge ({len(FASHION_KNOWLEDGE.split())} words)")
except Exception as e:
    print(f"! Error loading fashion knowledge: {e}")
    FASHION_KNOWLEDGE = "Fashion combines style, color, and personal expression."

# Ollama model names with preference order
OLLAMA_MODELS = ["deepseek-r1:1.5b", "gemma:2b", "llama3.2:latest"]
SELECTED_MODEL = None

# Find available Ollama model
def setup_ollama():
    """Check available Ollama models and select the best one"""
    global SELECTED_MODEL
    
    try:
        # Run the ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("✗ Error listing Ollama models")
            return False
        
        # Parse the output to get model names
        lines = result.stdout.strip().split('\n')[1:]  # Skip the header line
        available_models = []
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 1:
                available_models.append(parts[0])
        
        if not available_models:
            print("✗ No Ollama models found")
            return False
        
        print(f"Available Ollama models: {', '.join(available_models)}")
        
        # Select the preferred model if available
        for model in OLLAMA_MODELS:
            if model in available_models:
                SELECTED_MODEL = model
                print(f"✓ Selected Ollama model: {SELECTED_MODEL}")
                return True
        
        # If no preferred model is available, use any available model
        if available_models:
            SELECTED_MODEL = available_models[0]
            print(f"✓ Using available model: {SELECTED_MODEL}")
            return True
        
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama models: {e}")
        return False

# Initialize Ollama
setup_ollama()

# Add a list of greeting patterns to recognize
GREETING_PATTERNS = [
    "hi", "hello", "hey", "howdy", "greetings", "good morning", 
    "good afternoon", "good evening", "hiya", "what's up"
]

def get_chatbot_response(query, context=None):
    """Generate a concise, well-structured response using Ollama models"""
    print(f"Processing query: '{query}'")
    
    # Check if query is just a greeting
    query_lower = query.lower().strip()
    
    # Handle simple greetings with concise responses
    if query_lower in GREETING_PATTERNS or query_lower.startswith(tuple(GREETING_PATTERNS)):
        print("Identified as simple greeting - using concise response")
        return "Hi there! How can I help with your fashion questions today?"
    
    # Minimum query validation
    if not query or len(query.strip()) < 2:
        return "How can I help with your fashion questions today?"
    
    try:
        import ollama
        import re
        
        # Skip processing if no model is selected
        if not SELECTED_MODEL:
            return "I'm currently offline. Please check if Ollama is installed and models are available."
        
        # Create the system prompt with more specific formatting instructions
        system_prompt = f"""You are a helpful fashion assistant specializing in clothing advice.
           
Use this fashion knowledge in your responses:
{FASHION_KNOWLEDGE[:1000]}

RESPONSE FORMAT REQUIREMENTS:
- Use bullet points (•) for lists
- Include a short, clear headline before detailed explanation
- Keep the entire response under 100 words maximum
- Use markdown formatting: **bold** for important points
- For outfit suggestions, structure as "Top: ..., Bottom: ..., Accessories: ..."
- For color combinations, format as "Primary: ..., Secondary: ..., Accent: ..."
- NEVER use <think> tags or include your thinking process

Example response structure:
**Headline: Main Point**
• First key point with specific advice
• Second key point with practical example
• Third important consideration

Remember to be concise and direct with your fashion advice.
"""
        
        # Prepare the chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Include conversation history if available
        if context:
            # Extract the previous messages, limited to last 3 for context
            history_messages = []
            for msg in context[-3:]:  # Limit to last 3 messages
                if msg["role"] in ["user", "assistant"]:
                    history_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Insert history between system message and current query
            if history_messages:
                messages = [messages[0]] + history_messages + [messages[1]]
        
        print(f"Generating structured response using {SELECTED_MODEL}...")
        
        # Generate response with Ollama - ensure shorter responses with lowered num_predict
        response = ollama.chat(
            model=SELECTED_MODEL,
            messages=messages,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 300  # Lower token count for more concise responses
            }
        )
        
        # Extract and clean the response content
        model_response = response.get('message', {}).get('content', '').strip()
        
        # Process the response with improved cleaning
        
        # Remove any think tags and their content
        clean_response = re.sub(r'<think>.*?</think>', '', model_response, flags=re.DOTALL)
        
        # Remove any other formatting tags
        clean_response = re.sub(r'<[^>]+>', '', clean_response)
        
        # Remove any lines that look like prompt repetitions
        clean_response = re.sub(r'^.*?RESPONSE FORMAT REQUIREMENTS:.*$', '', clean_response, flags=re.MULTILINE)
        
        # Clean up excessive newlines and spaces
        clean_response = re.sub(r'\n{3,}', '\n\n', clean_response)
        clean_response = re.sub(r' {2,}', ' ', clean_response)
        
        # Hard limit on length - truncate if needed while maintaining completeness
        if len(clean_response) > 500:
            sentences = re.split(r'(?<=[.!?])\s+', clean_response)
            truncated_response = ""
            for sentence in sentences:
                if len(truncated_response + sentence) < 500:
                    truncated_response += sentence + " "
                else:
                    break
            clean_response = truncated_response.strip()
        
        if clean_response:
            print(f"Generated structured response ({len(clean_response)} chars)")
            return clean_response
        else:
            return "I'm having trouble formulating a response. Could you rephrase your fashion question?"
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm currently offline. Check if Ollama is running and try again soon."
