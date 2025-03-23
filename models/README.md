# Local Models Directory

This folder stores local AI models for the fashion assistant.

## Supported Model Types

The system supports these types of local models:

1. **Ollama Models** - Most efficient option, requires Ollama to be installed
   - Install Ollama from: https://ollama.com/
   - Pull fashion-friendly models: `ollama pull gemma:2b` or `ollama pull tinyllama`

2. **Transformers Models** - Downloaded automatically when needed
   - TinyLlama (1.1B parameters)
   - Other small models that can run on CPU

3. **GGUF Models** - For use with llama.cpp
   - Place .gguf model files in this directory
   - Recommended: Phi-2.Q4_K_M.gguf (small and fashion-friendly)
   - Download from: https://huggingface.co/TheBloke/phi-2-GGUF/tree/main

## Automatic Selection

The application will automatically select the most efficient available model:
1. First choice: Ollama models
2. Second choice: Transformers models 
3. Third choice: GGUF models
4. Fallback: Pattern-based responses

You do not need to install all model types - the system will use whatever is available.
