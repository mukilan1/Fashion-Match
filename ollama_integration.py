import ollama
import re

model_name = "deepseek-r1:1.5b"  # or "llama3.2:latest"

response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': 'what is your name'}])

# Remove <think> tags and limit response to one line
cleaned_response = re.sub(r"<think>.*?</think>", "", response['message']['content'], flags=re.DOTALL).strip()
print(cleaned_response.split("\n")[0])