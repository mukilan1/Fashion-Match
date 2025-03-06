import torch
from torchvision import transforms
from PIL import Image

# Try to load a pre-trained PyTorch model for sleeve classification.
# If the file is missing, set model to None.
try:
    model = torch.load("sleeve_classifier.pt", map_location=torch.device('cpu'))
    model.eval()
except FileNotFoundError:
    model = None

# List of sleeve labels in the same order as the model outputs
SLEEVE_LABELS = ["t-shirt", "sleeveless", "short sleeved", "full sleeved"]

# Preprocessing steps for the input image; adjust target size if needed.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_sleeve_type(image_path):
    if model is None:
        raise RuntimeError("Pre-trained sleeve model not found.")
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create mini-batch input
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.softmax(output, dim=1)
    index = probabilities.argmax().item()
    return SLEEVE_LABELS[index]
