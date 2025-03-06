import torch
from torchvision import transforms
from PIL import Image

try:
    model = torch.load("hand_classifier.pt", map_location=torch.device('cpu'))
    model.eval()
except FileNotFoundError:
    model = None

# Define hand labels in the order of model outputs
HAND_LABELS = ["full hand", "half hand", "no hand"]

# Preprocess input images for the hand classifier model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_hand_type(image_path):
    if model is None:
        raise RuntimeError("Pre-trained hand classifier model not found.")
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.softmax(output, dim=1)
    index = probabilities.argmax().item()
    return HAND_LABELS[index]
