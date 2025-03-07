import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained model (ResNet or Vision Transformer)
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and preprocess image
image_path = "s1.png"
image = Image.open(image_path)
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image)
print(output.argmax())  # Sample classification output