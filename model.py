import torch
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

# Download ImageNet classes if not present
classes_file = "imagenet_classes.txt"
if not os.path.exists(classes_file):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        classes_file
    )
with open(classes_file, "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing for input images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_cloth(image_path):
    if not os.path.exists(image_path):
        return "unknown"
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    # Return the predicted label from ImageNet classes
    return imagenet_classes[predicted_idx.item()]
