import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_model(device, model_path='inception_flower_classification_model.pth'):
    model = models.inception_v3(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 16)  # Adjust this to match the number of classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    return model

def predict_image(model, image_path, class_names, device):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Add a batch dimension

    with torch.no_grad():
        output = model(input_batch)
    _, predicted_class = output.max(1)
    predicted_class_name = class_names[predicted_class.item()]

    print(f'The predicted class is: {predicted_class_name}')

    image = np.array(image)
    plt.imshow(image)
    plt.axis('off')
    plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
    plt.show()

    return predicted_class_name
