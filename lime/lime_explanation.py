import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

def cnn_predict(model, images, device):
    transf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not isinstance(images, list):
        images = [images]
    
    batch = torch.stack([transf(img) for img in images], dim=0)
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch)
    
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def plot_explanations(img_pil, explanation):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    axes[0].imshow(img_pil)
    axes[0].axis('off')
    axes[0].set_title("Original Image")

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only=True, 
                                                negative_only=False, 
                                                num_features=5, 
                                                hide_rest=True)

    img_boundry = mark_boundaries(temp / 255.0, mask)
    axes[1].imshow(img_boundry)
    axes[1].set_title("Positive mask")
    axes[1].axis('off')

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only=False, 
                                                negative_only=True, 
                                                num_features=5, 
                                                hide_rest=True)

    img_boundry = mark_boundaries(temp / 255.0, mask)
    axes[2].imshow(img_boundry)
    axes[2].set_title("Negative mask")
    axes[2].axis('off')

    plt.show()

def explain_image(model, img_path, class_names, device):
    img = Image.open(img_path)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img), 
                                             lambda x: cnn_predict(model, [Image.fromarray(xi) for xi in x], device),  
                                             top_labels=2, 
                                             hide_color=0, 
                                             num_samples=1000)
    plot_explanations(img, explanation)
