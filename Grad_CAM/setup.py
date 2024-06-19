import torch
import torch.nn as nn 
import torchvision
import numpy as np
from torchvision.models import AlexNet
from torchvision.transforms import Resize , ToTensor
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import PIL
from IPython import display
import random
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data  import Dataset
import torch
import fastai
from fastai import vision
import os
import shutil
from sklearn.model_selection import train_test_split
#import pretrainedmodels as pm
import torch.nn.functional as F
import time
import glob
import os
import shutil
from typing import List
from torch.optim       import lr_scheduler
import numpy as np
import pandas as pd
import cv2
from utils import get_device


device, _= get_device(mps=True)

def get_tranformer():

    ''' Description: funtion that provide transform for the purpose of
     data augmentation
     arg: None
     return : train_transformer  , test_transformer
     
     '''
    

    train_transform0 = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(336),                                     
    transforms.RandomHorizontalFlip(),
    
    transforms.Resize((224 ,224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    


    test_transform0 = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    

     ])
    
    return  train_transform0 , test_transform0

 
def show(img):
    """Show PyTorch tensor img as an image in matplotlib."""
    npimg = 0.7*img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.grid(False)
    plt.gca().axis("off")
    plt.show();





def display_(data_loader, dataset_train):
    """
    Display a batch of images from the data_loader.

    Args:
        data_loader: DataLoader object containing images and labels.
        dataset_train: Dataset object containing class information.

    Returns:
        None, but displays images.
    """
    for images, labels in data_loader:
        print(images.shape)

        fig, axs = plt.subplots(8, 4, figsize=(10, 20))

        for i in range(len(images)):
            row = i // 4
            col = i % 4

            label = labels[i].item()
            axs[row, col].set_title(f"Class: {dataset_train.classes[label]}")

            img = images[i]
            img = img.cpu().numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            axs[row, col].imshow(img)
            axs[row, col].axis('off')

        plt.show()
        break   

# def display_(data_loader,dataset_train):

#     ''' description : Display the batch of image in data_loader
#         Arg: data_loader
#         return : None, but display
    
#     '''
#     for images , labels in  data_loader:
#         print(images.shape)
#         display.display(images[1])

#         for i in range(len(images)):

            

#             label = labels[i].item()
            
#             print(f"class:{dataset_train.classes[label]}==>{label}")

#             print(images[i].size())

#             show(img=images[i])
#         break


# Grad_CAM

def preprocess_image(image_path):

    ''' Function
        Description: return a tensor of an image from is path
        Arg: image_path

        return :  img_tensor
    
    '''

    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor



class GradCAM:
    '''
       Class:
       Description: To store the forward and gradient of the model to generate the CAM image

       attributes: model , gradient
       methodes : save_gradient , forward, generate_cam

       return : class cam
    '''


    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        self.model.zero_grad()
        x.requires_grad_()
        out = self.model(x)
        return out

    def generate_cam(self, image_tensor, target_class):
        output = self.forward(image_tensor).to(device)
        one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output.to(device))

        gradients = self.gradients.detach().cpu().numpy()
        feature_maps = self.model.feature_maps.detach().cpu().numpy()

        cam_weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)

        for i, weight in enumerate(cam_weights):
            cam += weight * feature_maps[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam
    



def register_hooks(model, grad_cam):
        def forward_hook(module, input, output):
            grad_cam.model.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            grad_cam.save_gradient(grad_output[0])

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                target_module = module

        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)



def apply_grad_cam(image_path, model, grad_cam):
        # Load and preprocess image
        image_tensor = preprocess_image(image_path).to(device)

        # Register hooks for Grad-CAM
        register_hooks(model, grad_cam)

        # Get the prediction from the model
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_class = probabilities.topk(1, dim=1)

        # Generate Grad-CAM
        cam = grad_cam.generate_cam(image_tensor, top_class.item())

        # Load the original image and overlay Grad-CAM
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (224, 224))
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        overlayed_image = cv2.addWeighted(original_image, 0.5, cam_heatmap, 0.5, 0)

        return top_class.item(), top_prob.item(), cam_heatmap, overlayed_image



def visualize_results(image_path, model, grad_cam,dataset_train):
    #     download_imagenet_labels()
#     imagenet_labels = load_imagenet_labels()

    flowers_labels = dataset_train.classes
    
    top_class, top_prob, cam_heatmap, overlayed_image = apply_grad_cam(image_path, model, grad_cam)
    class_label = flowers_labels[top_class]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(Image.open(image_path))
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    ax[1].imshow(cam_heatmap)
    ax[1].axis('off')
    ax[1].set_title('Grad-CAM Heatmap')

    ax[2].imshow(overlayed_image)
    ax[2].axis('off')
    ax[2].set_title(f'Overlayed Image (Class: {class_label}, Prob: {top_prob:.4f})')

    plt.show()



