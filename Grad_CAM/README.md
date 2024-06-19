# IMPLEMENTATION OF GradCAM TECHNIQUE



Train  ResNet18 on our dataset of flowers with 16 
    and  applying the Grad-CAM technique with,

`GradCAM Class`: Captures gradients and feature maps from the target layer to generate a heatmap highlighting important image regions for a specific class.  
`registers Function`:Registers to store gradients and feature maps from the target convolutional layer during forward and backward passes.  
`applyGradCam Function`: Applies Grad-CAM to an input image, generates a heatmap, and overlays it on the original image.  
`visualize Results Function`: Visualizes the original image, the GradCAM heatmap, and the overlayed image with the predicted class label and probability.