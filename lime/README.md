# Local Interpretable Model-agnostic Explanations (LIME)
## Introductiom
The original data points are perturbed before feeding them into the black box model and then the corresponding outputs are observed. The method then weighs those new data points as a function of their linear proximity to the original point. It then fits a surrogate faithful model like linear regression on the dataset with variations using the sample weights.
## User Instructions
This repository consists of several .py files:
* data_transformation.py:
Here we transform the image data to the required format by resizing, flipping the images, normalizing and converting the images to tensor.
* data_preparation.py:
Here we load our dataset and split it into train and validation sets before training our model.
* lime_explanation.py:
We have our CNN predict function , plotting the original image and the masked images that capture the image regions that contributes to our CNNs prediction.
* Inference.py:
Helps us to load the model, predict the image class and show the predicted image.
* training.py:
We train our model on the pretrained Inceptionv3 model.
* main.py:
We call all the above .py files here. This file then runs the entire code producing our output as shown below.

![lime_tulip](https://github.com/asimzz/xai-in-cnns/assets/162570349/2f217564-8426-4601-ba68-b8c7b904be50)

We can observe that the petals and the stalk of the flowers contribute positively to the model predicting it as a tulip with a probability of $0.9999$. On the other hand the background contributes negatively.
  
