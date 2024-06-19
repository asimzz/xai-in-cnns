
import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.nn. functional as F

from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image
from models import model1
from rise import RISE
from utils.visualize import visualize, reverse_normalize

from utils.setup import get_device , get_tranformer , give_data,train,model_evaluation




idx2label = { i:i for  i in range(0,15)}



label2idx = {label: idx for idx, label in idx2label.items()}







image = Image.open('/Users/atoukoffikougbanhoun/Desktop/AMMI/CV projects/XAI_project/data/train/astilbe/19596829_059e0d9d5a_c.jpg')
idx = label2idx[1]

imshow(image)
# preprocessing. mean and std from ImageNet
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])



# convert image to tensor
tensor = preprocess(image)

# reshape 4D tensor (N, C, H, W)
tensor = tensor.unsqueeze(0)
_, _, H, W = tensor.shape

# send tensor to gpu if gpu is available
device = 'mps:0' if torch.cuda.is_available else 'cpu'
tensor = tensor.to(device)



train_path =  "/Users/atoukoffikougbanhoun/Desktop/AMMI/CV projects/XAI_project/data/train"
test_path = "/Users/atoukoffikougbanhoun/Desktop/AMMI/CV projects/XAI_project/data/test"


train_transform , test_transform = get_tranformer()


train_loader, test_loader = give_data(train_path=train_path,test_path=test_path,
                                      train_transform=train_transform,test_transform=test_transform,
                                      batch_size=32
                                      )


Criterion = nn.CrossEntropyLoss()

learn_rate = 1e-4
optimizer = torch.optim.Adam(model1.parameters(),lr=learn_rate)




# training
num_epochs = 10
train(model1,Criterion, train_loader, optimizer, num_epochs=num_epochs)



model1.eval()

# send model to gpu if gpu is available
model = model1.to(device)


wrapped_model = RISE(model1, input_size=(H, W))
with torch.no_grad():
    saliency = wrapped_model(tensor)
saliency = saliency[idx]
# reverse normalization for display
img = reverse_normalize(tensor.to('cpu'))
saliency = saliency.view(1, 1, H, W)
heatmap = visualize(img, saliency)
hm = (heatmap.squeeze().numpy().transpose(1, 2, 0)).astype(np.int32)
imshow(hm)