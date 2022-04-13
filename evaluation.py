import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


# load model with transferred learning weights
label_map = ['Bulb', 'Left', 'Right', 'Straight', 'Off', 'Red', 'Yellow', 'Green']
test_model = models.mobilenet_v3_small(pretrained=True)
test_model.classifier[-1] = torch.nn.Linear(test_model.classifier[-1].in_features, len(label_map))
test_model.load_state_dict(torch.load('test.pth'))
test_model.eval()


def image_loader(image):
    
    # process image into tensor
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = data_transforms(image).unsqueeze(0)
    return image


def predict(image_path):
    
    # load image as tensor
    image = image_loader(Image.open(image_path))
    
     #evaluate
    with torch.no_grad():
        preds = torch.sigmoid(test_model(image))
        preds = preds.tolist()[0]
        preds = ['%.2f' % pred for pred in preds]
        return preds
        


preds = predict(r'C:\Users\Andrew\Documents\AutoDrive\Multi Class Transfer Learning\MIT\ALL\MIT_59.jpg')
for x in label_map:
    print(x.center(10), end='')
print('')
for x in preds:
    print(x.center(10), end='')







