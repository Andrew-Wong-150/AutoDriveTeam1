import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


# load model with transferred learning weights
test_model = models.mobilenet_v3_small(pretrained=True)
test_model.classifier[3] = torch.nn.Linear(test_model.classifier[3].in_features, 4)
test_model.load_state_dict(torch.load('best_model.pth'))
test_model.eval()


def color_mask(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


    # read image
    image = cv2.imread(image_path)
    filtered_image = image & cv2.bitwise_not(image)
    
    # convert image color space
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
    
    # define HSV color bounds
    bounds = {
        'red': [[np.array([0, 50, 20]), np.array([20, 255, 255])], [np.array([150,50,20]), np.array([180,255,255])]],
        'yellow': [[np.array([15, 50, 20]), np.array([35, 255, 255])]],
        'green': [[np.array([40, 50, 20]), np.array([100, 255, 255])]] 
    }
    
    for color in bounds:
        
        # reset mask
        mask = 0
        
        # create color mask based on bounds
        for lower, upper in bounds[color]:
            mask += cv2.inRange(hsv_image, lower, upper)
        result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
        
        # apply mask to filtered image
        filtered_image = filtered_image | result
    
    # display image
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Processed", filtered_image)
    cv2.waitKey(0)
    
    return filtered_image


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
    
    
    # color mask image
    image = color_mask(image_path)
    image = Image.fromarray(image)
    
    # load image as tensor
    image = image_loader(image)
    
     #evaluate
    with torch.no_grad():
        output = torch.nn.functional.softmax(test_model(image), dim=1)
    
    # define labels
    #label_map = ['G', 'GL', 'GR', 'GS', 'O', 'R', 'RL', 'RR', 'RS', 'Y', 'YL', 'YR', 'YS']
    label_map = ['G', 'O', 'R', 'Y']
    
    # return top pick
    k = 3
    probabilities, labels = output.topk(k, dim = 1)
    for x in range(k):
        label = label_map[labels[0][x].item()]
        #return label
        probability = probabilities[0][x].item() * 100
        print('{:100} {:.2f}%'.format(label, probability))
        
    
    
def test():
    
    path = r'C:\Users\Andrew\Documents\AutoDrive\Transfer Learning\MCity\Red'
    counter = 0
    missed = 0

    for file in os.listdir(path):
        counter += 1
        result = predict(os.path.join(path, file))
        
        if 'R' not in result:
            print(file, result)
            missed += 1

    print('{:.2f}%'.format(1 - (missed/counter)))


#test()
predict(r'C:\Users\Andrew\Documents\AutoDrive\Transfer Learning\MCity\green\IMG_7593.jpg')
#color_mask(r'C:\Users\Andrew\Documents\AutoDrive\Transfer Learning\MCity\green\17.jpg')
#predict(r'C:\Users\Andrew\Documents\AutoDrive\Transfer Learning\MCity\red\10.jpg')








