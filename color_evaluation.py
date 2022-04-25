import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


# load model with transferred learning weights
test_model = models.mobilenet_v3_small(pretrained=True)
test_model.classifier[3] = torch.nn.Linear(test_model.classifier[3].in_features, 3)
test_model.load_state_dict(torch.load('color_classifier_no_generated.pth'))
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
        output = torch.nn.functional.softmax(test_model(image), dim=1)
    
    # define labels
    label_map = ['Green', 'Red', 'Yellow']
    
    # return top pick
    k = 2
    probabilities, labels = output.topk(k, dim = 1)
    for x in range(k):
        label = label_map[labels[0][x].item()]
        return label
        probability = probabilities[0][x].item() * 100
        print('{:100} {:.2f}%'.format(label, probability))
        
    
    
def test():

    colors = ['Red', 'Yellow', 'Green']
    base_path = r'C:\Users\Andrew\Documents\AutoDrive\Color and Bulb Training\New Test'
    
    for color in colors:
    
        path = os.path.join(base_path, color)
        counter = 0
        missed = 0
    
        for file in os.listdir(path):
            counter += 1
            result = predict(os.path.join(path, file))
            
            if color != result:
                #print(file, result)
                missed += 1
            
        print('{:<10}: {}/{} = {:.2f}%'.format(color, counter - missed, counter, (1 - (missed/counter)) * 100))



test()
#predict(r'C:\Users\Andrew\Documents\AutoDrive\Transfer Learning\test.jpg')
#predict(r'C:\Users\Andrew\Documents\AutoDrive\Transfer Learning\test.jpg')








