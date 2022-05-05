import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

root = r'C:\Users\Andrew\Documents\AutoDrive\Color and Bulb Training\Old Data'
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
classes = {0: 'Green',
           1: 'Red',
           2: 'Yellow'}

class cv_transform(object):
        def __call__(self, PIL_img):
            
            # read image
            image = np.array(PIL_img)
            
            # convert image color space
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ostu_value, thresh = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #cv2.imshow("Processed", thresh)
            #cv2.waitKey(0)
            image = Image.fromarray(image)
            
            return image
        
        def __repr__(self):
            return self.__class__.__name__+'()'
        

def split_data():
    
    dataset = datasets.ImageFolder(
        root,
        transforms.Compose([
            #cv_transform(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    
    return train_loader, test_loader


def evaluate(model, loader, num_classes):
    model.eval()
    correct = [0] * num_classes
    count = [0] * num_classes

    for x, y in loader:
      x, y = x.to(device), y.to(device)
      with torch.no_grad():
          logits = model(x)
          pred = logits.argmax(dim=1)
      for i in range(len(y)):
        count[y[i]] += 1
        correct[y[i]] += torch.eq(pred[i], y[i]).sum().float().item()
    
    accuracies = [i / j if j != 0 else None for i, j in zip(correct, count)]
    return sum(correct)/sum(count), accuracies


def run():
    
    train_loader, test_loader = split_data()
    
    #model = models.resnet18(pretrained=True)
    #model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model = model.to(device)
    
    NUM_EPOCHS = 30
    BEST_MODEL_PATH = 'new_transforms.pth'
    best_accuracy = 0.0
    
    writer = SummaryWriter('runs/new_transforms')
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(NUM_EPOCHS):
        
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        test_acc, test_class_acc = evaluate(model, test_loader, len(classes))
        train_acc, train_class_acc = evaluate(model, train_loader, len(classes))
        writer.add_scalars('test/acc', {'avg': test_acc,
                                      classes[0]: test_class_acc[0],
                                      classes[1]: test_class_acc[1],
                                      classes[2]: test_class_acc[2]}, epoch)
        writer.add_scalars('train/acc', {'avg': train_acc,
                                      classes[0]: train_class_acc[0],
                                      classes[1]: train_class_acc[1],
                                      classes[2]: train_class_acc[2]}, epoch)
        
        print("test accuracy by class:\n", test_class_acc)
        
        if test_acc > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_acc
        
        if all(i >= 0.98 for i in test_class_acc) and epoch > 10:
            print('Exit Early')
            break
            
    writer.close()
 
    
run()
        
        
