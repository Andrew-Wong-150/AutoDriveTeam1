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

root = r'C:\Users\Andrew\Documents\AutoDrive\Color and Bulb Training\Old'
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
num_classes = 3
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
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size + val_size, test_size])
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
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
    
    
    return train_loader, val_loader, test_loader


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
    print("accuracy_by_class\n", accuracies)
    return sum(correct)/sum(count), accuracies


def run():
    
    train_loader, val_loader, test_loader = split_data()
    
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)
    
    NUM_EPOCHS = 16
    BEST_MODEL_PATH = 'color_classifier_no_generated.pth'
    best_accuracy = 0.0
    
    writer = SummaryWriter('runs/color_classifier_no_generated')
    
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
        
        val_acc, val_class_acc = evaluate(model, val_loader, num_classes)
        train_acc, train_class_acc = evaluate(model, train_loader, num_classes)
        writer.add_scalars('val/acc', {'avg': val_acc,
                                      classes[0]: val_class_acc[0],
                                      classes[1]: val_class_acc[1],
                                      classes[2]: val_class_acc[2]}, epoch)
        writer.add_scalars('train/acc', {'avg': train_acc,
                                      classes[0]: train_class_acc[0],
                                      classes[1]: train_class_acc[1],
                                      classes[2]: train_class_acc[2]}, epoch)
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = val_acc
        
        if all(i >= 0.99 for i in val_class_acc) and epoch > 5:
            break
            
    writer.close()
 
    
run()
        
        
