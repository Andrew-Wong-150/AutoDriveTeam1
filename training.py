import os
import torch
import pandas as pd
from PIL import Image
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split


class TrafficLightDataset(Dataset):
    
    # initializes dataframe, transforms, and location of images
    def __init__(self, df, img_folder, transforms=None):
        self.df = df
        self.transforms = transforms
        self.img_folder = img_folder
  
    # returns one hot encoding vector as well as image tensor
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_folder, item.name))
        label = torch.tensor(item.tolist() , dtype=torch.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    # returns size of dataset
    def __len__(self):
        return len(self.df)


def get_data():
    
    # function to parse image data and store in dataframe
    def parse(folder_list):
        
        data = {}
        
        # iterate through files in folders
        for folder in folder_list:
            for file in os.listdir(os.path.join(root, folder)):
                
                # one-hot encode
                full_path = os.path.join(root, folder, file)
                data[file] = [0] * len(folder_list)
                data[file][folder_list.index(folder)] = 1
        
        # create dataframe
        df = pd.DataFrame(list(data.items()), columns=['name', 'data'])
        df.set_index('name', inplace=True)
        df[folder_list] = pd.DataFrame(df.data.tolist(), index = df.index)
        df.drop(columns='data', inplace=True)
        return df
    
    # function to resove NaN values
    def clean(x):
        
        # resolve missing color values for tranformed images
        if '_flipped' in x.name:
    
            original_name = x.name.replace('_flipped', '')     
            for color in color_folders:
                df.loc[x.name, color] = df.loc[original_name, color]
            
        # resolve missing direction values for 'Off' lights
        if x['Off'] == 1:
            for direction in direction_folders:
                df.loc[x.name, direction] = 0
    
    # create master datafram
    direction_df = parse(direction_folders)
    color_df = parse(color_folders)
    df = pd.concat([direction_df, color_df], axis=1)
    
    # for transformed direction image, get color data for original image
    df.apply(lambda x: clean(x), axis=1)
    return df
        

def split_data(dataset):
    
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size + val_size, test_size])
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

'''
-------------------------------------
NOT WORKING: does not compute accuray
-------------------------------------
'''
def evaluate(device, model, loader, outputs):
    
    model.eval()
    correct = 0
    total = 0

    for xs, ts in loader:
      xs, ts = xs.to(device), ts.to(device)
      
      with torch.no_grad():
          preds = torch.sigmoid(outputs).data > 0.5
          preds = preds.to(torch.float32)

    return 0, [0]


def train(train_loader, val_loader, test_loader):
    
    # determines whether to run on cpu or gpu
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)
    
    # changes number of output features to match number of classes
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(device)
    
    # writer for tensorboard
    writer = SummaryWriter('runs/multi_classifier')
    
    # loss function and optimizer
    best_accuracy = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # train
    for epoch in range(num_epochs):
        
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        val_acc, val_class_acc = evaluate(device, model, val_loader, outputs)
        train_acc, train_class_acc = evaluate(device, model, train_loader, outputs)
        
        writer.add_scalars('val/acc', {'avg': val_acc}, epoch)
        writer.add_scalars('train/acc', {'avg': train_acc}, epoch)
        
        if val_acc > best_accuracy or True:
            torch.save(model.state_dict(), model_path)
            best_accuracy = val_acc
        
        if all(i >= 0.95 for i in val_class_acc):
            break

    writer.close()
    

if __name__ == '__main__':
    
    root = r'C:\Users\Andrew\Documents\AutoDrive\Multi Class Transfer Learning\MIT'
    direction_folders = ['Bulb', 'Left', 'Right', 'Straight']
    color_folders = ['Off', 'Red', 'Yellow', 'Green']
    
    num_classes = len(direction_folders) + len(color_folders)
    num_epochs = 16
    model_path = 'test.pth'
    
    df = get_data()
    img_folder = r'C:\Users\Andrew\Documents\AutoDrive\Multi Class Transfer Learning\MIT\ALL'
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TrafficLightDataset(df, img_folder, transform)
    train_loader, val_loader, test_loader = split_data(dataset)
    train(train_loader, val_loader, test_loader)

