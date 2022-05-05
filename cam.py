import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

def image_loader(image):
    
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


classes = ['Green', 'Red', 'Yellow']
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('ResNet18.pth'))
model.eval()

image_path = r'C:\Users\Andrew\Documents\AutoDrive\Color and Bulb Training\Old Test\Green\IMG_7594.jpg'
#image_path = r'C:\Users\Andrew\Documents\AutoDrive\Color and Bulb Training\New Test\Yellow\IMG_7890_1.jpg'
image = Image.open(image_path)
input_tensor = image_loader(image)

cam_extractor = GradCAM(model)
out = model(input_tensor)
#activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
print(classes[out.squeeze(0).argmax().item()])
activation_map = cam_extractor(0, out)

result = overlay_mask(image, transforms.functional.to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()