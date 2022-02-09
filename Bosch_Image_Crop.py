import os
import yaml
from PIL import Image

def parse_data(path, save=False, function=None):

    # load yaml file
    with open(os.path.join(path, 'train.yaml')) as stream:
        yaml_data = yaml.safe_load(stream)
        
        # iterate through images
        for data in yaml_data:
            
            # get image
            image_path = data['path']
            image = Image.open(os.path.join(path, image_path))
            
            # iterate through bounding boxes
            for box in data['boxes']:
                
                # get label
                label = box['label']
                
                # minimum bounding box size: 10x10
                if ((abs(box['x_max'] - box['x_min']) > 20) and (abs(box['y_max'] - box['y_min']) > 20)):
                    
                    # get cropped image
                    bbox = (box['x_min'], box['y_min'], box['x_max'], box['y_max'])
                    crop = image.crop(bbox)
                    
                    # save cropped image if flag is set to True
                    if save:
                        
                        base_path = os.path.join(path, 'sorted', label)
                        if not os.path.exists(base_path):
                            os.makedirs(base_path)
                        
                        file_name = os.path.join(base_path, image_path.rsplit('/')[-1])
                        crop.save(file_name)
                        
                    # call function if provided
                    if function:
                        function(crop, label)

parse_data(path = r'D:\Data\Bosch_Small_Traffic_Light_Dataset\dataset_train_rgb', save=True, function=None)
