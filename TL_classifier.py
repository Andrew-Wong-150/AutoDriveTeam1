import glob
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

IMAGE_DIR_TRAINING = "../0_Datasets/traffic_light_images/training/"
IMAGE_DIR_TEST = "../0_Datasets/traffic_light_images/test/"
IMAGE_DIR_Mcity = "../0_Datasets/Mcity_image_TL_ssd_color/"

# ------------------- Global Definitions -------------------
# Definition of the 3 possible traffic light states and theirs label
tl_states = ['red', 'yellow', 'green']
tl_state_red = 0
tl_state_yellow = 1
tl_state_green = 2
tl_state_count = len(tl_states)
tl_state_red_string = tl_states[tl_state_red]
tl_state_yellow_string = tl_states[tl_state_yellow]
tl_state_green_string = tl_states[tl_state_green]

# Index of image and label in image set
image_data_image_index = 0
image_data_label_index = 1

# Normalized image size
default_image_size = 32

crop_left_right = 12
crop_top_bottom = 3

# Thresholds for dominant colors 
dominant_sure_threshold = 0.15
dominant_threshold = 0.015


# ---------------- End of Global Definitions ---------------

def load_images(imgae_dir):
    image_list = []
    for name in glob.glob(imgae_dir+"*/*"):
        image_list.append([cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB), name.split("/")[-2]])
    return image_list

def standardize_input(image):
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = cv2.resize(image.astype('uint8'), dsize=(default_image_size, default_image_size))
    
    return standard_im

## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [0, 0, 0] 
    for state_index in range(tl_state_count):
        if label==tl_states[state_index]:
            one_hot_encoded[state_index] = 1
    
    return one_hot_encoded

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

def mask_image_get_brightness_vector(rgb_image):
    """
    Tries to identify highlights within the traffic light's inner region and removes a vector with the
    brightness history from top to bottom
    
    rgb_image: An RGB image of a traffic light
    return: The history vector
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv = hsv[crop_top_bottom:default_image_size-crop_top_bottom,crop_left_right:default_image_size-crop_left_right]
    brightness = hsv[:,:,2]
    summed_brightness = np.sum(brightness, axis=1)
    
    return (brightness,hsv[:,:,1],summed_brightness)

def get_color_dominance(rgb_image):
    """This function searches for a very dominant red, yellow or green color within the traffic lights
    inner image region and independent of it's position
    
    rgb_image: The traffic light image
    return: A vector containing the percentage of red, yellow and green, (NOT RGB channels!) within the image
    """
    
    agg_colors = [0,0,0]
    
    cropped_image = rgb_image[crop_top_bottom:default_image_size-crop_top_bottom,crop_left_right:default_image_size-crop_left_right]

    threshold_min = 140
    threshold_min_b = 120
    threshold_rel = 0.75
    total_pixels = len(cropped_image)*len(cropped_image[1])

    for row_index in range(len(cropped_image)):
        cur_row = cropped_image[row_index]
        for col_index in range(len(cropped_image[0])):
            pixel = cur_row[col_index]
            if pixel[0]>threshold_min and pixel[1]<pixel[0]*threshold_rel and pixel[2]<pixel[0]*threshold_rel:
                agg_colors[0] += 1
            if pixel[0]>threshold_min and pixel[1]>threshold_min and pixel[2]<pixel[0]*threshold_rel:
                agg_colors[1] += 1
            if pixel[1]>threshold_min and pixel[0]<pixel[1]*threshold_rel and pixel[2]>threshold_min_b:
                agg_colors[2] += 1

    agg_colors = np.array(agg_colors)/float(total_pixels)
    
    return agg_colors

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def create_feature(rgb_image):
    """
    Creates a brightness feature using the image of a traffic light
    
    
    rgb_image: An RGB image of a traffic light
    return: (The brightness mask, The saturation mask, The brightness history vector from top to bottom)"""
    (img_bright, img_sat, sb) = mask_image_get_brightness_vector(rgb_image)
    
    ## TODO: Create and return a feature value and/or vector
    feature = sb
    
    return feature

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label_color(rgb_image):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    
    # get the brightness vector feature first, this is a great fallback in any case
    feature = create_feature(rgb_image)
    
    # search for a visually dominant color as well
    dominant = get_color_dominance(rgb_image)
    
    max_dominant = np.argmax(dominant)
    
    one_hot = [0,0,0]
    
    maxc = len(feature)//3*3
    div = maxc//3
    prob = [np.sum(feature[0:div]), np.sum(feature[div:2*div]), np.sum(feature[2*div:3*div])]
    
    one_hot[np.argmax(prob)] = 1
    
    red_yellow_tolerance = 0.8
    
    # if one color is so dominant that it's not disusable: take it
    # if the algorithm is unsure combine it with the knowledge obtained by the brightness vector
    if(dominant[max_dominant]>dominant_threshold):  # is there a very dominant color ?
        # if max_dominant==tl_state_red or max_dominant==tl_state_yellow:
        #     val = dominant[max_dominant]
        #     scaled_val = val*red_yellow_tolerance
        #     if scaled_val<dominant[0] and scaled_val<dominant[1]:
        #         return one_hot
            
        one_hot = [0,0,0]
        one_hot[max_dominant] = 1
        return one_hot

    return one_hot

def get_misclassified_images_color(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label_color(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set

IMAGE_TRAIN_LIST = load_images(IMAGE_DIR_TRAINING)
IMAGE_TEST_LIST = load_images(IMAGE_DIR_TEST)
IMAGE_MCITY_LIST = load_images(IMAGE_DIR_Mcity)

STANDARDIZED_TRAIN_LIST = standardize(IMAGE_TRAIN_LIST)
STANDARDIZED_TRAIN_LIST = standardize(IMAGE_TEST_LIST)
STANDARDIZED_MCITY_LIST = standardize(IMAGE_MCITY_LIST)

MISCLASSIFIED = get_misclassified_images_color(STANDARDIZED_MCITY_LIST)

# Accuracy calculations
total = len(STANDARDIZED_MCITY_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total
opencv_accuracy = accuracy*100.0

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))