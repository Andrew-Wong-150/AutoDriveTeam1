import cv2
import numpy as np
import math


def color_mask(bounds, image):
    
    # convert image color space
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # create color mask based on bounds
    mask = 0
    
    for lower, upper in bounds:
        mask += cv2.inRange(hsv_image, lower, upper)
    
    # mask color
    result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
    cv2.imshow('Mask', result)
    cv2.waitKey(0)
    return result


# NOT WORKING
def threshold(image):
    
    # convert image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # calcuate block size based on image dimension to blur
    block_dim = int(gray_image.shape[0] / 100)
    if block_dim % 2 == 0:
        block_dim += 1
    blurred = cv2.GaussianBlur(gray_image, (block_dim, block_dim), 0)

    # apply adaptive threshold
    ostu_value, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_dim, 0)
    
    # apply erosion and dialation
    #kernel = np.ones((block_dim, block_dim), np.uint8)
    #erosion = cv2.erode(thresh, kernel, iterations = 1)
    
    #kernel = np.ones((block_dim, block_dim), np.uint8)
    #dilation = cv2.dilate(thresh, kernel, iterations = 1)
    
    # convert back to rgb
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.imshow('Contour', thresh)
    cv2.waitKey(0)

    cv2.imshow('Otus Adaptive Thresholding', result)
    return thresh


def contour_map(image):
    
    # convert image colorspace to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # calcuate block size based on image dimension to blur
    block_dim = int(gray_image.shape[0] / 10)
    if block_dim % 2 == 0:
        block_dim += 1
        
    blurred = cv2.GaussianBlur(gray_image, (block_dim, block_dim), 0)
    
    # apply adaptive threshold
    ret, thresh = cv2.threshold(blurred, 1, 255, cv2.THRESH_OTSU)
    
    # find and measure largest contour
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, 0, (0,255,0), 20)
    cv2.imshow('Contour', image)
    cv2.waitKey(0)
    
    if contours:
        area = cv2.contourArea(contours[0])
        return area
    return 1
    

def predict_image(file):
    
    # read and rescale smallest dimension to ~200px if less than 200px
    image = cv2.imread(file)
    scale_factor = math.ceil(200 / min(image.shape[0:1]))
    image = cv2.resize(image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation = cv2.INTER_AREA)
    
    filtered_image = image & cv2.bitwise_not(image)
    
    bounds = {
        'red': [[np.array([0, 50, 20]), np.array([20, 255, 255])], [np.array([150,50,20]), np.array([180,255,255])]],
        'yellow': [[np.array([15, 50, 20]), np.array([35, 255, 255])]],
        'green': [[np.array([40, 50, 20]), np.array([100, 255, 255])]] 
    }
    masks = {}
    masks = {}
    areas = {'Off': 0}
    
    for key in bounds:
        
        masks[key] = color_mask(bounds[key], image)
        areas[key] = contour_map(masks[key])
        filtered_image = filtered_image | masks[key]
    
    return(max(areas, key=areas.get))
    #cv2.imshow("Processed", filtered_image)
    #cv2.waitKey(0)


predict_image('green light.jpg')

