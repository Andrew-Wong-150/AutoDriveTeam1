from PIL import Image, ImageDraw
import os
from os import listdir
import time
import cv2 as cv
import numpy as np


total_images = 0
x_green = []
y_green = []

left_x = 10000000
high_y = 100000
right_x = 0
low_y = 0

filename = [r"C:\Users\nwalt\source\repos\traffic_light_images\Mcity_image_TL_sorted\red"]
t0 = time.time()
for z in range(1):
    
    for images in os.listdir(filename[z]):
        os.chdir(filename[z])
        if (images.endswith(".jpg") or images.endswith(".png") or images.endswith(".jpeg")):
            img = Image.open(images).convert('RGB')

            w, h = img.size
            total_images += 1
            left_x = 10000000
            high_y = 100000
            right_x = 0
            low_y = 0

            red_high = 0
            blue_high = 0
            green_high = 0
            for i in range(2, h-1):
                for j in range(2, w-1):
                    red, green, blue = img.getpixel((j,i))
                    if red > red_high and green < red*0.5 and blue < red*0.5:
                        red_high = red
                    if green > green_high:
                        green_high = green
                    if blue > blue_high:
                        blue_high = blue
            scale_factor = 256.0/(red_high)

            for i in range(w):
                for j in range(h):
                    red, green, blue = img.getpixel((i,j))

                    if red > 80*scale_factor and green < red*0.5 and blue < red*0.5:
                        if i < left_x:
                            left_x = i
                        if i > right_x:
                            right_x = i
                        if j < high_y:
                            high_y = j
                        if j > low_y:
                            low_y = j
                        img.putpixel((i,j), (255,255,255))
                    else:
                        img.putpixel((i,j), (0,0,0))

            #print(left_x, high_y, left_x, low_y)

            os.chdir(filename[z] + r'\red_detection')

            img.save(images)

            im = cv.imread(images)
            imgray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
            #ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
            try:
                contours, _ = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                print(len(contours))
                areas = [cv.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                m = contours[max_index]

                x,y,w,h = cv.boundingRect(m)
                print(x,y,w,h)

                os.chdir(filename[z])
                img = cv.imread(images)
                os.chdir(filename[z] + r'\red_detection')
                cv.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 3)
                cv.imwrite(images, img)
            except:
                print("no red")
            #print(m)

            #draw = ImageDraw.Draw(img)
            #draw.line((x,y,x,y+h), width = 2)
            #draw.line((x,y,x+w,y), width = 2)
            #draw.line((x+w,y,x+w,y+h), width = 2)
            #draw.line((x,y+h,x+w,y+h), width = 2)