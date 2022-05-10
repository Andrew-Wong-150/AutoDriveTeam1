from PIL import Image, ImageDraw
import os
from os import listdir
import time
import cv2 as cv
import numpy as np


filename = [r"C:\Users\nwalt\source\repos\traffic_light_images\Mcity_image_TL_sorted\green"]
t0 = time.time()
for z in range(1):
    
    for images in os.listdir(filename[z]):
        os.chdir(filename[z])
        if (images.endswith(".jpg") or images.endswith(".png") or images.endswith(".jpeg")):
            img = Image.open(images).convert('RGB')

            w, h = img.size
            left_x = 10000000
            high_y = 100000
            right_x = 0
            low_y = 0

            red_high = 0
            blue_high = 0
            green_high = 0
            for i in range(h):
                for j in range(w):
                    red, green, blue = img.getpixel((j,i))
                    if red > red_high and green < red*0.5 and blue < red*0.5:
                        red_high = red
                    if green > green_high and red < green and blue < green:
                        green_high = green
                    if blue > blue_high:
                        blue_high = blue
            scale_factor = (green_high)/256.0

            for i in range(w):
                for j in range(h):
                    red, green, blue = img.getpixel((i,j))

                    if green > 120*scale_factor and red < green*1.05 and blue < green*1.05:
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

            print(images)

            #print(left_x, high_y, left_x, low_y)

            os.chdir(filename[z] + r'\green_to_yellow')

            img.save(images)

            im = cv.imread(images)
            os.chdir(filename[z])
            img = cv.imread(images)
            os.chdir(filename[z] + r'\green_to_yellow')
            imgray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
            #ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
            try:
                contours, _ = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                #print(len(contours))
                areas = [cv.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                m = contours[max_index]

                x,y,w,h = cv.boundingRect(m)
                print(x,y,w,h)

                #cv.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
                cv.imwrite(images, img)

                img = Image.open(images)
                try:
                    for i in range(w):
                        for j in range(h):
                            red, green, blue = img.getpixel((x+i,y+j))

                            if green > 110*scale_factor and red < green*1.1 and blue < green*1.1:
                                img.putpixel((x+i,y+j), (green,int(green*0.85), int(blue*0.5)))

                    img.save(images)
                except:
                    os.chdir(filename[z] + r'\green_to_yellow')
                    os.remove(images)

            except:
                os.chdir(filename[z] + r'\green_to_yellow')
                os.remove(images)

            

            #print(m)

            #draw = ImageDraw.Draw(img)
            #draw.line((x,y,x,y+h), width = 2)
            #draw.line((x,y,x+w,y), width = 2)
            #draw.line((x+w,y,x+w,y+h), width = 2)
            #draw.line((x,y+h,x+w,y+h), width = 2)