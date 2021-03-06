from PIL import Image
import os
from os import listdir
import time

correct_red = 0
correct_green = 0
correct_yellow = 0
undefined = 0
total_images = 0

filename = r"C:\Users\nwalt\source\repos\traffic_light_images\Mcity_image_TL_ssd\yellow"
t0 = time.time()
os.chdir(filename)
for images in os.listdir(filename):
    if (images.endswith(".jpg") or images.endswith(".png") or images.endswith(".jpeg")):
        img = Image.open(images).convert('RGB')
        w, h = img.size
        total_images += 1

        total_red = float(0)
        total_yellow = float(0)
        total_green = float(0)
        red_high = 0
        blue_high = 0
        green_high = 0

        for i in range(2, h-1):
            for j in range(2, w-1):
                red, green, blue = img.getpixel((j,i))
                if red > red_high:
                    red_high = red
                if green > green_high:
                    green_high = green
                elif blue > blue_high:
                    blue_high = blue
        scale_factor = 256.0/((red_high+green_high+blue_high)/3)

        for i in range(2, h-1):
            for j in range(2, w-1):
                red, green, _ = img.getpixel((j,i))
                if red-green >= 125/scale_factor:
                    red_pixel = 2
                    green_pixel = 0
                    yellow_pixel = 0
                elif green-red >= 125/scale_factor:
                    green_pixel = 2
                    red_pixel = 0
                    yellow_pixel = 0
                elif (abs(green-red)<50/scale_factor) & (green >= 150/scale_factor) & (red >= 150/scale_factor):
                    yellow_pixel = 2
                    red_pixel = 0
                    green_pixel = 0
                else:
                    red_pixel = 0
                    green_pixel = 0
                    yellow_pixel = 0
                total_red += red_pixel*((h-i)**2+(w/2-abs(j-w/2))**2)/256
                total_yellow += yellow_pixel*((h-i)**2+(w/2-abs(j-w/2))**2)/256
                total_green += green_pixel*((h-i)**2+(w/2-abs(j-w/2))**2)/256

        if total_yellow != 0:
            total_yellow = total_yellow**(1/1.8)
        if (total_red != 0) | (total_green != 0) | (total_yellow != 0):
            avg_red = total_red/(total_red+total_yellow+total_green)
            avg_yellow = total_yellow/(total_red+total_yellow+total_green)
            avg_green = total_green/(total_red+total_yellow+total_green)
        else:
            avg_green = 0
            avg_yellow = 0
            avg_red = 0
        print(avg_red)
        print(avg_yellow)
        print(avg_green)

        if (avg_red > avg_green) & (avg_red > avg_yellow):
            print('red\n')
            correct_red += 1

        elif (avg_green > avg_yellow):
            print('green\n')
            correct_green += 1
        elif (avg_yellow == 0):
            print('undefined\n')
            undefined += 1
        else:
            print('yellow\n')
            correct_yellow += 1
t1 = time.time()
percent_correct = correct_yellow/total_images
print('\n\nRed:', correct_red)
print('Yellow:', correct_yellow)
print('Green:', correct_green)
print('Unknown:', undefined)
print('Total:', total_images)
print('Percent Accuracy:', percent_correct)
print('Avg Time Per Image: ', (t1-t0)/total_images)