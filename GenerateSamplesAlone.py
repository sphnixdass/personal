import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageOps
import random

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
import keras
from keras import layers

import shutil

img_width = 60
img_height = 60

outputFolder = 'S:\\InnovationDevelopment\\ChequeSamples\\img'

def delete_samples(outputFolder):
    for root, dirs, files in os.walk(outputFolder):
        for f in files:
            if f.endswith('.png'):
                os.unlink(os.path.join(root, f))


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
  return result

def zoom_at(img, zoom, angle, coord):
    # print(img.shape)
    cy, cx = img.shape
    cy = cy/2
    cx = cx/2

    # cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

def generate_sample_with_font():
    img_width = 60
    img_height = 60

    delete_samples()
    rc =1

        
    #number of sample image on each catergories
    for j in range(0,500):

            


        # x1,y1 ------
        # |          |
        # |          |
        # |          |
        # --------x2,y2

        #selecting numbers
        for i in range(0,10):

            img_3 = np.zeros([img_height,img_width,3],dtype=np.uint8)
            img_3.fill(255)

            # Convert to PIL Image
            cv2_im_rgb = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)

            draw = ImageDraw.Draw(pil_im)

            # Choose a font
            font = ImageFont.truetype("micr-e13b.ttf", 72)
        
            # Draw the text
            # text_layer = Image.new('L', (img_height,img_weight))
            # draw = ImageDraw.Draw(text_layer)
            draw.text((random.randint(0, 10), random.randint(0, 10)), str(i), fill ="black", font=font)
            # rotated_text_layer = text_layer.rotate(10.0, expand=1)

            # rotate image

            # pil_im = pil_im.rotate(random.randint(-10, 10), Image.NEAREST, expand = 0)

            # Save the image

            # pil_im.paste( ImageOps.colorize(rotated_text_layer, (100,0,0), (255, 200,1)),  rotated_text_layer)
            cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        #noise range
            for k in range(0, random.randint(5, 20)):

                random_x = random.randint(0, img_width)
                random_y = random.randint(0, img_height)
                noise_x1 = random_x
                noise_y1 = random_y

                noise_x2 = random_x + 3
                noise_y2 = random_y + 3

                noise_random_color = 0
                if random.randint(0, 1) == 0:
                    noise_random_color = 0
                else:
                    noise_random_color = 255
                    
                cv2.rectangle(cv2_im_processed, (noise_x1, noise_y1), (noise_x2, noise_y2), color=(noise_random_color,noise_random_color,noise_random_color), thickness=random.randint(1, 5))

            cv2_im_processed = rotate_image(cv2_im_processed, random.randint(-25, 25))
            cv2_im_processed = cv2.cvtColor(cv2_im_processed, cv2.COLOR_BGR2GRAY)
            temppath = outputFolder + r"\\" + str(i) + "\\" + str(i) + "_" + str(rc)  + ".png"
            print(temppath)
            cv2.imwrite(temppath, cv2_im_processed)
            rc = rc + 1


def resize_pdf_raw_img():
    img_width = 250
    img_height = 2200

    #ulster1001 0
    #ulster9508 1
    #ulster9509 2
    #ulsterccs9874 3
    #ulsterrwp9557 4
    #
    flists = ["ulster1001", "ulster9508", "ulster9509", "ulsterccs9874", "ulsterrwp9557"]
    frc = 0
    for flist in flists:

        foldernametmp = flist
        foldernametmp2 = str(frc)
        frc = frc + 1 
        #input folder
        data_dir2 = Path("X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\" + foldernametmp + "\\")
        images2 = sorted(list(map(str, list(data_dir2.glob("*.png")))))

        #output folder path
        outputRawPath = "S:\\InnovationDevelopment\\PythonImgReader\\HeaderImg\\InputImp\\" + foldernametmp + "\\"

        delete_samples(outputRawPath)
        # Read the image
        rc = 1
        for imgtmppath in images2:

            print(imgtmppath)

            img2 = cv2.imread(imgtmppath)

            #resize image
            img = cv2.resize(img2, (6000, 3000), interpolation = cv2.INTER_LINEAR)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to get a binary image
            #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            #crop image
            #gray[y:y+h, x:x+w]
            crop_img = gray[400:-400, :img_width]

            #resize image
            roi2 = cv2.resize(crop_img, (img_width, img_height), interpolation = cv2.INTER_LINEAR)


            for i in range(1, 501):
                cv2_im_processed = cv2.cvtColor(roi2, cv2.COLOR_RGB2BGR)

                cv2_im_processed = rotate_image(cv2_im_processed, random.randint(-10, 10))
            #noise range
                for k in range(0, random.randint(5, 2000)):

                    random_x = random.randint(0, img_width)
                    random_y = random.randint(0, img_height)
                    noise_x1 = random_x
                    noise_y1 = random_y

                    noise_x2 = random_x + 3
                    noise_y2 = random_y + 3

                    noise_random_color = 0
                    if random.randint(0, 1) == 0:
                        noise_random_color = 0
                    else:
                        noise_random_color = 255
                        
                    cv2.rectangle(cv2_im_processed, (noise_x1, noise_y1), (noise_x2, noise_y2), color=(noise_random_color,noise_random_color,noise_random_color), thickness=random.randint(1, 5))

                cv2_im_processed = cv2.cvtColor(cv2_im_processed, cv2.COLOR_BGR2GRAY)

                # cv2_im_processed = zoom_at(cv2_im_processed, random.randint(0, 1),0,None)

                    #save the resized image
                cv2.imwrite(outputRawPath + foldernametmp2 + "_" + str(rc) + "_" + foldernametmp + "_.png", cv2_im_processed)
                rc = rc + 1

resize_pdf_raw_img()
#generate_sample_with_font()
