import cv2
import numpy as np
import shutil
import os

outputContourPath = "S:\\InnovationDevelopment\\PythonImgReader\\InputCharecter\\"

def delete_samples():
    for root, dirs, files in os.walk(outputContourPath):
        for f in files:
            if f.endswith('.png'):
                os.unlink(os.path.join(root, f))

def extract_char(input_img):
        # Read the image
        img2 = cv2.imread(input_img)

        #resize image
        img = cv2.resize(img2, (6000, 3000), interpolation = cv2.INTER_LINEAR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find contours in the binary image
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black image to draw the contours on
        black_img = np.zeros_like(img)

        # Loop through the contours
        i = 0
        for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w > 35 and h > 35 and y >2200:
                        
                        # Draw the contour on the black image
                        cv2.drawContours(black_img, [c], -1, (0, 255, 0), 2)

                        # Crop the contour from the original image
                        # roi = img[y:y+h, x:x+w]
                        roi = img[(y-2):(y+h + 2), (x-2):(x+w + 2)]
                        row, col = roi.shape[:2]
                        bottom = roi[row-2:row, 0:col]
                        mean = cv2.mean(bottom)[0]

                        border_size = 30
                        border = cv2.copyMakeBorder(
                        roi,
                        top=int(border_size/3),
                        bottom=int(border_size/3),
                        left=int(border_size),
                        right=int(border_size/3),
                        borderType=cv2.BORDER_CONSTANT,
                        value=[mean, mean, mean]
                        )



                        #resize image
                        roi2 = cv2.resize(border, (60, 60), interpolation = cv2.INTER_LINEAR)

                        # Save the contour as a separate image
                        cv2.imwrite(outputContourPath + "contour_" + str(i) + "_" + str(x)  + "_" + str(y) + "_" + str(w) + "_" + str(h) + ".png", roi2)


                        # Save the contour as an image
                        # cv2.imwrite(f"contour_{i}.png", black_img)

                        # Clear the black image for the next contour
                        black_img = np.zeros_like(img)
                        i += 1


        # # Loop through the contours
        # for c in cnts:
        #     # Get the bounding box of the contour
        #     x, y, w, h = cv2.boundingRect(c)

        #     # Check if the contour is large enough to be a character
        #     if w > 5 and h > 5:
        #         # Draw a rectangle around the contour
        #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #         # Save the contour as an image
        #         cv2.imwrite(f"contour_{i}.png", black_img)

        #         # Clear the black image for the next contour
        #         black_img = np.zeros_like(img)
        #         i += 1

        # # Show the output image
        # cv2.imshow("Output", img)
        # cv2.waitKey(0)

        # # Save the output image

        # cv2.imwrite("img\outputr.png", img)
