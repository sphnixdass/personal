
import json

#X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\20231117BOIInClearingBK1800HR-1.png
#r"X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\20231117BOI2InClearingBK1800HR-1.png"
#'X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\20231117BOIInClearingBK1800HR-3.png'
import cv2
import numpy as np
import shutil
import os, sys
from PIL import ImageFont, ImageDraw, Image, ImageOps
import random


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
import keras
from keras import layers
# import numpy
import scipy.cluster.hierarchy as hcluster


import shutil


DassEnv = "dev" #dev or prod

outputContourPath = "D:\\Application\\SCEPOC\\PythonAPI\\py\\ChequeInputChar\\"
inputFolderPath = "D:\\Application\\SCEPOC\\PythonAPI\\Data\\Cheque\\"
# modeledb3 = "D:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelEdb13"
# modeleorcb = "D:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelfont2"
data_dir2 = Path("D:\\Application\\SCEPOC\\PythonAPI\\py\\ChequeInputChar\\")

if DassEnv == "dev":
    outputContourPath = "P:\\Application\\SCEPOC\\PythonAPI\\py\\ChequeInputChar\\"
    inputFolderPath = "P:\\Application\\SCEPOC\\PythonAPI\\Data\\Cheque\\"
    # modeledb3 = "P:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelEdb13"
    # modeleorcb = "P:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelfont2"
    data_dir2 = Path("P:\\Application\\SCEPOC\\PythonAPI\\py\\ChequeInputChar\\")
else:
    outputContourPath = "D:\\Application\\SCEPOC\\PythonAPI\\py\\ChequeInputChar\\"
    inputFolderPath = "D:\\Application\\SCEPOC\\PythonAPI\\Data\\Cheque\\"
    # modeledb3 = "D:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelEdb13"
    # modeleorcb = "D:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelfont2"
    data_dir2 = Path("D:\\Application\\SCEPOC\\PythonAPI\\py\\ChequeInputChar\\")



        # print("EEEEEEEEEEEEEE")
    # modelCheque = keras.models.load_model("D:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodel")

img_width = 60
img_height = 60
# generate_sample()
# Path to the data directory
# data_dir = Path("./img/")


print("Python cheque read called ... ")
# Batch size for training and validation
batch_size = 16
# batch_size = 20

# Desired image dimensions
# img_width = 200
# img_height = 50

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4





def CheqeuMain(modelChequearg, modelChequearg2):
    
    # modelCheque = modelChequearg

    all_files = os.listdir(inputFolderPath)
    png_files = filter(lambda x: x[-4:] == ".png", all_files)
    for png_file in png_files:
        try:
            inputTempFileNa = inputFolderPath + png_file
            print("Clearing old temp files....")
            delete_samples()
            print("Processing file name", inputTempFileNa)
            print("Extracting characters from image....")
            extract_char(inputTempFileNa)
            if DassEnv != "dev":
                os.remove(inputTempFileNa)
            lst = os.listdir(outputContourPath)
            if len(lst) > 5:
                print("Artificial Intelligence deep learning neural networks model loading........")
                validation_dataset, prediction_model, num_to_char, images2 = load_mod(modelChequearg, modelChequearg2, 3)
                print("Artificial Intelligence predicting the characters........")
                resultcal = model_pred(validation_dataset, prediction_model,num_to_char, images2)

                print("Dass", resultcal[0][0], resultcal[0][-1])

                filtered_list = list(filter(lambda row: int(row[2]) >= (int(resultcal[0][2]) - 150), resultcal))

                sorted_list = sorted(filtered_list, key=lambda x: int(x[1]))  # Sort by second column

                # print(sorted_list)
                # get average width and height
                avgHeightList = []
                avgWeightList = []

                for x in sorted_list:
                    # print((x[3].split('_')[-1]).replace(".png",""))
                    avgHeightList.append(int((x[3].split('_')[-1]).replace(".png","")))
                    avgWeightList.append(int((x[3].split('_')[-2])))
                
                avgHeight = sum(avgHeightList) / len(avgHeightList)
                avgWeightList = sum(avgWeightList) / len(avgWeightList)

                print("avg hight", avgHeight, (avgHeight * 2))
                print("avg weight", avgWeightList, (avgWeightList * 2 ))

                outputList = []
                outputList2 = []
                precol = int(sorted_list[0][1])
                for x in sorted_list:

                    print(x, x[1], precol, str(int(x[1]) - int(precol)))
                    #confident should be above 98
                    if float(x[4]) >= 98:
                        #character between space
                        if int(int(x[1]) - int(precol)) >= ((avgWeightList * 2)+(avgWeightList/3)) :
                            outputList.append("".join(outputList2))
                            outputList2.clear()
                            outputList2.append(x[0])
                        else:
                            outputList2.append(x[0])
                        precol = int(int(x[1]))

                outputList.append("".join(outputList2))
                    # print(type(x))
                with open(inputTempFileNa + "_output.txt", "w") as f:
                    f.write(json.dumps(outputList))
                print("*****************************************************************************")
                print("")
                print("Final output :",outputList, json.dumps(outputList))
                print("")
                print("*****************************************************************************")

                
                with open(inputTempFileNa + "_done.txt", "w") as f:
                    f.write("success")
            else:
                print("This is not an trained cheque")
                with open(inputTempFileNa + "_output.txt", "w") as f:
                    f.write("This is not an trained cheque")
                with open(inputTempFileNa + "_done.txt", "w") as f:
                    f.write("success")
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Exception occured ", inputTempFileNa, error, type(error).__name__, exc_type, exc_obj, exc_tb, exc_tb.tb_lineno)
            with open(inputTempFileNa + "_output.txt", "w") as f:
                f.write("error: unable to process the file")
            if DassEnv != "dev":
                os.remove(inputTempFileNa)
            with open(inputTempFileNa + "_done.txt", "w") as f:
                f.write("success")


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
        resultcalRemove = []
        for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w > 35 and h > 75 and y >2200 and x > 100:
                        
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

                       
                        # # im_gray = cv2.imread(roi2, cv2.IMREAD_GRAYSCALE)
                        # (thresh, im_bw) = cv2.threshold(roi2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                        # thresh = 127
                        # im_bw = cv2.threshold(roi2, thresh, 255, cv2.THRESH_BINARY)[1]

                        # stretch_near = cv2.resize(im_bw, (img_width, img_height), 
                        #             interpolation = cv2.INTER_LINEAR)
                        roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
                        

                        # Save the contour as a separate image
                        cv2.imwrite(outputContourPath + "contour_" + str(i) + "_" + str(x)  + "_" + str(y) + "_" + str(w) + "_" + str(h) + ".png", roi2)
                        resulttemp = []
                        resulttemp.append("")
                        resulttemp.append(str(x))
                        resulttemp.append(str(y))
                        resulttemp.append(outputContourPath + "contour_" + str(i) + "_" + str(x)  + "_" + str(y) + "_" + str(w) + "_" + str(h) + ".png")
                        resulttemp.append("")

                        

                        resultcalRemove.append(resulttemp)



                        # Save the contour as an image
                        # cv2.imwrite(f"contour_{i}.png", black_img)

                        # Clear the black image for the next contour
                        black_img = np.zeros_like(img)
                        i += 1
        
        lst = os.listdir(outputContourPath)
        if len(lst) > 0:

            #below code is to remove unwanted contour and keep single line alone
            filtered_list = list(filter(lambda row: int(row[2]) >= (int(resultcalRemove[0][2]) - 150), resultcalRemove))
            sorted_list = sorted(filtered_list, key=lambda x: int(x[1]))  # Sort by second column
            outputList = []
            outputList2 = []
            precol = int(sorted_list[0][1])
            # print(sorted_list)
        
            for x in sorted_list:
                if int(int(x[1]) - int(precol)) >=150 :
                    outputList.append("".join(outputList2))
                    outputList2.clear()
                    outputList2.append(x[3])
                else:
                    outputList2.append(x[3])
                precol = int(int(x[1]))

            outputList.append("".join(outputList2))
            # print("Dasssssssssssssssssssssssssssssss", outputList)

            
            for x in os.listdir(outputContourPath):
                availableFlag = False
                if x.endswith(".png"):
                    availableFlag = False
                    for y in outputList:
                        if x in y:
                            availableFlag = True
                if availableFlag == False:
                    # print("GGGGGGGGGGGG", availableFlag, x)
                    os.remove(outputContourPath + x)


            #delete if less then 15 chars
            lst = os.listdir(outputContourPath)
            # print("FFFFFFFFFFFFFFFFFFFFFFF", len(lst))
            if len(lst) < 15:
                for x in os.listdir(outputContourPath):
                    os.remove(outputContourPath + x)



            # print(outputList)

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

def readInputImage2():
    images2 = sorted(list(map(str, list(data_dir2.glob("*.png")))))
   
    characters = sorted(list(['0','1','2','3','4','5','6','7','8','9','l','g','+','x']))

    print("Number of images found: ", len(images2))

    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)

    # Maximum length of any captcha in the dataset
    max_length = len(images2)

    # Mapping characters to integers
    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )


    x_valid = np.array(images2)
    return x_valid, num_to_char, images2, len(images2)

def readInputImage2a():
    images2 = sorted(list(map(str, list(data_dir2.glob("contour_2_*.png")))))
   
    characters = sorted(list(['0','1','2','3','4','5','6','7','8','9','l','g','+','x']))

    print("Number of images found: ", len(images2))

    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)

    # Maximum length of any captcha in the dataset
    max_length = len(images2)

    # Mapping characters to integers
    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )


    x_valid = np.array(images2)
    return x_valid, num_to_char, images2, len(images2)



def readInputImage():
    images2 = sorted(list(map(str, list(data_dir2.glob("*.png")))))
    # images2 = ['pred/sample4.png']

    # for img in images2:
    #     im_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #     (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #     thresh = 127
    #     im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

    #     stretch_near = cv2.resize(im_bw, (img_width, img_height), 
    #                 interpolation = cv2.INTER_LINEAR)
    #     # stretch_near = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)
    #     cv2.imwrite(img, stretch_near)
    # images2= ['dasstesting.png']
    # labels = [img.split(os.path.sep)[-1].split("_")[0] for img in images]

    # characters = set(char for label in labels for char in label)
    characters = sorted(list(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

    print("Number of images found: ", len(images2))

    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)



    # Maximum length of any captcha in the dataset
    max_length = len(images2)

    # Mapping characters to integers
    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )


    x_valid = np.array(images2)
    return x_valid, num_to_char, images2, len(images2)
# y_valid = np.array(labels[0:1])

def encode_single_sample(img_path):
    print("Dass Error testing", str(img_path))
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    # label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img}




def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon())
    # print(y_pred)
    input_length = tf.cast(input_length, tf.int32)

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    # print("decoded_dense, log_prob", decoded_dense, log_prob)
    return (decoded_dense, log_prob)

# A utility function to decode the output of the network
def decode_batch_predictions(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    # results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    results, logs = ctc_decode(pred, input_length=input_len, greedy=True)
    results = results[0][:, :1]
    # Iterate over the results and get back the text
    # print("results", results, logs)
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text, logs



def load_model(x_valid, num_to_char, images2, batch_size, modelCheque):
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid))
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # model = keras.models.load_model("P:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodel")
    # print("DDDDDDDDDDDDD", DassEnv,DassEnv == "dev")
    

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        modelCheque.input[0], modelCheque.get_layer(name="dense2").output
    )
    prediction_model.summary()
    return validation_dataset, prediction_model



def model_pred(validation_dataset, prediction_model, num_to_char, images2):
    # print(type(validation_dataset))
    #  Let's check results on some validation samples
    resultcal = []
    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        # batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
        # print("preds",preds)
        indexes = tf.argmax(preds, axis=1)
        # print("pred value", indexes) # prints [0 2 2]
        pred_texts, logs = decode_batch_predictions(preds, num_to_char)
        # print("pred_texts", pred_texts, images2)
        for x in range(0, len(pred_texts)):
            resulttemp = []
            resulttemp.append(pred_texts[x])
            resulttemp.append(images2[x].split("_")[2])
            resulttemp.append(images2[x].split("_")[3])
            resulttemp.append(images2[x])
            resulttemp.append(str(100 - (logs[x].numpy() * 100)).replace("[","").replace("]",""))

            resultcal.append(resulttemp)
            # print("Result : ", pred_texts[x],images2[x], 100 - (logs[x].numpy() * 100))
    
    
    return resultcal

def group_y(temp_data):

    ndata = [[td, td] for td in temp_data]
    data = np.array(ndata)

    # clustering
    thresh = (40.0/100.0) * (max(temp_data) - min(temp_data))  #Threshold 11% of the total range of data

    clusters = hcluster.fclusterdata(data, thresh, criterion="distance")

    total_clusters = max(clusters)

    clustered_index = []
    for i in range(total_clusters):
        clustered_index.append([])

    for i in range(len(clusters)):
        clustered_index[clusters[i] - 1].append(i)

    clustered_range = []
    for x in clustered_index:
        clustered_index_x = [temp_data[y] for y in x]
        clustered_range.append((min(clustered_index_x) , max(clustered_index_x)))

    # print(clustered_range[0])
    return clustered_range[0]
    # print(resultcal)
#finding cluster of y
# temp_y = []
# for x in resultcal:
#     print("Dass", x)
#     temp_y.append(int((x[1].split("\\")[-1]).split("_")[1]))
#     print((x[1].split("\\")[-1]).split("_")[1])

# min_y, max_y = group_y(temp_y)


# print(min_y, max_y)

def load_mod(modelCheque, modelCheque2, model_mode):
    if model_mode == 1:
        x_valid, num_to_char, images2, batch_size = readInputImage2a()
        validation_dataset, prediction_model = load_model(x_valid, num_to_char, images2, batch_size, modelCheque2)
    elif model_mode == 2:
        x_valid, num_to_char, images2, batch_size = readInputImage()
        validation_dataset, prediction_model = load_model(x_valid, num_to_char, images2, batch_size, modelCheque)
    elif model_mode == 3:
        x_valid, num_to_char, images2, batch_size = readInputImage2()
        validation_dataset, prediction_model = load_model(x_valid, num_to_char, images2, batch_size, modelCheque2)
    else:
        x_valid, num_to_char, images2, batch_size = readInputImage()
        validation_dataset, prediction_model = load_model(x_valid, num_to_char, images2, batch_size, modelCheque)

    return validation_dataset, prediction_model, num_to_char,images2
# validation_dataset, prediction_model, num_to_char, images2 = load_mod()
# model_pred(validation_dataset, prediction_model,num_to_char, images2)

if DassEnv == "dev":
    CheqeuMain(keras.models.load_model("P:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodel"), keras.models.load_model("P:\\Application\\SCEPOC\\PythonAPI\\py\\dassmodelfont2"))