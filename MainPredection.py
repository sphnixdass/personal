
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageOps
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
import keras
from keras import layers
# import numpy
import scipy.cluster.hierarchy as hcluster


import shutil

img_width = 60
img_height = 60
# generate_sample()
# Path to the data directory
# data_dir = Path("./img/")
data_dir2 = Path("S:\\InnovationDevelopment\\PythonImgReader\\InputCharecter\\")


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

# print(data_dir.glob("./**/*.png"))
# Get list of all the images
# images = sorted(list(map(str, list(data_dir.glob("./**/*.png")))))

def readInputImage():
    images2 = sorted(list(map(str, list(data_dir2.glob("*.png")))))
    # images2 = ['pred/sample4.png']

    for img in images2:
        im_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        thresh = 127
        im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

        stretch_near = cv2.resize(im_bw, (img_width, img_height), 
                    interpolation = cv2.INTER_LINEAR)
        # stretch_near = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img, stretch_near)
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



def load_model(x_valid, num_to_char, images2, batch_size):
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid))
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    model = keras.models.load_model("S:\\InnovationDevelopment\\ChequeSamples\\dassmodel")

    # Get the prediction model by extracting layers till the output layer
    prediction_model = keras.models.Model(
        model.input[0], model.get_layer(name="dense2").output
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

def load_mod():
    x_valid, num_to_char, images2, batch_size = readInputImage()

    validation_dataset, prediction_model = load_model(x_valid, num_to_char, images2, batch_size)
    return validation_dataset, prediction_model, num_to_char,images2
# validation_dataset, prediction_model, num_to_char, images2 = load_mod()
# model_pred(validation_dataset, prediction_model,num_to_char, images2)
