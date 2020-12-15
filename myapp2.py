# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:13:25 2020

@author: admin
"""

import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
import pandas
import h5py
import pathlib

#import tensorflow as tf
import tensorflow as tf
import keras

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers.merge import add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout

from tqdm.notebook import tqdm as tqdm
tqdm().pandas()
#model = load_model('C:/Users/admin/aegis/Capstone_project/streamlit/project/models/model_9.h5')
#dataset_text = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_text"
#filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

#load the data 
#filename = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_text/Flickr8k.token.txt"
filename = "Flickr8k.token.txt"
# Loading a text file into memory

def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#filename = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_text/Flickr_8k.trainImages.txt"
#filename = "Flickr_8k.trainImages.txt"
#def load_photos(filename):
#    file = load_doc(filename)
#    photos = file.split("\n")[:-1]
#    return photos
#train_imgs = load_photos(filename)

#train_imgs
#from zipfile import ZipFile 
#file_name = "Flickr8k_images.zip" 
#with ZipFile(file_name, 'r') as zip:
#    img_path=zip.read() 
#img_path = open('Flickr8k_images.zip')
#img_path = "https://raw.githubusercontent.com/gulfairus/project2/tree/master/Flicker8k_Dataset"
#img_path = "C:/Users/admin/aegis/Capstone_project/streamlit/project/Flickr8k_Dataset/Flicker8k_Dataset"
img_path = "Flicker8k_Dataset"
train_imgs=[]
for img in tqdm(os.listdir(img_path)):
    train_imgs.append(img)



import pandas as pd
import streamlit as st
#import plotly.express as px
def dict_pics(train_imgs):
    pics={}
    for i in range(len(train_imgs)):
        #    pics[] = {train_imgs[i]: '/'.join([img_path, train_imgs[i]])}
        pics[train_imgs[i]] = '/'.join([img_path, train_imgs[i]])
    return pics
pics = dict_pics(train_imgs)

pic = st.selectbox("Picture choices", list(pics.keys()), 0)


import numpy as np
from PIL import Image
    #import matplotlib.pyplot as plt
    #import argparse

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
            return None
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text
#
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))


#model = load_model('C:/Users/admin/aegis/Capstone_project/streamlit/project/models/model_9.h5')
#images = r"C:\Users\Piyush Mishra\Desktop\AI_Assignment\Flicker8k_Dataset/"
#model = load_model(r'C:/Users/admin/aegis/Capstone_project/streamlit/project/models/model_9.h5')
# Convert the model.
#model_keras=tf.keras.models.load_model('model_9.h5')

#converter = tf.lite.TFLiteConverter.from_keras_model('model_9.h5')
#tflite_model = converter.convert()
# Save the model.
#with open('model.tflite', 'wb') as f:
#  f.write(tflite_model)
#model = load_model('model.tflite')

#converter = tf.lite.TFLiteConverter.from_saved_model("model_9.h5") # path to the SavedModel directory
#tflite_model = converter.convert()
# Save the model.
#with open('model.tflite', 'wb') as f:
#  f.write(tflite_model)
#model = load_model('model.tflite')

#import urllib.request
#link = "https://drive.google.com/file/d/17SVXrmUW-dINgipPR3hqrl4M8bmIESaW/view?usp=sharing"
#f = urllib.request.urlopen(link)
#myfile = f.read()
#model = load_model(myfile)
# Save tf.keras model in H5 format.
#keras_file = 'model9.h5'
#tf.keras.models.save_model(myfile, keras_file)
#model = load_model(link)

#import os 
#os.system('model.py')

#link = r'C:/Users/admin/aegis/Capstone_project/streamlit/project/models/model_9.h5'
#model_dir = pathlib.Path(link)
#import boto3
#model = load_model(r'https://project2model.s3.us-east-2.amazonaws.com/model_9.h5')

import boto3
client = boto3.client('s3', 
                      aws_access_key_id='AKIAJGBGWMQVODVC3UGQ',
                      aws_secret_access_key='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')
client.download_file('project2model',
                     'model_9.h5',
                     'model_9.h5')
model = load_model('model_9.h5')

xception_model = Xception(include_top=False, pooling="avg")


st.write("""
# Image Captioning Prediction
""")
st.write("This is a simple image captioning web app to predict caption of the image")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

photo = extract_features(file, xception_model)
#photo = extract_features(pics[pic], xception_model)
#    img = Image.open(pics[pic])
description = generate_desc(model, tokenizer, photo, max_length)
#st.image(pics[pic], use_column_width=True, caption=0)
#st.image(pics[pic], use_column_width=True, caption=description)
st.image(file, use_column_width=True, caption=description)
