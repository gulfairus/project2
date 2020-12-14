# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:55:04 2020

@author: admin
"""
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
import pandas

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

model = load_model('model_9.h5')