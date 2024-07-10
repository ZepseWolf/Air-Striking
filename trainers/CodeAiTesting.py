import fasttext
import numpy as np
import tensorflow as tf
import keras
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from keras import backend as K
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt

train_labels = []
code_array = []
countPhishing=0
countLegit=0

num_samples = 1000
vocab_size = 20000
max_sequence_length = 20
embedding_dim = 50
max_len = 200

embedding_model = fasttext.load_model('code_embedding_model.bin')

def remove_surrogates(input_string):
    # Filter out characters that fall within the surrogate ranges
    return ''.join(char for char in input_string if not (0xD800 <= ord(char) <= 0xDFFF))

def count_words(text):
    return len(text.split())

def pad_array(arr, length):
    return np.pad(arr, (0, length - len(arr)), mode='constant')

def manual_embedding (current_website_in_word):
    tempArr = []
    if len(current_website_in_word) < max_len:
        for x in range(max_len):
        # for word in current_website_in_word:
            # Padding with values
            if x < len(current_website_in_word):
                phase = remove_surrogates(current_website_in_word[x]).replace("\n"," ")
                tempArr.append(embedding_model.get_sentence_vector(phase))
            else:
                tempArr.append([0] * 128)
        tempArr = pad_array(tempArr, max_len)

        
    else:
        for x in range(max_len):
            # Fair sampling in range
            gap = len(current_website_in_word)//max_len
            phase = remove_surrogates(" ".join(current_website_in_word[x*gap:x*gap+gap ]).replace("\n"," "))
            # print(phase)
            tempArr.append(embedding_model.get_sentence_vector(phase))
    return tempArr

# Analytics
def getCorrectlyIdentifiedArr(model, x_val, y_val):
    y_pred = model.predict(x_val)
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int).flatten()
    false_negatives = np.where((y_val == 1) & (y_pred_binary == 0))[0]

    # Find correctly identified samples
    correctly_identified = np.where(y_val == y_pred_binary)[0]

    print("False Negatives %:", len(false_negatives)/len(y_pred))
    print("Correctly Identified (indices):", correctly_identified)
    return correctly_identified

with open('code_cleaned_variable.pkl', 'rb') as file:
    data = pickle.load(file)

x_train, x_val, y_train, y_val = train_test_split(data['code_array'], data['train_labels'], test_size=0.3, random_state=24)

## ------------------------------------------ cnn -------------------------------------
model2 = tf.keras.models.load_model('code_classification_cnn3.h5')
getCorrectlyIdentifiedArr(model2, x_val, y_val)
# ------------------------------------------ attention bilstm -------------------------------------

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")

        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)


model3 = tf.keras.models.load_model('code_classification_biLSTM_attention3.h5', custom_objects = {'attention': attention})
getCorrectlyIdentifiedArr(model3, x_val, y_val)
