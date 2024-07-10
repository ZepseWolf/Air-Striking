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

def remove_surrogates(input_string):
    # Filter out characters that fall within the surrogate ranges
    return ''.join(char for char in input_string if not (0xD800 <= ord(char) <= 0xDFFF))

def count_words(text):
    return len(text.split())

def pad_array(arr, length):
    return np.pad(arr, (0, length - len(arr)), mode='constant')

# Analytics
def getCorrectlyIdentifiedArr(model, x_val, y_val,index_arr):
    y_pred = model.predict(x_val)
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int).flatten()
    false_negatives = np.where((y_val == 1) & (y_pred_binary == 0))[0]

    # Find correctly identified samples
    correctly_identified = np.where(y_val == y_pred_binary)[0]
    correctly_identified_indices = index_arr[correctly_identified]

    print("False Negatives %:", len(false_negatives)/len(y_pred))
    print("Correctly Identified (indices):", correctly_identified_indices)
    return correctly_identified_indices

def find_subsets(A, B):
    set_A = set(A)
    set_B = set(B)

    # Elements in A but not in B
    subset_A_not_B = set_A - set_B

    # Elements in B but not in A
    subset_B_not_A = set_B - set_A

    return list(subset_A_not_B), list(subset_B_not_A)

def write_to_file(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            file.write("%s\n" % item)

with open('refined_variables.pkl', 'rb') as file:
    data = pickle.load(file)
code_x_train, code_x_val, code_y_train, code_y_val = train_test_split(data['code_array'], data['train_labels'], test_size=0.3,shuffle = False, random_state=24)
text_x_train, text_x_val, text_y_train, text_y_val = train_test_split(data['context_array'], data['train_labels'], test_size=0.3,shuffle = False, random_state=24)

index_array = np.array(data['index_array'])

code_cnn = tf.keras.models.load_model('code_classification_cnn3.h5')
code_cnn_result = getCorrectlyIdentifiedArr(code_cnn, code_x_val, code_y_val , index_array)

code_bilstm = tf.keras.models.load_model('code_classification_biLSTM_attention3.h5', custom_objects = {'attention': attention})
code_bilstm_result = getCorrectlyIdentifiedArr(code_bilstm, code_x_val, code_y_val , index_array)

text_cnn = tf.keras.models.load_model('text_classification_cnn.h5')
text_cnn_result =getCorrectlyIdentifiedArr(text_cnn , text_x_val, text_y_val, index_array)

text_bilstm = tf.keras.models.load_model('text_classification_biLSTM_attention.h5', custom_objects = {'attention': attention})
text_bilstm_result = getCorrectlyIdentifiedArr(text_bilstm , text_x_val, text_y_val, index_array)

cnn_subset_A_not_B, cnn_subset_B_not_A = find_subsets(text_cnn_result, code_cnn_result)

print("cnn text dif : ", cnn_subset_A_not_B ," cnn code dif* : ", cnn_subset_B_not_A )

bilstm_A_not_B, bilstm_subset_B_not_A = find_subsets(text_bilstm_result, code_bilstm_result)

print("bilstm text dif : ", bilstm_A_not_B ," bilstm code dif* : ", bilstm_subset_B_not_A )
# print(text_data['colums_text'][0])
# print(code_data['code_array'][0])

# outputArray = mainArray[bilstm_subset_B_not_A]