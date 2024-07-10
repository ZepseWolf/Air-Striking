import fasttext
import numpy as np
import json
import pickle

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt

train_labels = []
code_array = []
index_array = []
content_array = []
countPhishing=0
countLegit=0

num_samples = 1000
vocab_size = 20000
max_sequence_length = 20
embedding_dim = 50
max_len = 200

# Load models
embedding_model = fasttext.load_model('code_embedding_model.bin')

# Pre processing for JS code sampling
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

def text_embedding(content_array):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ '
                                                        , lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(content_array)
    t = tokenizer.texts_to_sequences(content_array)
    tokenized_text = tf.keras.utils.pad_sequences(t, max_len, padding='post', value=0)

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenized_text

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
    
def printModelPerformance(model_history , typeOfModel):
    train_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    train_accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(typeOfModel +' Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title(typeOfModel +' Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

n = 300
for i in range(1,n):
    print(round(i/n*100 , 2),"%")
    file_path = '../scrapperBot/database/signatures/websites'+str(i)+'.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        # Load JSON data from the file
      
        for i , row in enumerate(json.load(file)):
            current_classification = row['classification']
            current_website_in_word = row['websites']
            
            if current_classification == 0 and countLegit < countPhishing:
                # Ensure both legit and malisious webiste are equal numbered
           
                tempArr = manual_embedding(current_website_in_word)
                code_array.append(tempArr)
                train_labels.append(row['classification'])
                index_array.append(row['index'])
                content_array.append(row['signatures'])
                countLegit += 1
                
            elif current_classification == 1:
                tempArr = manual_embedding(current_website_in_word)
                code_array.append(tempArr)
                train_labels.append(row['classification'])
                index_array.append(row['index'])
                content_array.append(row['signatures'])
                countPhishing += 1

train_labels = np.array(train_labels)
code_array = np.array(code_array)
content_array = np.array(content_array)

with open('refined_variables.pkl', 'wb') as file:
    pickle.dump({"train_labels" : train_labels, 
                 "code_array" : code_array,
                 "context_array" : text_embedding(content_array),
                 "index_array" : index_array
                 }, file)
