import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import *
from keras.models import *
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

train_labels = []
colums_text = []
countPhishing=0
countLegit=0

num_samples = 1000
size_of_vocabulary = 20000
max_sequence_length = 20
embedding_dim = 50
max_len = 200

# Training Analytics
def getCorrectlyIdentifedArr(model , x_val, y_val):
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

with open('refined_variables.pkl', 'rb') as file:
    data = pickle.load(file)
    
# Cross validaiton setup
x_train, x_val, y_train, y_val = train_test_split(data['context_array'], data['train_labels'], test_size=0.3, random_state=24)

## ------------------------------------------ cnn -------------------------------------
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(input_dim=size_of_vocabulary, output_dim=128, input_length=max_len))
model2.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model2.add(keras.layers.LeakyReLU(0.1))
model2.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model2.add(keras.layers.LeakyReLU(0.1))
model2.add(keras.layers.GlobalAveragePooling1D())
model2.add(keras.layers.Dense(16, activation=tf.nn.relu))
model2.add(keras.layers.Dense(16, activation=tf.nn.relu))
model2.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model2.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
model2.summary()
model2_history = model2.fit( x_train,
            y_train,
            epochs=100,
            batch_size=500,
            validation_data=(x_val, y_val),
            verbose=1)

model2.save('text_classification_cnn.h5')

getCorrectlyIdentifedArr(model2 , x_val, y_val)
printModelPerformance(model2_history, "CNN")

# ------------------------------------------ attention bilstm ------------------------------------- 

class attention(Layer):
    def __init__(self, return_sequences=False):
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

model3 = Sequential()
model3.add(Embedding(input_dim=size_of_vocabulary, output_dim=128, input_length=max_len))
model3.add(Bidirectional(LSTM(64, return_sequences=True)))
model3.add(attention(return_sequences=False)) # receive 3D and output 3D
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

model3.summary()
history_model3=model3.fit(x_train, y_train,
           batch_size=500,
           epochs=40,
           validation_data=[x_val, y_val])

model3.save('text_classification_biLSTM_attention.h5')

getCorrectlyIdentifedArr(model3 , x_val, y_val)
printModelPerformance(history_model3, "BILSTM-Attention")