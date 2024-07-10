from datetime import datetime
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import *
from keras.models import *
import json
import pandas as pd
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


for i in range(1,2):

    file_path = '../scrapperBot/database/signatures/websites'+str(i)+'.json'
    with open(file_path, 'r') as file:
        # Load JSON data from the file
      
        for i , row in enumerate(json.load(file)):
            
            # json_data[i]['signatures'] = ' '.join(row['signatures'])
            current_classification = row['classification']
            if not row['certs'] is None:
                if current_classification == 0 and countLegit < countPhishing:
                    date_format = "%b %d %H:%M:%S %Y GMT"
                    # Parse the dates
                    not_before = datetime.strptime(row['certs']["notBefore"], date_format)
                    not_after = datetime.strptime(row['certs']["notAfter"], date_format)
                    time_difference = not_after - not_before
                    del row['certs']["notBefore"]
                    del row['certs']["notAfter"]
                    row['certs']["time_dif"] = time_difference.days
                    colums_text.append(row['certs'])
                    train_labels.append(row['classification'])
                    countLegit += 1
                    
                elif current_classification == 1:
                    
                    date_format = "%b %d %H:%M:%S %Y GMT"
                    # Parse the dates
                    not_before = datetime.strptime(row['certs']["notBefore"], date_format)
                    not_after = datetime.strptime(row['certs']["notAfter"], date_format)
                    time_difference = not_after - not_before
                    del row['certs']["notBefore"]
                    del row['certs']["notAfter"]
                    row['certs']["time_dif"] = time_difference.days
                    colums_text.append(row['certs'])
                    train_labels.append(row['classification'])
                    countPhishing += 1

print("Legit : " , countLegit, " Phishing : " , countPhishing )

train_labels = pd.Series(train_labels)
colums_text = pd.DataFrame(colums_text)
ssl_c =colums_text["C"]
ssl_o =colums_text["O"]
ssl_cn =colums_text["CN"]
ssl_time_dif =colums_text["time_dif"]

# ------------------------------------------ tokenization -------------------------------------

# ssl_c_tokenizer = keras.preprocessing.text.Tokenizer(num_words=200, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ '
#                                                         , lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
# ssl_c_tokenizer.fit_on_texts(ssl_c)
# with open('ssl_c_tokenizer.pkl', 'wb') as f:
#     pickle.dump(ssl_c_tokenizer, f)
# ssl_c_t = ssl_c_tokenizer.texts_to_sequences(ssl_c)
# colums_text["C"] = ssl_c_t

ssl_c_tokenizer = {sentence: i+1 for i, sentence in enumerate(set(ssl_c))}
with open('ssl_c_tokenizer.json', 'w') as file:
    json.dump(ssl_c_tokenizer, file)
ssl_c_t = [ssl_c_tokenizer[sentence] for sentence in ssl_c]
colums_text["C"] = ssl_c_t

# ssl_o_tokenizer = keras.preprocessing.text.Tokenizer(num_words=200
#                                                         , lower=True, char_level=False, oov_token=None, document_count=0)
# ssl_o_tokenizer.fit_on_texts(ssl_o)
# with open('ssl_o_tokenizer.pkl', 'wb') as f:
#     pickle.dump(ssl_o_tokenizer, f)
# ssl_o_t = ssl_o_tokenizer.texts_to_sequences(ssl_o)
# colums_text["O"] = ssl_o_t

ssl_o_tokenizer = {sentence: i+1 for i, sentence in enumerate(set(ssl_o))}
with open('ssl_o_tokenizer.json', 'w') as file:
    json.dump(ssl_o_tokenizer, file)
ssl_o_t = [ssl_o_tokenizer[sentence] for sentence in ssl_o]
colums_text["O"] = ssl_o_t

# ssl_cn_tokenizer = keras.preprocessing.text.Tokenizer(num_words=200
#                                                         , lower=True, char_level=False, oov_token=None, document_count=0)
# ssl_cn_tokenizer.fit_on_texts(ssl_cn)
# with open('ssl_cn_tokenizer.pkl', 'wb') as f:
#     pickle.dump(ssl_cn_tokenizer, f)
# ssl_cn_t = ssl_cn_tokenizer.texts_to_sequences(ssl_cn)
# colums_text["CN"] = ssl_cn_t

ssl_cn_tokenizer = {sentence: i+1 for i, sentence in enumerate(set(ssl_cn))}
with open('ssl_cn_tokenizer.json', 'w') as file:
    json.dump(ssl_cn_tokenizer, file)
ssl_cn_t = [ssl_cn_tokenizer[sentence] for sentence in ssl_cn]
colums_text["CN"] = ssl_cn_t

print(colums_text.apply(lambda row: row.values.flatten().tolist(), axis=1).head())
x_train, x_val, y_train, y_val = train_test_split(colums_text, train_labels, test_size=0.3, random_state=24)

## ------------------------------------------ cnn -------------------------------------
# model2 = keras.Sequential()
# model2.add(keras.layers.Embedding(input_dim=size_of_vocabulary, output_dim=128, input_length=max_len))
# model2.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
# model2.add(keras.layers.LeakyReLU(0.1))
# model2.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
# model2.add(keras.layers.LeakyReLU(0.1))
# model2.add(keras.layers.GlobalAveragePooling1D())
# model2.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model2.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model2.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# model2.compile(optimizer='adam',
#                         loss='binary_crossentropy',
#                         metrics=['accuracy'])
# model2.summary()
# model2_history = model2.fit(  x_train,
#             y_train,
#             epochs=100,
#             batch_size=500,
#             validation_data=(x_val, y_val),
#             verbose=1)
# model2.save('text_classification_cnn.h5')

# # Extract training history
# train_loss = model2_history.history['loss']
# val_loss = model2_history.history['val_loss']
# train_accuracy = model2_history.history['accuracy']
# val_accuracy = model2_history.history['val_accuracy']
# epochs = range(1, len(train_loss) + 1)

# # Plot training and validation loss
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('CNN Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Plot training and validation accuracy
# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
# plt.title('CNN Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()
# # ------------------------------------------ attention bilstm ------------------------------------- 

# class attention(Layer):
#     def __init__(self, return_sequences=False):
#         self.return_sequences = return_sequences

#         super(attention,self).__init__()

#     def build(self, input_shape):
#         self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
#                                initializer="normal")
#         self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
#                                initializer="normal")
#         self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
#                                initializer="normal")
#         self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
#                                initializer="normal")

#         super(attention,self).build(input_shape)

#     def call(self, x):
#         e = K.tanh(K.dot(x,self.W)+self.b)
#         a = K.softmax(e, axis=1)
#         output = x*a
#         if self.return_sequences:

#             return output
#         return K.sum(output, axis=1)

# model3 = Sequential()
# model3.add(Embedding(input_dim=size_of_vocabulary, output_dim=128, input_length=max_len))
# model3.add(Bidirectional(LSTM(64, return_sequences=True)))
# model3.add(attention(return_sequences=False)) # receive 3D and output 3D
# # model3.add(Dropout(0.5))
# model3.add(Dense(1, activation='sigmoid'))
# model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# model3.summary()
# history_model3=model3.fit(x_train, y_train,
#            batch_size=500,
#            epochs=40,
#            validation_data=[x_val, y_val])

# model3.save('text_classification_biLSTM_attention.h5')

# # Extract training history
# train_loss = history_model3.history['loss']
# val_loss = history_model3.history['val_loss']
# train_accuracy = history_model3.history['accuracy']
# val_accuracy = history_model3.history['val_accuracy']
# epochs = range(1, len(train_loss) + 1)

# # Plot training and validation loss
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('BILSTM Attention Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Plot training and validation accuracy
# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
# plt.title('BILSTM Attention Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()