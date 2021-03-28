"""
POetry Generation using LSTM deep neural network
"""

import requests
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten, Embedding, Dropout

from sklearn.model_selection import train_test_split
from wordcloud import WordCloud as ws
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

url = 'https://raw.githubusercontent.com/laxmimerit/poetry-data/master/adele.txt'

poem = requests.get(url)

poetry = poem.text.splitlines()

token = Tokenizer()

len(' '.join(poetry))

copy_ = np.copy(poetry)

## Word visualization (most occuring words)

from matplotlib import pyplot as plt
wc = ws(width = 800, height = 400).generate(str(copy_))
plt.imshow(wc)
plt.axis('off')
plt.show()


token.fit_on_texts(poetry)

encoder_txt = token.texts_to_sequences(poetry)

vocab_size = len(token.word_counts) + 1

## PREPARING TRAINING DATA

arr_of_stanza = []
for i in encoder_txt:
    if len(i)>1:
        for d in range(2,len(i)):
            arr_of_stanza.append(i[:d])
            
## Padding

max_len = 20
seq =  pad_sequences(arr_of_stanza, maxlen = max_len, padding = 'pre')

x = seq[:,:-1]
y = seq[:,-1]

print(x.shape,y.shape)

## Convert (y) to one hot encoder

y = np_utils.to_categorical(y, num_classes = vocab_size)

## MODEL TRAINING

seque_length = x.shape[1]

model =  Sequential()

model.add(Embedding(vocab_size, 50, input_length=seque_length))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation ='relu'))
model.add(Dense(vocab_size, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adam', loss= "categorical_crossentropy", metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor = 'val_loss', patience =  2, restore_best_weights = True, verbose = 1)
mc = ModelCheckpoint('best_model.h5', monitor = "val_acc", mode = "max",save_best_only = True, verbose = 1)

model.fit(x, y, batch_size = 32, epochs = 120, callbacks = (early_stopping,mc))

model.evaluate(x,y)
## poetry generation

poetry_length = 10

def generate_poem(word_text,n_lines):
    for i in  range(n_lines):
        text = []
        for _ in range(poetry_length):
            encode = token.texts_to_sequences([word_text])
            encode = pad_sequences(encode,maxlen = max_len, padding = 'pre')
            y_pred = np.argmax(model.predict(encode),axis = 1)
            
            predicted_word = ""
            for word,index in token.word_index.items():
                if index == y_pred:
                    predicted_word = word
                    break
            Word_text = word_text + ' ' + predicted_word
            text.append(predicted_word)
        Word_text = text[-1]
        text = " ".join(text)
        print(text)



Word_text = 'love'
(generate_poem(word_text,5))
