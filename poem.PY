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
from keras.optimizers import Adam 

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
