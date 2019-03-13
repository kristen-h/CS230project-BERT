#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:32:28 2019

@author: kristen
"""

import numpy as np, pandas as pd, datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Input, Dropout, LSTM
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.callbacks import CSVLogger
from keras.callbacks import Callback

# Import GloVe vectors and data
EMBEDDING_FILE='glove.6B.50d.txt'
TRAIN_DATA_FILE='train.csv'

# Set hyperparameters
embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

# Read in data and replace missing values
train = pd.read_csv(TRAIN_DATA_FILE)
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

# Turn each comment into a list of word indexes of equal length (with truncation or padding \
# as needed).
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

# Split train and test set
X_train, X_val, y_train, y_val = train_test_split(X_t, y, train_size=0.9, random_state=233)

# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

# Create embedding matrix with GloVe word vectors
# with random initialization for words that aren't in GloVe. 
# Use same mean and stdev of embeddings the GloVe has when generating the random init.
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    
# LSTM model
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# logger
now = datetime.datetime.now()
log_filename = str('log_glove_lstm')+now.strftime("%Y-%m-%d-%H%M")+str('.csv')
csv_logger = CSVLogger(log_filename, append=True, separator='|')

# fit the model, make prediction
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val), callbacks=[csv_logger])

y_pred = model.predict(X_val)

# total AUC
y_true = y_val
AUC = sklearn.metrics.roc_auc_score(y_true, y_pred)
print(AUC)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
print(roc_auc)


AUC1 = str(AUC)
roc_auc1 = str(roc_auc)

file = open('/home/ec2-user/glove_lstm/glove_lstm__results.txt','w') 

file.write(AUC1)
file.write(roc_auc1)

file.close()








