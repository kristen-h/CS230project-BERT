
# coding: utf-8

# # CNN with Word2Vec embeddings

# Import libraries and check correct execution path

# In[66]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

#import nltk
#nltk.download('punkt')

import gensim
from gensim.models import Word2Vec
from keras import optimizers
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
import gensim.models.keyedvectors as word2vec
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding, Input
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')


# read the data and replace missing values

# In[14]:


# read the data
TRAIN_DATA_FILE='train.csv'
TEST_DATA_FILE='test.csv'

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

#train_x
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#train_y
y = train[list_classes].values

#test_x
list_sentences_test = test["comment_text"].fillna("_na_").values


# In[21]:


word2vecDict = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)


# set hyperparameters

# In[24]:


embed_size = 300    # Word vector dimensionality                      
maxlen = 100 
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)


# Turn each comment into a list of word indexes of equal length (with truncation or padding as needed).

# In[25]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[26]:


X_train, X_val, y_train, y_val = train_test_split(X_t, y, train_size=0.9, random_state=233)


# In[29]:


embeddings_index = dict()
for word in word2vecDict.wv.vocab:
    embeddings_index[word] = word2vecDict.word_vec(word)
print('Loaded %s word vectors.' % len(embeddings_index))


# In[30]:


all_embs = np.stack(list(embeddings_index.values()))
emb_mean,emb_std = all_embs.mean(), all_embs.std()
nb_words = len(tokenizer.word_index)


# In[31]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[32]:


embedding_matrix.shape


# CNN model

# In[56]:


model = Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(128, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='sigmoid'))  #multi-label (k-hot encoding)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


# In[57]:


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]


# In[58]:


model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_val, y_val))


# In[59]:


model.save('w2v_cnn')


# Calculate AUC

# In[60]:


y_pred = model.predict(X_val)


# In[67]:


# total AUC
y_true = y_val
AUC = sklearn.metrics.roc_auc_score(y_true, y_pred)
print(AUC)


# In[68]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[69]:


print(roc_auc)

