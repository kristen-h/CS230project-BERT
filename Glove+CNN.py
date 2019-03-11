
# coding: utf-8

# # CNN with GloVe embeddings

# Import libraries and check correct execution path
# 

# In[18]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding, Input
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import sklearn.metrics
from keras import initializers, regularizers, constraints, optimizers, layers


# Import GloVe vectors and data

# In[6]:


#path = '../input/'
#comp = 'jigsaw-toxic-comment-classification-challenge/'
EMBEDDING_FILE='glove.6B.50d.txt'
TRAIN_DATA_FILE='train.csv'
TEST_DATA_FILE='test.csv'
TEST_LABEL = 'test_labels.csv'


# Set hyperparameters

# In[7]:


embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use


# Read in data and replace missing values:

# In[8]:


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)
test_label = pd.read_csv(TEST_LABEL)


#train_x
list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#train_y
y = train[list_classes].values

#test_x
list_sentences_test = test["comment_text"].fillna("_na_").values


# Turn each comment into a list of word indexes of equal length (with truncation or padding as needed).

# In[9]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[10]:


X_train, X_val, y_train, y_val = train_test_split(X_t, y, train_size=0.9, random_state=233)


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.

# In[11]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# Create embedding matrix with GloVe word vectors, with random initialization for words that aren't in GloVe. Use same mean and stdev of embeddings the GloVe has when generating the random init.

# In[12]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[13]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[14]:


embedding_matrix.shape


# CNN model

# In[19]:


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


# Fit the model.

# In[20]:


model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_val, y_val))


# In[21]:


model.save('Glove_lstm')


# Calculate AUC

# In[22]:


y_pred = model.predict(X_val)


# In[23]:


# total AUC
y_true = y_val
AUC = sklearn.metrics.roc_auc_score(y_true, y_pred)
print(AUC)


# In[24]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[25]:


print(roc_auc)

