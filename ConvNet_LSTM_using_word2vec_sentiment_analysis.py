import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import Word2Vec
import gensim
import nltk
from random import shuffle
import zipfile
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', -1)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('display.max_colwidth', -1)

import theano
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras import backend as K
from keras import layers
from keras.layers import BatchNormalization, Flatten,Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional,LSTM, Input,Dropout, Add,concatenate, Dense, Activation, ZeroPadding2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical


train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')
sub = pd.read_csv('sample_submission_gfvA5FD.csv')
total = train.append(test, ignore_index=True)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return(input_txt)

total['tidy_tweet'] = np.vectorize(remove_pattern)(total['tweet'], "@[\w]*")
total['tidy_tweet'] = total['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
total['tidy_tweet'] = total['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
tokenized_tweet = total['tidy_tweet'].apply(lambda x: x.split())

from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
total['tidy_tweet'] = tokenized_tweet


total.tidy_tweet.fillna('',inplace=True)
t = total['tidy_tweet'].apply(lambda x: x.split())

mod = Word2Vec(t)  #training word2vec model

def f1(y_true, y_pred):       #f1 score metric
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives+true_positives ) 
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives+true_positives )  
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall ))  

def sentences_to_indices(text , mod, max_len):
    m = len(text)                                 
    text_indices = np.zeros((m, max_len))
    
    for i in range(m):                      
        j=0
        for w in text[i]:
            if j==max_len:
                break
            if w not in mod.wv.vocab:
                continue
            text_indices[i, j] = mod.wv.vocab[w].index  # Set the (i,j)th entry of X_indices to the index of the correct word.
            j = j + 1
            
    return text_indices


def pretrained_embedding_layer(mod):
    vocab_len = len(mod.wv.vocab) + 1                 
    
    emb_dim = mod["father"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    index=0
    for word in mod.wv.vocab:
        emb_matrix[index, :] = mod[word]
        index+=1
        
    embedding_layer = Embedding(vocab_len, emb_dim)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def MODEL(input_shape,mod):
    
    sentence_indices = Input(shape=input_shape)
    embedding_layer =  pretrained_embedding_layer(mod)
    embeddings = embedding_layer(sentence_indices)   
    
    bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(embeddings)
    #bigram_branch = GlobalMaxPooling1D()(bigram_branch)
    bigram_branch = MaxPooling1D(pool_size=2)(bigram_branch)
    trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(embeddings)
    #trigram_branch = GlobalMaxPooling1D()(trigram_branch)
    trigram_branch = MaxPooling1D(pool_size=2)(trigram_branch)
    fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(embeddings)
    #fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
    fourgram_branch = MaxPooling1D(pool_size=2)(fourgram_branch)
    
    merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)
    
    X = Bidirectional(LSTM(100))(merged)
    
    X = Dense(256,activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(2,activation='sigmoid')(X)
    
    model = Model (inputs = sentence_indices , outputs= X, name= 'MODEL')
    
    return(model)
    
max_len=20
model = MODEL((max_len,),mod)
#print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer= 'adam', metrics=[f1])

x_total=total['tidy_tweet'].apply(lambda x: x.split())
y_total=total['label'].values
x_total=x_total.values

x_train = x_total[:31962]
x_test = x_total[31962:]
y_train = y_total[:31962]

x_train_indices = sentences_to_indices(x_train, mod,max_len)
x_test_indices=sentences_to_indices(x_test,mod,max_len)
y_train_ohe=to_categorical(y_train, num_classes=2)


model.fit(x_train_indices, y_train_ohe, epochs = 5, batch_size = 32, shuffle=True)

filename = 'model_cnn_ngram234_maxlen20_biLSTM.sav'
pickle.dump(model, open(filename, 'wb'))

prediction = model.predict(x_test_indices)
#plt.hist(prediction[:,1],bins=10)

prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
sub = pd.read_csv('sample_submission_gfvA5FD.csv')
sub['label']=prediction_int
sub.to_csv('word2vec_cnn.csv',index=False)
