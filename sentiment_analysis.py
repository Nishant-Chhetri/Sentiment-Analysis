import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('display.max_colwidth', -1)

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


#import nltk
#nltk.download('wordnet')
#from nltk.stem import WordNetLemmatizer   
#lemmatizer = WordNetLemmatizer()
#lemmatized_reviews = [' '.join([lemmatizer.lemmatize(word) for word in review]) for review in tokenized_tweet]
#total['tidy_tweet']=lemmatized_reviews

#from sklearn.feature_extraction.text import CountVectorizer
#bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=10000, stop_words='english')
#bow = bow_vectorizer.fit_transform(total['tidy_tweet'])


from sklearn.feature_extraction.text import CountVectorizer
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1,4),stop_words='english',max_df=0.9,min_df=2,max_features=None)
ng = ngram_vectorizer.fit_transform(total['tidy_tweet'])


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,ngram_range=(1,4), min_df=2, max_features=3000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(total['tidy_tweet'])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = ng[:31962,:]
test_bow = ng[31962:,:]
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)
lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) 
prediction = lreg.predict_proba(xvalid_bow) 
prediction_int = prediction[:,1] >= 0.29 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
print('logistic reg. valid score:',f1_score(yvalid, prediction_int)) 


from sklearn.svm import LinearSVC
train_bow = ng[:31962,:]
test_bow = ng[31962:,:]
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)
svm = LinearSVC(C=0.5)
svm.fit(xtrain_bow, ytrain)
prediction = svm.predict(xvalid_bow) 
print('svm validation score:',f1_score(yvalid,prediction))    

#Prediction on test set

train_bow = ng[:31962,:]
test_bow = ng[31962:,:]
lreg = LogisticRegression()
lreg.fit(train_bow, train['label']) 
prediction = lreg.predict_proba(test_bow) 
prediction_int = prediction[:,1] >= 0.29 
prediction_int = prediction_int.astype(np.int) 
sub['label']=prediction_int
#sub.to_csv('lr_ngram_pred.csv',index=False)   

svm = LinearSVC(C=0.5)
svm.fit(train_bow, train['label']) 
prediction = svm.predict(test_bow) 
sub['label']=prediction
#sub.to_csv('LinearSVC_ngram_pred.csv',index=False)  #0.7438
