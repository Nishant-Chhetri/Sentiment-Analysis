# Sentiment-Analysis

## Problem Statement

The objective of this task is to detect hate speech in tweets. 
We say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. 
So, the task is to classify racist or sexist tweets from other tweets.
Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the 
tweet is not racist/sexist, your objective is to predict the labels on the test dataset.
https://datahack.analyticsvidhya.com/contest/all/?utm_source=global-header

#### Using Logistic Regression and LinearSVC

1. Applied various data cleaning methods like removing stopwords,removing special characters and digits. 
2. Then used Port Stemmer and Lemmatizer to stem the words. Port Stemmer performed better with this dataset, therefore I used
Stemming instead of Lemmatization. 
3. Then used Bag of Words techniques using CountVectorizer to create ngrams of range (1,4).
4. After Spliting the dataset to validate 30% of data I trained the data with Logistic Regression and LinearSVC to get predictions



#### Using CNN and LSTM

1. The data cleaning procedure is similar as in Logistic Regression model to remove stopwords,special characters and digits. 
2. Used Port Stemmer to stem each word.
3. Then used Gensim word2vec model to create vector of each word in the dataset.
4. Using Keras Embeddings, embedded the vectors in place of words with maxlen=20 for each tweet. 
5. As dataset was unbalanced using accuracy won't give clear idea of performance as model could predict all values 0 and still get high accuracy, therefore built a f1 score metric. 




