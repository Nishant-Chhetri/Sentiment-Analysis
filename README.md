# Sentiment-Analysis

## Problem Statement

The objective of this task is to detect hate speech in tweets. 
We say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. 
So, the task is to classify racist or sexist tweets from other tweets.
Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the 
tweet is not racist/sexist, your objective is to predict the labels on the test dataset.
https://datahack.analyticsvidhya.com/contest/all/?utm_source=global-header

#### Text Processing 

First I applied various data cleaning methods like removing stopwords,removing special characters and digits. 
Then I used Port Stemmer and Lemmatizer to stem the words. Port Stemmer performed better with this dataset, therefore I used
Stemming instead of Lemmatization. 

#### Using Logistic Regression





