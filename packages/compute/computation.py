#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy sklearn nltk tqdm sklearn xgboost')


# In[2]:


import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import *
from sklearn.naive_bayes import *


# In[3]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[4]:


train_data.info()


# In[5]:


train_data.head()


# In[6]:


train_data.isnull().sum()


# In[19]:


def clean_text(text):
    clean_data = text.lower()
    clean_data = clean_data.replace(r"http.*", " ")
    clean_data = clean_data.replace(r"<.*?>", "")
    clean_data = clean_data.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    return clean_data
train_data["clean_data"] = train_data["text"].apply(clean_text)
test_data["clean_data"] = test_data["text"].apply(clean_text)


# In[20]:



stemmer = nltk.stem.PorterStemmer()
train_data["clean_data"] = train_data["clean_data"].apply(stemmer.stem)
test_data["clean_data"] = test_data["clean_data"].apply(stemmer.stem)
lemmatizer=nltk.stem.WordNetLemmatizer()
train_data["clean_data"] = train_data["clean_data"].apply(lemmatizer.lemmatize)
test_data["clean_data"] = test_data["clean_data"].apply(stemmer.stem)


# In[21]:


tokenizer = RegexpTokenizer(r'\w+')
train_data["token_data"] = train_data["clean_data"].apply(tokenizer.tokenize)
test_data["token_data"] = test_data["clean_data"].apply(tokenizer.tokenize)


# In[22]:


def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words
train_data["token_data"] = train_data["token_data"].apply(remove_stopwords)
test_data["token_data"] = test_data["token_data"].apply(remove_stopwords)


# In[23]:


def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

train_data['preprocessed_text'] = train_data['token_data'].apply(lambda x : combine_text(x))
test_data['preprocessed_text'] = test_data['token_data'].apply(lambda x : combine_text(x))
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train_data['preprocessed_text'])
test_tfidf = tfidf.transform(test_data["preprocessed_text"])


# In[31]:


count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_data['preprocessed_text'])
test_vectors = count_vectorizer.transform(test_data['preprocessed_text'])


# In[25]:


# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
scores = cross_val_score(clf, train_vectors, train_data["target"], cv=5, scoring="f1")
scores


# In[26]:


# Fitting a simple Logistic Regression on TFIDF
clf_tfidf = LogisticRegression(C=1.0)
scores = cross_val_score(clf_tfidf, train_tfidf, train_data["target"], cv=5, scoring="f1")
scores


# In[27]:


# Fitting a simple Naive Bayes on Counts
clf_NB = BernoulliNB()
scores = cross_val_score(clf_NB, train_vectors, train_data["target"], cv=5, scoring="f1")
scores


# In[28]:


# Fitting a simple Naive Bayes on TFIDF
clf_NB_TFIDF = BernoulliNB()
scores = cross_val_score(clf_NB_TFIDF, train_tfidf, train_data["target"], cv=5, scoring="f1")
scores


# In[29]:


clf_NB.fit(train_vectors, train_data["target"])


# In[32]:


submission = pd.DataFrame()
submission['id']=test_data['id'].to_list()
submission['target']=clf_NB.predict(test_vectors)
submission.to_csv("submission.csv", index=False)


# In[ ]:




