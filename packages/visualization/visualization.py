#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy seaborn matplotlib missingno wordcloud nltk')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import missingno
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


# In[114]:


train_data = pd.read_csv('notebooks/train.csv')
test_data = pd.read_csv('notebooks/test.csv')


# In[4]:


#Missing Value
missingno.bar(train_data, color="dodgerblue", sort="ascending")


# In[5]:


missingno.bar(test_data, color="dodgerblue", sort="ascending")


# In[6]:


#word cloud for keywords
all_data = pd.concat([test_data,train_data])
plt.figure(figsize=(25,12))
all_data["keyword"].value_counts(sort=True, dropna=True).nlargest(30).plot.bar()


# In[7]:


wordcloud = WordCloud(background_color='white').generate_from_frequencies(all_data["keyword"].value_counts(sort=True, dropna=True).to_dict())
# plot the WordCloud image                      
plt.figure(figsize = (15, 15), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[24]:


#word cloud for text 
import re
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return  ' '.join(words)

def clean_text(text):
    clean_data = text.lower()
    clean_data = re.sub(r"http.*", " ",clean_data)
    clean_data = re.sub(r"[0-9]+", " ",clean_data)
    clean_data = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ",clean_data)
    return clean_data


plt.figure(figsize=(25,12))
all_data["clean_data"] = all_data["text"].apply(clean_text)
tokenizer = RegexpTokenizer(r'\w+')
all_data["token_data"] = all_data["clean_data"].apply(tokenizer.tokenize)
all_data["token_data"] = all_data["token_data"].apply(remove_stopwords)
pd.Series(' '.join(all_data["token_data"]).split()).value_counts(sort=True).nlargest(30).plot.bar()


# In[25]:


wordcloud = WordCloud(background_color='white').generate_from_frequencies(pd.Series(' '.join(all_data["token_data"]).split()).value_counts(sort=True, dropna=True).to_dict())
# plot the WordCloud image                      
plt.figure(figsize = (15, 15), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[138]:


location_map = {"Worldwide":"Earth","Everywhere":"Earth","United Kingdom":"UK","Manchester":"UK","London, England":"UK","Memphis, TN":"USA","Dallas, TX":"USA","Oklahoma City, OK":"USA","San Diego, CA":"USA","Pennsylvania, USA":"USA","Texas":"USA","Seattle":"USA","Chicago":"USA","Florida":"USA","NYC":"USA","San Francisco":"USA","Atlanta, GA":"USA","San Francisco, CA":"USA","New York City":"USA","US":"USA","Denver, Colorado":"USA","Chicago, IL":"USA","Nashville, TN":"USA","Los Angeles":"USA","Sacramento, CA":"USA","Toronto":"Canada","Los Angeles ":"USA","New York":"USA","California":"USA","New York, NY":"USA","United States":"USA","Mumbai":"India","Manhattan, NY":"USA","Los Angeles, CA":"USA","NYC area":"USA","Washington, DC":"USA","Washington, D.C.":"USA","London": "UK","California, USA":"USA"}


# In[145]:


train_data.drop(train_data[(train_data["location"] == "304") | (train_data["location"] == 'ss')].index, inplace=True)
disaster_data = train_data[train_data['target'] == 1]
non_disaster_data= train_data[train_data['target'] == 0]

loc_disaster = disaster_data["location"].str.replace('\$\$','\\$\\$').apply(lambda x : x if (x not in location_map)  else location_map.get(x)).value_counts(sort=True, dropna=True)
loc_no_disaster = non_disaster_data["location"].str.replace('\$\$','\\$\\$').apply(lambda x : x if (x not in location_map)  else location_map.get(x)).value_counts(sort=True, dropna=True)

plt.figure(figsize=(25,12))
loc_disaster.nlargest(10).plot.bar()


# In[146]:


#disaster location visualization
plt.figure(figsize=(25,12))
loc_no_disaster.nlargest(10).plot.bar()


# In[ ]:




