#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install scikit-plot


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt


from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score , confusion_matrix, classification_report


# In[5]:


from google.colab import files
uploaded = files.upload()


# In[7]:


df = pd.read_csv(r"C:\Users\Thamaiyanthi\Documents\Arathi-Projects\ML Project\spam_ham_dataset.csv")


# In[8]:


df


# In[9]:


df['label'] = df.label.map({'ham' : 0, 'spam' : 1})


# In[10]:


count_Class = pd.value_counts(df.label, sort = True)

# Data to Plot
labels = 'NotSpam', 'Spam'
sizes = [count_Class[0], count_Class[1]]
colors = ['green', 'red']
explode = (0.1, 0.1)

# Plot
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%', shadow = True, startangle = 90)
plt.axis('equal')
plt.show()


# In[12]:


X_train, X_test, y_train, y_test = tts(df['text'], df['label'], test_size=0.2, random_state=1)


# In[13]:


count_vector = CountVectorizer()
train_data = count_vector.fit_transform(X_train)
test_data = count_vector.transform(X_test)


# In[14]:


Mnb = MultinomialNB()
Mnb.fit(train_data, y_train)


# In[15]:


MnbPredicts = Mnb.predict(test_data)


# In[16]:


print("The accuracy of our Naïve Bayes multinomial model is {} %".format(accuracy_score(y_test, MnbPredicts) * 100))
print("The Precision of our Naïve Bayes multinomial model is {} %". format(precision_score(y_test, MnbPredicts)* 100))
print("The Recall of our Naïve Bayes multinomial model is {} %" . format(recall_score(y_test, MnbPredicts)* 100))


# In[17]:


confusionmatrix = confusion_matrix(y_test, MnbPredicts)
print("The accuracy of Naive Bayes clasifier is {}%".format(accuracy_score(y_test, MnbPredicts) * 100))
print("\n", confusionmatrix)
skplt.metrics.plot_confusion_matrix(y_test, MnbPredicts, normalize = True)
plt.show()


# In[25]:


new_test_sample_ham = ["Hi, I'm Arathi."]


# In[26]:


new_test_sample_spam= ["Congratulations, you've won a free Hp Laptop."]


# In[27]:


new_test_sample_ham_vectorized = count_vector.transform(new_test_sample_ham)


# In[28]:


new_test_sample_spam_vectorized = count_vector.transform(new_test_sample_spam)


# In[29]:


sample_predict = Mnb.predict(new_test_sample_ham_vectorized)
sample_predict


# In[30]:


sample_predict = Mnb.predict(new_test_sample_spam_vectorized)
sample_predict


# In[ ]:




