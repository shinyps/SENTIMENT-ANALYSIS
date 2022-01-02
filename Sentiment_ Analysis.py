#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"D:\DATASET\IMDB_sample.csv")


# In[3]:


df


# In[121]:


df['review'][0]


# In[122]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# In[123]:


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def stemmer(text):
    return [porter.stem(word) for word in text.split()]


# In[124]:


stemmer(df['review'][0])


# In[137]:


#vectorizing 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, tokenizer = stemmer, use_idf = True, norm ='l2',smooth_idf = True )


# In[138]:


Y = df.sentiment.values
X = tfidf.fit_transform(df.review)


# In[139]:


#classification using Logistic Regression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.5, shuffle=False)


# In[140]:


import pickle
from sklearn.linear_model import LogisticRegressionCV

logit = LogisticRegressionCV(cv=5, scoring='accuracy', max_iter=100)
logit.fit(X_train,y_train)


# In[141]:


filename = 'lr.sav'
pickle.dump(logit, open(filename, 'wb'))


# In[142]:


logit = pickle.load(open(filename, 'rb'))
# get_accuracy_metrics(y_test, logit.predict(X_test))


# In[143]:


logit.score(X_test,y_test)


# In[144]:


y_pred = logit.predict(X_test)


# In[145]:


def get_accuracy_metrics(y_test, y_hat):

  #Generating accuracy score
  print("\nAccuracy attained : {}\n".format(accuracy_score(y_hat,y_test)))

  #Getting the classification matrix
  print("The classification report is :\n\n{}".format(classification_report(y_test, y_hat)))

  #Confusion matrix
  # print("Confusion matrix generated"confusion_matrix(y_test, y_hat))
  print("\n")
  sns.heatmap(confusion_matrix(y_test, y_hat),annot=True,fmt='g', square=True)

def plot_accuracies(y_hat,model=''):
  x,accs=[],[]
  max_x,max_acc = 0,0

  #iterating through various values of threshold possible ie 0-100
  for i in range(0,105,5):
    
    x.append(i)
    z=[]
    
    #checking if the probability is greater than threshold for each y_hat predicted
    for row in y_hat*100:
      if max(row)>i:
        z.append(np.argmax(row))
      else :
        z.append(np.argmin(row))
    
    #Generating a list for accuracy scores
    accs.append(accuracy_score(z,y_test))
    
    if accuracy_score(z,y_test) >= max_acc : 
      max_acc = accuracy_score(z,y_test)
      max_x = i
    else : 
      continue

  #Plotting function
  plt.figure(figsize=(10,5))
  plt.plot(x,accs)
  plt.axvline(x=max_x,label='Maximum accuracy at x = {}, value is {}'.format(max_x, max_acc), color='red')
  plt.title("Accuracies of various thresholds of {}".format(model))
  plt.xlabel("Value of threshold")
  plt.ylabel("Accuracy")
  plt.legend()

  return accs,x


# In[146]:


import numpy    as np
import pandas   as pd
import seaborn  as sns

import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing    import StandardScaler
from sklearn.preprocessing    import RobustScaler
from sklearn.preprocessing    import MinMaxScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import accuracy_score
from sklearn.metrics          import classification_report
from sklearn.metrics          import confusion_matrix

from sklearn.decomposition  import PCA
get_accuracy_metrics(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




