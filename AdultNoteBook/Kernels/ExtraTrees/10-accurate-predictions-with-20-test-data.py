#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.DataFrame.from_csv("../input/adult.csv", header=0, index_col=None)


# In[3]:


data.head(5)


# In[4]:


data["income"].value_counts()


# In[5]:


data.isnull().values.any()


# In[6]:


data = data.replace("?", np.nan)


# In[7]:


data.isnull().sum()


# In[8]:


null_data = data[pd.isnull(data).any(1)]
null_data["income"].value_counts()


# In[9]:


data.dropna(inplace=True)


# In[10]:


bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
categories = pd.cut(data["age"], bins, labels=group_names)
data["age"] = categories


# In[11]:


sns.countplot(y='native.country',data=data)


# In[12]:


sns.countplot(y='education', hue='income', data=data)


# In[13]:


sns.countplot(y='occupation', hue='income', data=data)


# In[14]:


# education = education.num
data.drop(["education", "fnlwgt"], axis=1, inplace=True)


# In[15]:


from sklearn import preprocessing


# In[16]:


for f in data:
    if f in ["age", "workclass", "marital.status", "occupation", "relationship", "race", "sex", "native.country", "income"]:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[f])
        data[f] = le.transform(data[f])
data.head(5)


# In[17]:


y = data["income"]
X = data.drop(["income"], axis=1)


# In[18]:


from sklearn.ensemble import ExtraTreesClassifier


# In[19]:


forest = ExtraTreesClassifier(n_estimators=100,random_state=0)

forest.fit(X, y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[20]:


X = data.drop(["race", "native.country", "sex", "capital.loss", "workclass", "age"], axis=1)


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[24]:


forest = RandomForestClassifier(10)
forest.fit(X_train, y_train)


# In[25]:


predictions = forest.predict_proba(X_test)
predictions = [np.argmax(p) for p in predictions]


# In[26]:


precision = accuracy_score(predictions, y_test) * 100


# In[27]:


print("Precision: {0}%".format(precision))

