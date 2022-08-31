#!/usr/bin/env python
# coding: utf-8

# # KNN 

# # 1. Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier


# # 2. Dataset

# In[2]:


dataset = pd.read_csv('../input/adult.csv')
display(dataset.head(5))


# In[3]:


dataset.describe()


# In[4]:


display(dataset.info())


# ## 2.1 Analising columns values
# No line seems to have Nan or empty values. However futher investigation is needed once it's already possible to see '?' values in different columns. 

# In[5]:


print("Work class categories \n")
print(dataset['workclass'].unique())


# In[6]:


print("Education categories")
education_dataset = dataset[['education','education.num']]
education_dataset = education_dataset.drop_duplicates()

data = {'education': education_dataset['education'], 'education.num': education_dataset['education.num']}

education_dataset = pd.DataFrame(data=data)
education_dataset['education'].astype('category')
education_dataset.index = education_dataset['education.num']
print(education_dataset[['education']].sort_values('education.num'))


# The columns 'education' and 'education.num' represent the same information. 'education.num' is the respective label for an education level. 

# In[7]:


print('marital status')
print(dataset['marital.status'].unique())
print(' \n occupation')
print(dataset['occupation'].unique())
print(' \n relationship')
print(dataset['relationship'].unique())
print(' \n race')
print(dataset['race'].unique())
print(' \n native.country')
print(dataset['native.country'].unique())


# ## 2.2 Dataset cleaning
# 
# As mentioned before, in many columns are present the '?' value for a missing information. So it will be removed every observation that contains a missing value. Also, it will be removed the columns 'capital.gain' and 'capital.loss', which doesn't seem so clear its meaning, and 'fnlwgt' which represents some final avaliation about the observation, thus the fitting process will get an equal result.

# In[8]:


#Replacing ? for a nan value to drop every line with it
dataset = dataset.replace({'?': np.nan})
dataset = dataset.dropna()
dataset = dataset.drop(['fnlwgt', 'capital.gain','capital.loss'], axis=1)


# # 3. EDA

# In[9]:


ax = dataset['sex'].value_counts().plot(kind="bar")
ax.set_ylabel("Quantity")
plt.title("Sex quantities")
plt.show()

ax = dataset['age'].hist()
ax.set_xlabel("Age")
ax.set_ylabel("Quantity")
plt.title("Age quantities")
plt.show()

ax = dataset['education.num'].hist()
ax.set_xlabel("Education label")
ax.set_ylabel("Quantity")
plt.title("Education level quantities")
plt.show()

ax = dataset['race'].value_counts().plot(kind="bar")
ax.set_ylabel("Quantity")
plt.title("Race quantities")
plt.show()

dataset['native.country'].value_counts().plot(kind="bar")
ax.set_ylabel("Quantity")
plt.title("Countries quantities")
plt.show()

dataset['income'].value_counts().plot(kind="bar")
ax.set_ylabel("Quantity")
plt.title("Income quantities")
plt.show()


# # 4. Fitting a model

# In[10]:


#Preparing the features and target
features = dataset.drop("income", axis=1)
target = dataset.income

#encoding the category features
features_to_encode = features[['workclass', 'education', 'marital.status',
       'occupation', 'relationship', 'race', 'sex',
       'native.country']]

features_encoded = features_to_encode.apply(preprocessing.LabelEncoder().fit_transform)
target = preprocessing.LabelEncoder().fit_transform(target)
features[['workclass', 'education', 'marital.status',
       'occupation', 'relationship', 'race', 'sex',
       'native.country']] = features_encoded

print(features.shape, target.shape)

display(features.head(5))


# In[11]:


#Dividing train and test data
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.3)


# In[12]:


#Analising the % importance level in each feature
forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(features, target)
importances = forest.feature_importances_
feature_importances = pd.DataFrame(importances*100,index = X_train.columns,columns=['importance']).sort_values('importance', ascending=False)
display(feature_importances)


# In[13]:


#Analisng the accuracy by increasing the number of K
scores = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

sns.lineplot(range(1,30), scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.title("Respective accuracy when increased the number of K")
plt.grid(True)
plt.show()


# In[14]:


print("The best K value in this dataset is {0} - Accuracy = {1}".format(scores.index(max(scores)), max(scores)))

