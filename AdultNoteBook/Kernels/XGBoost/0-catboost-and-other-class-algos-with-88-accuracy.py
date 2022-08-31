#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
data = pd.read_csv('../input/adult.csv')
print(data.shape)
data.count()[1]


# In[2]:


data.head()


# In[3]:


def cc(x):
    return sum(x=='?')
data.apply(cc)


# In[4]:


data.loc[data.workclass == '?'].apply(cc)


# In[5]:


data.groupby(by='workclass')['hours.per.week'].mean()


# In[6]:


df = data[data.occupation !='?']


# In[7]:


df.loc[df['native.country']!='United-States','native.country'] = 'non_usa'


# In[8]:


for i in df.columns:
    if type(df[i][1])== str:
        print(df[i].value_counts())


# In[9]:


df.columns


# In[10]:


import seaborn as sns
fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(15,20))
plt.xticks(rotation=45)
sns.countplot(df['workclass'],hue=df['income'],ax=f)
sns.countplot(df['relationship'],hue=df['income'],ax=b)
sns.countplot(df['marital.status'],hue=df['income'],ax=c)
sns.countplot(df['race'],hue=df['income'],ax=d)
sns.countplot(df['sex'],hue=df['income'],ax=e)
sns.countplot(df['native.country'],hue=df['income'],ax=a)


# In[11]:


fig, (a,b)= plt.subplots(1,2,figsize=(20,6))
sns.boxplot(y='hours.per.week',x='income',data=df,ax=a)
sns.boxplot(y='age',x='income',data=df,ax=b)


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


df_backup =df


# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in df.columns:
    df[i]=le.fit_transform(df[i])


# In[15]:


import random
import sklearn
random.seed(100)
train,test = train_test_split(df,test_size=0.2)


# # Baseline model 
# 
# We always set a baseline model which determines the threshold accuracy. The accuracy above this is only selected. Unlike regression model for which the baseline accuracy is mean of all outcomes. The baseline accuracy for classification model is mode of target variable.

# In[16]:


l=pd.DataFrame(test['income'])
l['baseline'] =0
k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],l['baseline']))
print(k)
(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# Hence our  baseline accuracy comes out to be 75%

# In[17]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
x=train.drop('income',axis=1)
y=train['income']
clf.fit(x,y)


# In[18]:


clf.score(x,y)


# In[19]:


pred = clf.predict(test.drop('income',axis=1))


# # Note this our Confusion Matrix

# In[20]:


import sklearn
k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred))
print(k)


# # precision of the model

# In[21]:


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# In[22]:


y_score = clf.fit(x,y).decision_function(test.drop('income',axis=1))

fpr,tpr,the=sklearn.metrics.roc_curve(test['income'],y_score)
sklearn.metrics.roc_auc_score(test['income'],pred)
plt.plot(fpr,tpr,)


# # Area under ROC curve.
#  which comes out pretty well !

# In[23]:


sklearn.metrics.roc_auc_score(test['income'],y_score)


# In[24]:


col=['age','fnlwgt','capital.gain','capital.loss','hours.per.week','education','education.num','marital.status','relationship','sex']


# # PCA analysis
# #### Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.
# 
# #### Generally this is called a data reduction technique. A property of PCA is that you can choose the number of dimensions or principal component in the transformed result.
# 
# #### In this case we will use it to analyse the feature importanace

# In[25]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=14)
Y_sklearn = sklearn_pca.fit_transform(X_std)

cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()

sklearn_pca.explained_variance_ratio_[:10].sum()

cum_sum = cum_sum*100

fig, ax = plt.subplots(figsize=(8,8))
plt.bar(range(14), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)


# ## Results:
# As you can see every feature adds significant variance hence droping the variables in this case is not a good idea!
# Therefore we should use every feature present to build our model as done previously.

# ## Decision Tree Model.

# In[26]:


from sklearn.tree import DecisionTreeClassifier


# ## Tuning the parameters of decision tree.
# max_features =10  (from above cum_sum chart) <br>
# min_samples_leaf=100 controling the branches by setting the limits (Pruning)

# In[27]:


clf = DecisionTreeClassifier(max_features=14,min_samples_leaf=100,random_state=10)
clf.fit(x,y)


# In[28]:


pred2 = clf.predict(test.drop('income',axis=1))


# In[29]:


k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))
print(k)


# In[30]:


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# # Nice result!

# # Now comes Xgboost

# In[31]:


from xgboost import XGBClassifier

clf= XGBClassifier()

clf.fit(x,y)

pred2 = clf.predict(test.drop('income',axis=1))

k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# 86.7% accuracy! quite good.

# In[35]:


from catboost import CatBoostClassifier

clf= CatBoostClassifier(learning_rate=0.04)

clf.fit(x,y)

pred2 = clf.predict(test.drop('income',axis=1))

k = pd.DataFrame(sklearn.metrics.confusion_matrix(test['income'],pred2))


(k[0][0]+k[1][1])/(k[0][0]+k[1][1]+k[0][1]+k[1][0])


# In[36]:


clf.score(x,y)


# In[ ]:




