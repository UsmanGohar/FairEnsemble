#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # **Stacked Classifier : Top 10 % on LB** 
# 
# 
# 
# ## **Introduction**
# 
# 
# Prashant Banerjee
# 
# 
# April 2020
# 
# 
# This notebook gives a very simple and basic introduction to an ensemble learning technique known as **stacking**. The objective of this notebook is to provide an intuitive understanding and implement **stacking**. We have used the famous titanic dataset for the illustration purposes.
# 
# 
# There is an excellent notebook on titanic survival. It is -
# 
# 
# [Titanic Survival Prediction End to End ML Pipeline](https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline) by **Poonam Ligade**. Nice data exploration.
# 
# 
# I have adapted several lines of code from the above notebook.
# 
# 
# Now let's begin our journey to understand stacking. So, let's dive in.

# **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> keep me motivated.**
# 

# <a class="anchor" id="0.1"></a>
# # **Notebook Contents**
# 
# - [Part 1 - Introduction to Stacking](#1)
# - [Part 2 - Stacking is prone to Overfitting](#2)
# - [Part 3 - Basic Set Up](#3)
#    - [3.1 Import libraries](#3.1)
#    - [3.2 Load data](#3.2)
# - [Part 4 - Data Exploration](#4)
# - [Part 5 - Data Visualization](#5)
# - [Part 6 - Data Preprocessing](#6)
# - [Part 7 - Feature Engineering](#7)
# - [Part 8 - Categorical Encoding](#8)
# - [Part 9 - Feature Scaling](#9)
# - [Part 10 - Declare feature vector and target variable](#10)
# - [Part 11 - Individual Classifier](#11)
# - [Part 12 - Stacked Classifier](#12)
# 
# 

# # **1. Introduction to Stacking** <a class="anchor" id="1"></a>
# 
# [Notebook Contents](#0.1)
# 
# 
# - [Stacking](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/) is an ensemble machine learning technique to combine multiple individual classification models via a meta-classifier. 
# 
# - But, wait what is a meta-classifier?
# 
# - Let's visualize the schematic representation of meta classifier below.
# 

# ![Stacked Classifier](https://www.researchgate.net/profile/David_Powers2/publication/264125265/figure/fig1/AS:295914087436290@1447562824204/Fusion-system-based-on-stacking.png)

# From the above diagram, we can conclude that stacking can be thought of as a two step process.
# 
# ### **Step 1** : In the first step, the individual classification models are trained based on the complete training set and their individual outputs are stored. These individual classification models are referred to as **Level One or Base Classifiers**.
# 
# 
# ### **Step 2** : In the second step, the predictions of individual classifiers (referred to as **Level One or Base Classifiers**) are used as new features to train a new classifier. This new classifier is called **Meta Classifier**. The meta-classifier can be any classifier of our choice. 
# 
# 
# The meta-classifier is fitted based on the outputs -- **meta-features** -- of the individual classification models in the ensemble. The meta-classifier can either be trained on the predicted class labels or probabilities from the ensemble.
# 
# The figure below shows how three different classifiers get trained. Their predictions get stacked and are used as features to train the meta-classifier which makes the final prediction.

# ![Stacked Classifier](https://miro.medium.com/max/2044/1*5O5_Men2op_sZsK6TTjD9g.png)

# # **2. Stacking is prone to overfitting** <a class="anchor" id="2"></a>
# 
# [Notebook Contents](#0.1)
# 
# 
# - This type of Stacking is prone to overfitting due to information leakage.
# 
# - To prevent information leakage into the training set from the target set, the level one predictions should come from a subset of the training data that was not used to train the level one classifiers.
# 
# - This can be applied by applying k-fold cross validation technique. In this technique, the training data is split into k-folds. Then the first k-1 folds are used to train the level one classifiers. The validation fold is then used to generate a subset of the level one predictions. The process is repeated for each unique group to generate the level one predictions.
# 
# - The figure below illustrates this process -

# ![k-fold Cross Validation Techniques](https://miro.medium.com/max/2972/1*RP0pkQEOSrw9_EjFu4w3gg.png)

# - Now, let's get to implementation of stacking or stacked classifier.
# 
# - The first step is to import the libraries and dataset

# # **3. Basic Set Up** <a class="anchor" id="3"></a>
# 
# [Notebook Contents](#0.1)
# 
# 

# ## **3.1 Import Libraries** <a class="anchor" id="3.1"></a>
# 
# [Notebook Contents](#0.1)
# 

# In[1]:


## Ignore warning
import warnings 
warnings.filterwarnings('ignore') 


# Data processing and analysis libraries
import numpy as np
import pandas as pd
import re


# Data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)


# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier


# Data preprocessing :
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Modeling helper functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score


# Classification metrices
from sklearn.metrics import accuracy_score


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## **3.2 Load data** <a class="anchor" id="3.2"></a>
# 
# [Notebook Contents](#0.1)
# 
# 

# In[3]:


# Load train and Test set

get_ipython().run_line_magic('time', '')

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
submission_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
IDtest = test_df['PassengerId']


# # **4. Data Exploration** <a class="anchor" id="4"></a>
# 
# [Notebook Contents](#0.1)
# 

# ### **Check the shape of the datasets**

# In[4]:


print(f'The shape of the training set : ', (train_df.shape))
print(f'The shape of the test set : ', (test_df.shape))
print(f'The shape of the submission set : ', (submission_df.shape))


# ### **Preview training set**

# In[5]:


train_df.head()


# ### **Preview test set**

# In[6]:


test_df.head()


# ### **View concise summary of training set**

# In[7]:


train_df.info()


# We suspect missing values in `Age`,`Cabin` and `Embarked` in training set. We will explore it later.

# ### **View concise summary of test set**

# In[8]:


test_df.info()


# Here, missing values occur in `Age`,`Fare` and `Cabin`. We will see it later.

# ### **Check for missing values**

# In[9]:


# missing values in training set

var1 = [col for col in train_df.columns if train_df[col].isnull().sum() != 0]

print(train_df[var1].isnull().sum())


# So, we are right that `Age`, `Cabin` and `Embarked` contain missing values in training set.

# In[10]:


# missing values in test set

var2 = [col for col in test_df.columns if test_df[col].isnull().sum() != 0]

print(test_df[var2].isnull().sum())


# `Age`, `Fare` and `Cabin` contain missing values in test set.

# ### **View statistical properties**

# In[11]:


train_df.describe()


# In[12]:


test_df.describe()


# ### **Types of Variables**
# 
# 
# Now, we will classify the variables into categorical and numerical variables.

# In[13]:


# find categorical variables in training set

categorical1 = [var for var in train_df.columns if train_df[var].dtype =='O']

print('There are {} categorical variables in training set.\n'.format(len(categorical1)))

print('The categorical variables are :', categorical1)


# In[14]:


# find numerical variables in training set

numerical1 = [var for var in train_df.columns if train_df[var].dtype !='O']

print('There are {} numerical variables in training set.\n'.format(len(numerical1)))

print('The numerical variables are :', numerical1)


# In[15]:


# find categorical variables in test set

categorical2 = [var for var in test_df.columns if test_df[var].dtype =='O']

print('There are {} categorical variables in test set.\n'.format(len(categorical2)))

print('The categorical variables are :', categorical2)


# In[16]:


# find numerical variables in test set

numerical2 = [var for var in test_df.columns if test_df[var].dtype !='O']

print('There are {} numerical variables in test set.\n'.format(len(numerical2)))

print('The numerical variables are :', numerical2)


# # **5. Data Visualization** <a class="anchor" id="5"></a>
# 
# [Notebook Contents](#0.1)
# 

# ## **5.1 Missing values** <a class="anchor" id="5.1"></a>
# 
# [Notebook Contents](#0.1)

# In[17]:


# view missing values in training set

msno.matrix(train_df, figsize = (30,10))


# In[18]:


# view missing values in test set

msno.matrix(test_df, figsize = (30,10))


# ## **5.2 Survived**  <a class="anchor" id="5.2"></a>
# 
# [Notebook Contents](#0.1)

# In[19]:


train_df['Survived'].value_counts()


# Here 0 stands for not survived and 1 stands for survived.
# 
# So, 549 people survived and 342 people did not survive.
# 
# Let's visualize it by plotting.

# In[20]:


fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train_df['Survived'], data = train_df, palette = 'PuBuGn_d')
graph.set_title('Distribution of people who survived', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# Now females have higher probability of survival than males.
# 
# Let' check it

# In[21]:


train_df.groupby('Survived')['Sex'].value_counts()


# In[22]:


fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train_df['Survived'], data = train_df, hue='Sex', palette = 'PuBuGn_d')
graph.set_title('Distribution of people who survived', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# Let's check the percentage of survival for males and females separately.

# In[23]:


females = train_df[train_df['Sex'] == 'female']
females.head()


# In[24]:


females['Survived'].value_counts()/len(females)


# In[25]:


males = train_df[train_df['Sex'] == 'male']
males.head()


# In[26]:


males['Survived'].value_counts()/len(males)


# As expected females have higher probability of survival (value 1) 74.20% than males 18.89%.
# 
# Let's visualize it.

# In[27]:


# create the first of two pie-charts and set current axis
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 1)   # (rows, columns, panel number)
labels1 = females['Survived'].value_counts().index
size1 = females['Survived'].value_counts()
colors1=['cyan','pink']
plt.pie(size1, labels = labels1, colors = colors1, shadow = True, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage of females who survived', fontsize = 20)
plt.legend(['1:Survived', '0:Not Survived'], loc=0)
plt.show()

# create the second of two pie-charts and set current axis
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 2)   # (rows, columns, panel number)
labels2 = males['Survived'].value_counts().index
size2 = males['Survived'].value_counts()
colors2=['pink','cyan']
plt.pie(size2, labels = labels2, colors = colors2, shadow = True, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage of males who survived', fontsize = 20)
plt.legend(['0:Not Survived','1:Survived'])
plt.show()


# ## **5.3 Sex** <a class="anchor" id="5.3"></a>
# 
# [Table of Contents](#0.1)
# 

# In[28]:


train_df['Sex'].value_counts()


# In[29]:


fig, ax = plt.subplots(figsize=(6,6))
graph = sns.countplot(ax=ax,x=train_df['Sex'], data=train_df, palette = 'bone')
graph.set_title('Distribution of sex among passengers', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[30]:


train_df['Sex'].value_counts()/len(train_df)


# In[31]:


plt.figure(figsize=(8,6))
labels = train_df['Sex'].value_counts().index
size = train_df['Sex'].value_counts()
colors=['cyan','pink']
plt.pie(size, labels = labels, shadow = True, colors=colors, autopct='%1.1f%%',startangle = 90)
plt.title('Percentage distribution of sex among passengers', fontsize = 20)
plt.legend()
plt.show()


# ## **5.4 Pclass** <a class="anchor" id="5.4"></a>
# 
# [Table of Contents](#0.1)
# 

# In[32]:


train_df.groupby('Pclass')['Sex'].value_counts()


# In[33]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train_df['Pclass'], data=train_df, palette = 'bone')
graph.set_title('Number of people in different classes', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[34]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train_df['Pclass'], data=train_df, hue='Survived', palette = 'bone')
graph.set_title('Distribution of people segregated by survival', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# Here 0 stands for not survived and 1 stands for survived.
# 
# So, we can see that Pclass plays a major role in survival.
# 
# Majority of people survived in Pclass 1 while a large number of people do not survive in Pclass 3.

# In[35]:


# percentage of survivors per class
sns.factorplot('Pclass', 'Survived', data = train_df)


# ## **5.5 Embarked** <a class="anchor" id="5.5"></a>
# 
# [Table of Contents](#0.1)

# In[36]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train_df['Embarked'], data=train_df, palette = 'bone')
graph.set_title('Number of people across different embarkment', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# In[37]:


fig, ax = plt.subplots(figsize=(8,6))
graph = sns.countplot(ax=ax,x=train_df['Embarked'], data=train_df, hue='Survived', palette = 'bone')
graph.set_title('Number of people who survived across different embarkment', fontsize = 12)
graph.set_xticklabels(graph.get_xticklabels(),rotation=30)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")


# We can see that port of embarkment plays a major role in survival probability.

# ## **5.6 Age** <a class="anchor" id="5.6"></a>
# 
# [Table of Contents](#0.1) 

# In[38]:


x = train_df['Age']
plt.figure(figsize=(8,6))
plt.hist(x, bins=25, color='g')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.title('Age distribution of passengers', fontsize = 20)
plt.show()


# We can see that majority of passengers are aged between 20 and 40.

# In[39]:


plt.figure(figsize=(8,6))
train_df.Age[train_df.Pclass == 1].plot(kind='kde')    
train_df.Age[train_df.Pclass == 2].plot(kind='kde')
train_df.Age[train_df.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;


# ## **5.7 Visualizations about training set** <a class="anchor" id="5.7"></a>
# 
# [Table of Contents](#0.1) 

# In[40]:


train_df.hist(bins=10,figsize=(12,8),grid=False);


# We can see that `Age` and `Fare` are measured on very different scaling. So we need to do feature scaling before predictions.

# In[41]:


g = sns.FacetGrid(train_df, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age", color="green");


# ## **5.8 Correlation Heatmap** <a class="anchor" id="5.8"></a>
# 
# [Table of Contents](#0.1) 

# In[42]:


corr = train_df.corr()#["Survived"]
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01, square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# In[43]:


#correlation of features with target variable
train_df.corr()["Survived"]


# So, `Pclass` has got highest negative correlation with `Survived` and `Fare` has got highest positive correlation with `Survived`.

# In[44]:


g = sns.factorplot(x="Age", y="Embarked",
                    hue="Sex", row="Pclass",
                    data=train_df[train_df.Embarked.notnull()],
                    orient="h", size=2, aspect=3.5, 
                   palette={'male':"purple", 'female':"blue"},
                    kind="violin", split=True, cut=0, bw=.2);


# # **6. Data Preprocessing** <a class="anchor" id="6"></a>
# 
# [Table of Contents](#0.1)
# 

# ## **6.1 Missing Values Imputation** <a class="anchor" id="6.1"></a>
# 
# [Table of Contents](#0.1)
# 
# 

# It is important to fill missing values, because some machine learning algorithms can't accept them eg SVM.
# 
# 
# But filling missing values with mean/median/mode is also a prediction which may not be 100% accurate, instead we can use models like Decision Trees and Random Forest which handle missing values very well.

# ### **Embarked Column**

# In[45]:


#Lets check which rows have null Embarked column
train_df[train_df['Embarked'].isnull()]


# **PassengerId** **62** and **830** have missing embarked values. Both have **Passenger class 1** and **fare $80**.
# 
# 
# Now, lets plot a graph to visualize and try to guess from where they embarked.

# In[46]:


plt.figure(figsize=(8,6))
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train_df)
plt.show()


# We can see that for 1st class median line is coming around fare $80 for embarked value 'C'. So we can replace NA values in Embarked column with 'C'.

# In[47]:


train_df["Embarked"] = train_df["Embarked"].fillna('C')


# In[48]:


#there is an empty fare column in test set
test_df.describe()


# ### **Fare Column**

# In[49]:


test_df[test_df['Fare'].isnull()]


# In[50]:


#we can replace missing value in fare by taking median of all fares of those passengers 
#who share 3rd Passenger class and Embarked from 'S' 
def fill_missing_fare(df):
    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
#'S'
       #print(median_fare)
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

test_df=fill_missing_fare(test_df)


# # **7. Feature Engineering** <a class="anchor" id="7"></a>
# 
# [Table of Contents](#0.1)
# 
# 

# ### **Deck- Where exactly were passenger on the ship?**

# In[51]:


train_df["Deck"]=train_df.Cabin.str[0]
test_df["Deck"]=test_df.Cabin.str[0]
train_df["Deck"].unique() # 0 is for null values


# In[52]:


g = sns.factorplot("Survived", col="Deck", col_wrap=4,
                    data=train_df[train_df.Deck.notnull()],
                    kind="count", size=2.5, aspect=.8);


# In[53]:


train_df = train_df.assign(Deck=train_df.Deck.astype(object)).sort_values("Deck")
g = sns.FacetGrid(train_df, col="Pclass", sharex=False,
                  gridspec_kws={"width_ratios": [5, 3, 3]})
g.map(sns.boxplot, "Deck", "Age");


# In[54]:


train_df.Deck.fillna('Z', inplace=True)
test_df.Deck.fillna('Z', inplace=True)
train_df["Deck"].unique() # Z is for null values


# How Big is your family?

# In[55]:


# Create a family size variable including the passenger themselves
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"]+1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"]+1
print(train_df["FamilySize"].value_counts())


# In[56]:


# Discretize family size
train_df.loc[train_df["FamilySize"] == 1, "FsizeD"] = 'singleton'
train_df.loc[(train_df["FamilySize"] > 1)  &  (train_df["FamilySize"] < 5) , "FsizeD"] = 'small'
train_df.loc[train_df["FamilySize"] >4, "FsizeD"] = 'large'

test_df.loc[test_df["FamilySize"] == 1, "FsizeD"] = 'singleton'
test_df.loc[(test_df["FamilySize"] >1) & (test_df["FamilySize"] <5) , "FsizeD"] = 'small'
test_df.loc[test_df["FamilySize"] >4, "FsizeD"] = 'large'


# In[57]:


print(train_df["FsizeD"].unique())
print(train_df["FsizeD"].value_counts())


# In[58]:


sns.factorplot(x="FsizeD", y="Survived", data=train_df);


# ### **Do you have longer names?**

# In[59]:


#Create feature for length of name 
# The apply method generates a new series

train_df["NameLength"] = train_df["Name"].apply(lambda x: len(x))
test_df["NameLength"] = test_df["Name"].apply(lambda x: len(x))
bins = [0, 20, 40, 57, 85]
group_names = ['short', 'okay', 'good', 'long']
train_df['NlengthD'] = pd.cut(train_df['NameLength'], bins, labels=group_names)
test_df['NlengthD'] = pd.cut(test_df['NameLength'], bins, labels=group_names)


# In[60]:


sns.factorplot(x="NlengthD", y="Survived", data=train_df)
print(train_df["NlengthD"].unique())


# ### **What's in the name?**

# In[61]:


import re

#A function to get the title from a name.
def get_title(name):
    """Use a regular expression to search for a title.  
       Titles always consist of capital and lowercase letters, and end with a period"""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# In[62]:


#Get all the titles and print how often each one occurs.
titles = train_df["Name"].apply(get_title)
print(pd.value_counts(titles))


# In[63]:


#Add in the title column.
train_df["Title"] = titles


# In[64]:


# Titles with very low cell counts to be combined to "rare" level
rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']


# In[65]:


# Also reassign mlle, ms, and mme accordingly
train_df.loc[train_df["Title"] == "Mlle", "Title"] = 'Miss'
train_df.loc[train_df["Title"] == "Ms", "Title"] = 'Miss'
train_df.loc[train_df["Title"] == "Mme", "Title"] = 'Mrs'
train_df.loc[train_df["Title"] == "Dona", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Lady", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Countess", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Capt", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Col", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Don", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Major", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Rev", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Sir", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Dr", "Title"] = 'Rare Title'


# ### **Do the same with test set**

# In[66]:


titles = test_df["Name"].apply(get_title)
print(pd.value_counts(titles))


# In[67]:


#Add in the title column.
test_df["Title"] = titles


# In[68]:


# Titles with very low cell counts to be combined to "rare" level
rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']


# In[69]:


# Also reassign mlle, ms, and mme accordingly
test_df.loc[test_df["Title"] == "Mlle", "Title"] = 'Miss'
test_df.loc[test_df["Title"] == "Ms", "Title"] = 'Miss'
test_df.loc[test_df["Title"] == "Mme", "Title"] = 'Mrs'
test_df.loc[test_df["Title"] == "Dona", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Lady", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Countess", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Capt", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Col", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Don", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Major", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Rev", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Sir", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Dr", "Title"] = 'Rare Title'


# In[70]:


test_df["Title"].value_counts()


# ### **Ticket column**

# In[71]:


train_df["Ticket"].tail()


# In[72]:


train_df["TicketNumber"] = train_df["Ticket"].str.extract('(\d{2,})', expand=True)
train_df["TicketNumber"] = train_df["TicketNumber"].apply(pd.to_numeric)


# In[73]:


test_df["TicketNumber"] = test_df["Ticket"].str.extract('(\d{2,})', expand=True)
test_df["TicketNumber"] = test_df["TicketNumber"].apply(pd.to_numeric)


# In[74]:


#some rows in ticket column dont have numeric value so we got NaN there
train_df[train_df["TicketNumber"].isnull()]


# In[75]:


train_df.TicketNumber.fillna(train_df["TicketNumber"].median(), inplace=True)
test_df.TicketNumber.fillna(test_df["TicketNumber"].median(), inplace=True)


# # **8. Categorical Encoding** <a class="anchor" id="8"></a>
# 
# [Table of Contents](#0.1)

# In[76]:


labelenc=LabelEncoder()

cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']
for col in cat_vars:
    train_df[col]=labelenc.fit_transform(train_df[col])
    test_df[col]=labelenc.fit_transform(test_df[col])


# In[77]:


train_df.head()


# ### **Age Column**
# 
# Age seems to be promising feature. So it doesnt make sense to simply fill null values out with median/mean/mode.
# 
# We will use Random Forest algorithm to predict ages.

# In[78]:


with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(train_df["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="red")
    plt.title("Age Distribution")
    plt.ylabel("Count");


# In[79]:


from sklearn.ensemble import RandomForestRegressor
#predicting missing values in age using Random Forest
def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',
                 'TicketNumber', 'Title','Pclass','FamilySize',
                 'FsizeD','NameLength',"NlengthD",'Deck']]
    # Split sets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df


# In[80]:


train_df=fill_missing_age(train_df)
test_df=fill_missing_age(test_df)


# # **9. Feature Scaling** <a class="anchor" id="9"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# We can see that Age, Fare are measured on different scales, so we need to do Feature Scaling first before we proceed with making predictions with **stacked classifier**.

# In[81]:


from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(train_df[['Age', 'Fare']])
train_df[['Age', 'Fare']] = std_scale.transform(train_df[['Age', 'Fare']])


std_scale = preprocessing.StandardScaler().fit(test_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = std_scale.transform(test_df[['Age', 'Fare']])


# ### **Correlation of features with target**

# In[82]:


train_df.corr()["Survived"]


# # **10. Declare feature vector and target label** <a class="anchor" id="10"></a>
# 
# [Table of Contents](#0.1)
# 

# In[83]:


# Declare feature vector and target variable
X_train = train_df.drop(labels = ['Survived'],axis = 1)
y_train = train_df['Survived']
X_test = test_df


# # **11. Individual Classifiers** <a class="anchor" id="11"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# For the purpose of illustration, we will train a **Support Vector Classifier (SVC)**, **Multi-layer Perceptron (MLP) classifier**, **Nu-Support Vector classifier (NuSVC)** and a **Random Forest (RF) classifier** — classifiers available in Scikit-learn. 
# 

# In[84]:


# Initializing Support Vector classifier
clf_svc = SVC(C = 50, degree = 1, gamma = "auto", kernel = "rbf", probability = True)

# Initializing Multi-layer perceptron  classifier
clf_mlp = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = 1000)

# Initialing Nu Support Vector classifier
clf_nusvc = NuSVC(degree = 1, kernel = "rbf", nu = 0.25, probability = True)

# Initializing Random Forest classifier
clf_rfc = RandomForestClassifier(n_estimators = 500, criterion = "gini", max_depth = 10,
                                     max_features = "auto", min_samples_leaf = 0.005,
                                     min_samples_split = 0.005, n_jobs = -1, random_state = 1000)


# # **12. Stacked Classifier** <a class="anchor" id="12"></a>
# 
# [Table of Contents](#0.1)
# 
# 
# To stack the above classifiers, we will use the [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier) from scikit-learn library.
# 
# 
# We can also use the [StackingCVClassifier](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/) from MLXTEND for the same purpose. We can take a look at the [official documentation](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/) since it goes in detail over useful examples of how to implement the StackingCVClassifier.
# 

# In[85]:


classifiers = [('svc', clf_svc),
               ('mlp', clf_mlp),                             
               ('nusvc', clf_nusvc),
               ('rfc', clf_rfc)]


# In[86]:


clf = StackingClassifier(estimators=classifiers, 
                         final_estimator=LogisticRegression(),
                         stack_method='auto',
                         n_jobs=-1,
                         passthrough=False)


# In[87]:


predictors=["Pclass", "Sex", "Age", "Fare", "Embarked","NlengthD",
              "FsizeD", "Title","Deck","NameLength","TicketNumber"]

clf.fit(X_train[predictors],y_train)


# In[88]:


test_predictions=clf.predict(X_test[predictors])


# In[89]:


test_predictions=test_predictions.astype(int)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_predictions
    })

submission.to_csv("titanic_submission.csv", index=False)


# In this notebook, we have demonstrated the stacked classifier.
# 
# Now we will come to the end of this kernel. I hope you find this kernel useful and enjoyable.
# 
# Your comments and feedback are most welcome.
# 
# Thank you
# 

# [Go to Top](#0)
