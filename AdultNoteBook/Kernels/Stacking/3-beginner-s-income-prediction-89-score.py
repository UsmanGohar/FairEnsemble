#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this Notebook, I am working through the Income Prediction problem associated with the Adult Income Census dataset. The prediction task is to determine whether a person makes over $50K a year.
# 
# Following steps are used:
# 
# 1. Load Libraries
# 2. Load Data
# 3. Very Basic Data Analysis
# 4. Very Basic Visualizations
# 5. Feature Selection + Engineering
# 6. Modeling + Algorithm Tuning (will use Logistic Regression, Random Forest, XGBoost & CatBoost)
# 7. Finalizing the Model + Prediction
# 
# ### Evaluation Metric
# I will be using roc_auc_score for evaluation.
# 
# This is my first Notebook, and hence, I am pretty sure there is a lot of room to improve and add ons. Please feel free to leave me any comments with regards to how I can improve.

# # 1. Importing/ Loading Relevant Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# # 2. Loading Data

# In[ ]:


df = pd.read_csv('../input/adult-census-income/adult.csv')


# In[ ]:


df.head()


# Target variable : income

# # 3. Analyzing Data

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# Let's have a look at all the categorical columns present in our dataset

# In[ ]:


df.select_dtypes(exclude=np.number).columns


# Let's have a look at all the numerical columns present in our dataset

# In[ ]:


df.select_dtypes(include=np.number).columns


# In[ ]:


# Checking for null values, if any

df.isnull().sum()


# In[ ]:


# Checking for class imbalance

df['income'].value_counts()


# In[ ]:


# Converting the same into percentage, for better understanding

df['income'].value_counts(normalize=True)*100


# ##### Conclusions from basic analysis of data
# 
# 1. We can clearly see right off the bat that although no null values are present, there are some missing values as '?'
# 2. Something doesn't seem right with captial.gain, and capital.loss
# 3. fnlwgt seems similar to like an ID column
# 4. There is class imbalance. Almost 76% observations are earning less than, or equal to 50K

# # 4. Visualizations

# First we will visualize categorical features. Then numeric.

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(df['workclass'], hue=df['income']);


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.countplot(df['education'], hue=df['income']);

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(df['marital.status'], hue=df['income']);


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.countplot(df['occupation'], hue=df['income']);

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(df['relationship'], hue=df['income']);


# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(df['race'], hue=df['income']);


# In[ ]:


sns.countplot(df['sex'], hue=df['income']);


# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.countplot(df['native.country']);

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# #### Conclusions
# 
# 1. workclass, occupations, and native.country have missing values denoted as '?'
# 2. Need to check if education is correlated to numeric feature 'education.num'
# 3. Will need to see if native.country has any predictive power

# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['age']);


# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['fnlwgt']);


# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['education.num']);


# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['capital.gain'], kde=False);


# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['capital.loss'], kde=False);


# In[ ]:


plt.figure(figsize=(15,8))
sns.distplot(df['hours.per.week']);


# #### Conclusions
# 
# 1. age, fnlwgt seems right skewed.
# 2. education.num is multi-modal, which makes sense.
# 3. Majority of the values in capital.gain, and capital.loss are 0.
# 4. hours_per_week seems like normally distributed (although multi-modal), with majority of the people working around 40 hours a week.

# In[ ]:


# Checking correlation

sns.heatmap(df.corr(), annot=True, cmap='viridis');


# # 5. Feature Selection and Engineering

# In[ ]:


# Number of '?' in the dataset

for col in df.columns:
    print(col,':', df[df[col] == '?'][col].count())


# Instead of dropping the rows with seemingly missing values '?', I'll just rename it to 'Unknown', that way, if there is unseen data which the model sees with '?', it can help predict with better accuracy.

# In[ ]:


for cols in df.select_dtypes(exclude=np.number).columns:
    df[cols] = df[cols].str.replace('?', 'Unknown')


# In[ ]:


# Unique values in each categorical feature

for cols in df.select_dtypes(exclude=np.number).columns:
    print(cols, ':', df[cols].unique(), end='\n\n')


# In[ ]:


# Checking for correlation between columns 'education' and 'education-num'

pd.crosstab(df['education.num'],df['education'])


# We can clearly see that categorical feature 'education' can perfectly be described numeric feature 'education.num'. Hence, we can drop one column.

# Majority of the values in native.country seem to be USA. Let's find out percentage

# In[ ]:


df['native.country'].value_counts(normalize=True)*100


# Let's drop columns. We will drop - 
# 1. fnlwgt - seems exactly like ID column, so basically useless
# 2. native.country - almost 90% observations are from one country. Seems useless to me
# 3. capital.gain - majority of the values are 0
# 4. capital.loss - same as above
# 5. education - as this can be described by education.num

# In[ ]:


df.drop(['fnlwgt', 'capital.gain', 'capital.loss', 'native.country', 'education'], axis=1, inplace=True)


# In[ ]:


# Dropping rows with hours.per.week = 99

df.drop(df[df['hours.per.week'] == 99].index, inplace=True)


# In[ ]:


# Converting values in target column to numbers

df['income'] = df['income'].map({'<=50K':0, '>50K':1})


# In[ ]:


# Encoding categorical features

categorical_columns = df.select_dtypes(exclude=np.number).columns
new_df = pd.get_dummies(data=df, prefix=categorical_columns, drop_first=True)


# In[ ]:


new_df.shape


# In[ ]:


pd.set_option('max_columns', 50)
new_df.head()


# # 6. Modelling + Algorithm Tuning

# In[ ]:


X = new_df.drop('income', axis=1)
y = new_df['income']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


# Hyperparameter tuning of Logistic Regression

param_grid = {'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.001, 0.01, 0.1, 1, 10, 100],
             'solver':['lbfgs', 'liblinear'], 'l1_ratio':[0.001, 0.01, 0.1]}

grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, verbose=3)

grid.fit(X, y)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


log_reg = LogisticRegression(C=1, l1_ratio=0.001, solver='lbfgs', penalty='l2')


# In[ ]:


# Hyperparameter tuning of Random Forest

param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 5, 7, 9, 10], 'n_estimators':[100, 200, 300, 400, 500]}

grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=3)

grid.fit(X, y)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


rfc = RandomForestClassifier(max_depth=10, n_estimators=100, criterion='gini')


# In[ ]:


# Hyperparameter tuning of XGBoost

param_grid = {'max_depth':[2, 4, 5, 7, 9, 10], 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3], 'min_child_weight':[2, 4, 5, 6, 7]}

grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, verbose=3)

grid.fit(X, y)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


xgb = XGBClassifier(learning_rate=0.2, max_depth=4, min_child_weight=2)


# In[ ]:


# Hyperparameter tuning of CatBoost

param_grid = {'depth':[2, 4, 5, 7, 9, 10], 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3], 'iterations':[30, 50, 100]}

grid = GridSearchCV(CatBoostClassifier(), param_grid, verbose=3)

grid.fit(X, y)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


cb = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.1, verbose=False)


# Now, I can directly fit the model on training data, and then make predictions. However, I want to try different approach wherein I choose optimal hyperparameters, and train and predict on folds. Reason behind using this is, model
# on each fold will be better and could give a better score when we blend them.
# 
# Basically, training models on fold is done for two purposes:
# 1. to calculate average models
# 2. to train several models, predict with each of them and average their predictions. This makes the result more stable & robust
# 
# The below code is referenced from __[artgor's work](https://www.kaggle.com/artgor/bayesian-optimization-for-robots)__ (shoutout to him)
# 

# In[ ]:


classifiers = [log_reg, rfc, xgb, cb]

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)

scores_dict = {}

for train_index, valid_index in folds.split(X_train, y_train):
    # Need to use iloc as it provides integer-location based indexing, regardless of index values.
    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]
    
    for classifier in classifiers:
        name = classifier.__class__.__name__
        classifier.fit(X_train_fold, y_train_fold)
        training_predictions = classifier.predict_proba(X_valid_fold)
        # roc_auc_score should be calculated on probabilities, hence using predict_proba
        
        scores = roc_auc_score(y_valid_fold, training_predictions[:, 1])
        if name in scores_dict:
            scores_dict[name] += scores
        else:
            scores_dict[name] = scores

# Taking average of the scores
for classifier in scores_dict:
    scores_dict[classifier] = scores_dict[classifier]/folds.n_splits


# In[ ]:


scores_dict


# # 7. Finalising the Model + Prediction

# Clearly from the above scores dictionary, we can see that XGBoost fares better than the rest. Hence we will use the same, and predict our data

# In[ ]:


final_predictions = xgb.predict_proba(X_test)

print(roc_auc_score(y_test, final_predictions[:, 1]))

