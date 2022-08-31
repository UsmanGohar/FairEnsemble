#!/usr/bin/env python
# coding: utf-8

# **Please give it a vote if you like it**

# <a></a>
# 
# ## Table of Contents
# 
# 
# [1. Abstract](#1)
# 
# [2. Division of the data between train and test](#2)
# 
# [3. Data Exploration with target varaible](#3)
# 
# [4. Data pre-processing](#4)
# 
# [5. Baseline model:](#5)
# 
#    -Model with missing values
#    
#    -Model imputing missing values
#    
#    -Models with SMOTE and UNDER techniques for imbalanced datasets
# 
# [6. Tuning and testing](#6)
# 
# [7. Shapply values - Feature importance](#7)
# 
# [8. Prediction and Evaluation](#8)
# 
# [9. Conclusion](#9)
# 
# 
#     

# 
# # 1.  Abstract 
# 
# 
# 
# 
# This data set was provided by Ronny Kohavi and Barry Becker from the 1994 US Cenus and retrieved from: "http://archive.ics.uci.edu/ml/datasets/Adul".
# 
# The task of this project is to classify whether a person makes more or less than 50K per year.
# 
# The dataset is composed of 14 variables (6 continuous and 8 nominal).
#     
# 
# Data with missing values: 
#     - 0 : 76%
#     - 1: 24%
#     
# Data dropping missing values: 
#     - 0: 0.7510% 
#     - 1: 0.2589%
# 
# 
# ****Imbalance data set probelm**
# **

# In[ ]:


## Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
from category_encoders.one_hot import OneHotEncoder
import imblearn 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import shap


# 
# # 2. Division of the data between train and test.

# In[ ]:


income_df = pd.read_csv("../input/adult-census-income/adult.csv")
print('Full data shape is {}'.format(income_df.shape))
income_df.head()


# In[ ]:


# Before dividing the data set between train and test. We will do some quick cleaning

# # Deleting duplicates 
income_df.duplicated().sum()

#Replacing '?' for nans
income_df = income_df.replace('?', np.NaN)


# Mapping already our target variable in  1 and 0.
income_df.income = income_df.income.map({'<=50K':0, '>50K':1}) ## mapping already our target variable in  1 and 0.


# In[ ]:


income_df.head()


# In[ ]:


train_data, test_data = train_test_split(income_df,
                                         test_size=0.2,
                                         stratify=income_df['income'],
                                         random_state=5)
print('Train data shape is {}'.format(train_data.shape))
print('Test data shape is {}'.format(test_data.shape))


#  
# # 3. Data Exploration with target varaible

# #### Correlation Matrix

# In[ ]:


#In order for LabelEncoder to work we need to drop all nans
corr_data = train_data.dropna()
categorical_data =corr_data.select_dtypes(include=['object'])
numerical_data = corr_data.select_dtypes(exclude=['object'])
encoder = LabelEncoder()
categorical_data = categorical_data.apply(encoder.fit_transform)
### concatenating numerical and categorical data which was enconded
corr_data = pd.concat([numerical_data, categorical_data], axis=1)

fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(111)

#Correlation Matrix with  full data
correlations = corr_data.corr()
matrix = np.triu(correlations.corr())
sns.heatmap(correlations, annot= True,  mask=matrix, cmap='coolwarm', ax=ax1)


# #### Correlation with the target variable

# In[ ]:


fig = plt.figure(figsize=((7,7)))
ax2 = fig.add_subplot(111)
#Correlation Matix with Target Variable
sns.heatmap(corr_data.corr()[['income']].sort_values('income').tail(10),vmin=0, cmap='Blues', annot=True, ax=ax2)
ax2.invert_yaxis()


# We can observe that we do not have highly correlated variables. Only 0.58 sex and relationship, followed by education and education_num 0.35. 
# 
# Additionally, we can observe the above correlation matrix graph where we can see the correlation with our target variable.
# - Education_num is positively correlated with income (0.34). This is reasonable since the more education you have, you more likely you are to have a higher salary. 
# - Age is another variable  that is positively correlated with income (0.23). However, this variable is parabolic and can bee seen in the below visualiztion that there is a point in the age where income will start decreasing. 
# - Hours per week is another variable postively correlated with income (0.23). The more hours you work, the higher your salary will tend to be. 

# ### Descriptive statistics

# ##### For categorical variables

# In[ ]:


cat_cols = ['workclass', 'education', 'marital.status', 'occupation',
       'relationship', 'race', 'sex', 'native.country']
train_data[cat_cols].describe()


# #### For numerical variables

# In[ ]:


num_cols =['age', 'education.num', 'capital.gain', 'capital.loss']
Mean =train_data[num_cols].mean()
Standard_deviation = train_data[num_cols].std()
result = pd.DataFrame({'Standard_deviation': Standard_deviation, 'Mean': Mean})
result


# We can observe that the average age is around 38.6 and education_num is 10.07 (which equals to some college).
# 
# We can see that capital_gain and capital_loss standard_deviation is higher than mean. This shows that people vary greatly in terms of capital gain and loss. 
# 

# 
# ## 3.1 Data visualization with target variable

# In[ ]:


train_data.income.value_counts()/len(train_data)


# We can observe that our target variable is imbalanced ( <= 50K:76%, > 50k: 24%). Therefore, we will address this problem later with some techniques for imbalanced classification.

# ### Age vs income

# In[ ]:


fig = plt.figure(figsize=(20,4))
ax1 = fig.add_subplot(111)

data_over50k=train_data[train_data['income']==1]
data_less50k= train_data[train_data['income']==0]
sns.kdeplot(data_less50k['age'], label = '<=50K', shade=True, color='#ADC6DF', ax=ax1)
sns.kdeplot(data_over50k['age'], label = '>50K', shade=True, color='#9B73B5', ax=ax1)

#Removing lines from the graph
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

#Title
ax1.set_title("Income by Age", loc='center',fontweight='bold',fontsize=14 )
ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
#X-Axis
plt.xticks(np.arange(10,100,10))

#Legend
line_labels = ["<=50K", ">50K"]
ax1.legend(
    loc="upper right",
    labels=line_labels)   


# In[ ]:


fig = plt.figure(figsize=(20,4))
ax1 = fig.add_subplot(111)
sns.countplot(x='age', hue='income', data= train_data, palette=['#ADC6DF','#9B73B5'])
#Removing lines from the graph
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

#Title
ax1.set_title("Income by Age", loc='center',fontweight='bold',fontsize=14)
ax1.set_xlabel(" ")
ax1.set_ylabel(' ')

#Legend
line_labels = ["<=50K", ">50K"]
ax1.legend(
    loc="upper right",
    labels=line_labels)   


# We can see that younger people are more likely to earn less than 50K per year. Which is obvious since those people do not have any work experience. The rates of people earning more than 50K start to increase as people age into their 30s. 
# 
# This is also seen in the purple line(>50K). The line starts to increase from 20  to 45 years old and then slowly decreasing.

# ### Workclass and occupation vs Income

# In[ ]:


fig = plt.figure(figsize=(15,15))
ax1= fig.add_subplot(411)
ax2= fig.add_subplot(412)
ax3= fig.add_subplot(413)
ax4= fig.add_subplot(414)

data_workclass = round(pd.crosstab(train_data.workclass, train_data.income).div(pd.crosstab(train_data.workclass, train_data.income).apply(sum,1),0),2)
data_occupation = round(pd.crosstab(train_data.occupation, train_data.income).div(pd.crosstab(train_data.occupation, train_data.income).apply(sum,1),0),2)

## Setting space between both subplots
plt.subplots_adjust(left=None,
                    bottom=None, 
                    right=None, 
                    top=1, 
                    wspace=None, 
                    hspace=0.5)

## Grapphing
sns.countplot(x='workclass', hue='income', data= train_data, ax=ax1, palette=['#ADC6DF','#9B73B5'])
data_workclass.plot.bar(color=['#ADC6DF','#9B73B5'], ax=ax2, edgecolor='w',linewidth=1.3)

sns.countplot(x='occupation', hue='income', data= train_data, ax=ax3, palette=['#ADC6DF','#9B73B5'])
data_occupation.plot.bar(color=['#ADC6DF','#9B73B5'], ax=ax4, edgecolor='w',linewidth=1.3 )

## Removing lines from the graph
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)


## Removing subplots legends
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30, ha='right')
ax3.get_legend().remove()
ax4.get_legend().remove()



## Title
ax1.set_title("Workclass", loc='center',fontweight='bold',fontsize=14)
ax3.set_title("Occupation", loc='center',fontweight='bold',fontsize=14)
ax2.set_title("Ratio of Workclass", loc='center',fontweight='bold',fontsize=14)
ax4.set_title("Ratio of Occupation", loc='center',fontweight='bold',fontsize=14)
ax1.set_xlabel(" ")
ax1.set_ylabel(' ')
ax2.set_xlabel(" ")
ax2.set_ylabel(' ')
ax3.set_xlabel(" ")
ax3.set_ylabel(' ')
ax4.set_xlabel(" ")
ax4.set_ylabel(' ')


## Legend
line_labels = ["<=50K", ">50K"]
fig.legend(
    loc="upper right",
    labels=line_labels) 


# Here we can see interesting things about our data:
# 
# 
# **Workclass**: 
#     From our data Private is the largest category by number.  If we look at the ratio, self-empl-inc has almost the same amount of people who earn more than 50K and less than 50K, and is the only category where more people earn above 50K. 
#     
# **Occupation**:
# 
#    We can see that Adm-Clerical and Machine-op-Inspect are the most frequent jobs. In terms of income, we can say that Exec-Managerical has the highest rate of >50K, followed by Armed Forces.  The job with the lowest >50K is Priv-house-serv. 
#     

# ### Education vs Income

# In[ ]:


fig = plt.figure(figsize=(15,7.5))
ax1= fig.add_subplot(211)
ax2= fig.add_subplot(212)
data_education = round(pd.crosstab(train_data.education, train_data.income).div(pd.crosstab(train_data.education, train_data.income).apply(sum,1),0),2)
## Setting space between both subplots
plt.subplots_adjust(left=None,
                    bottom=None, 
                    right=None, 
                    top=1, 
                    wspace=None, 
                    hspace=0.5)

## Grapphing
sns.countplot(x='education', hue='income', data= train_data, ax=ax1, palette=['#ADC6DF','#9B73B5'],order=train_data.education.value_counts().index)
data_education.sort_values(by=[0], ascending=False).plot.bar(color=['#ADC6DF','#9B73B5'], ax=ax2, edgecolor='w',linewidth=1.3)

## Removing lines from the graph
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

## Removing subplots legends
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')

## Title
ax1.set_title("Education", loc='center',fontweight='bold',fontsize=14)
ax2.set_title("Ratio of Education", loc='center',fontweight='bold',fontsize=14)
ax1.set_xlabel(" ")
ax1.set_ylabel(' ')
ax2.set_xlabel(" ")
ax2.set_ylabel(' ')

## Legend
line_labels = ["<=50K", ">50K"]
fig.legend(
    loc="upper right",
    labels=line_labels) 


# Here we can see that HS-grad is the most frequent education followed by Some college. 
# Additionally, we can see in the second graph, ratio of education, how the number of <=50 tends to decrease as people tend to have higher education. 
# 
# For example:  Doctorate. It seems that just 0.20 people who have doctorate tend to have <=50, the 0.80 tend to have more that 50K per year, which makes sense.  We observe similar ratios with prof-school and Masters.
# 
# Therefore, as we have seen before, Education_num has a huge impact in the income variable. This variable has the stronges correlation with our target variable: 0.34.
# 

# ### Race vs Income

# In[ ]:


fig = plt.figure(figsize=(20,10))
ax1= fig.add_subplot(221)
ax2= fig.add_subplot(222)

data_race = round(pd.crosstab(train_data.race, train_data.income).div(pd.crosstab(train_data.race, train_data.income).apply(sum,1),0),2)

## Grapphing
sns.countplot(x='race', hue='income', data= train_data, ax=ax1, palette=['#ADC6DF','#9B73B5'])
data_race.sort_values(by=[0], ascending=False).plot.bar(color=['#ADC6DF','#9B73B5'], ax=ax2, edgecolor='w',linewidth=1.3 )

#Removing lines from the graph
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha='right')

## Title
ax1.set_title("Race", loc='center',fontweight='bold',fontsize=14)
ax2.set_title("Ratio of Race", loc='center',fontweight='bold',fontsize=14)
ax1.set_xlabel(" ")
ax1.set_ylabel(' ')
ax2.set_xlabel(" ")
ax2.set_ylabel(' ')

#Removing subplots legends
ax1.get_legend().remove()
ax2.get_legend().remove()

#Legend
line_labels = ["<=50K", ">50K"]
fig.legend(
    loc="upper right",
    labels=line_labels)


# We can see that our sample is monstly White followed by Black race. One interesting thing we can see is that Asian-Pac-Islander have the highest ratio of >50K, followed by White, although they are very similar in terms of the ratio betwee our two classes. 

# ### Gender vs Income

# In[ ]:


fig = plt.figure(figsize=(20,9))
ax1= fig.add_subplot(221)
ax2= fig.add_subplot(222)

data_gender = round(pd.crosstab(train_data.sex, train_data.income).div(pd.crosstab(train_data.sex, train_data.income).apply(sum,1),0),2)

## Graphing
sns.countplot(x='sex', hue='income', data= train_data, ax=ax1, palette=['#ADC6DF','#9B73B5'])
data_gender.sort_values(by=[0], ascending=False).plot.bar(color=['#ADC6DF','#9B73B5'], ax=ax2, edgecolor='w',linewidth=1.3 )

#Removing lines from the graph
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha='right')

#Removing subplots legends
ax1.get_legend().remove()
ax2.get_legend().remove()


## Title
ax1.set_title("Gender", loc='center',fontweight='bold',fontsize=14)
ax2.set_title("Ratio of Gender", loc='center',fontweight='bold',fontsize=14)
ax1.set_xlabel(" ")
ax1.set_ylabel(' ')
ax2.set_xlabel(" ")
ax2.set_ylabel(' ')

#Legend
line_labels = ["<=50K", ">50K"]
fig.legend(
    loc="upper right",
    labels=line_labels)


# We can observe that male gender is the most frequent and also we observe that there are some disproportions distribuitions regarding income.
# Close to 10% of females earn more that 50.000 dollars compared to Male gender, where almost 40 percent earn more than 50.000 dollars.
# 

# # 4. Data pre-processing

# #### Dropping duplicates

# In[ ]:


train_data.drop_duplicates(inplace=True)
train_data.duplicated().sum()


# In[ ]:


train_data.duplicated().sum()


# ####  ? Dropping missing values

# In[ ]:


train_data.isnull().mean().sum()


# We are missing 13% of our data. We will create a baseline and compare if dropping these instances will drastically change our model performance.

# #### Dropping unncessary columns 

# We will drop the following columns
# 
# - Final weight: this column represents the number of people the census believes the entry represents. Therefore, we will drop this column since it does not give any meaningful information to our model.
# - Education: We already have an Education_num column which presents this information as an ordinal variable. We will drop it to avoid redundant information.

# In[ ]:


cols_to_drop = ['fnlwgt', 'education']
train_data.drop(cols_to_drop, inplace=True, axis=1)


# ## 5. Baseline model

# ## 5.1 Baseline model dropping missing values

# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('model',RandomForestClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'Random Forest AUC imputing missing values: {round(mean(scores),3)}')


# In[ ]:


train_data_without_na = train_data.dropna()

X = train_data_without_na.drop('income', axis=1)
y = train_data_without_na['income']
#Transformin categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('model',XGBClassifier())]
pipeline = Pipeline(steps=steps)

stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)
print(f'XGBClassifier AUC dropping missing values: {round(mean(scores),3)}')


# In[ ]:


train_data_without_na = train_data.dropna()

X = train_data_without_na.drop('income', axis=1)
y = train_data_without_na['income']
#Transformin categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('model',GradientBoostingClassifier())]
pipeline = Pipeline(steps=steps)

stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=-1)
print(f'Gradient Boosting AUC dropping missing values: {round(mean(scores),3)}')


# In[ ]:


train_data_without_na = train_data.dropna()

X = train_data_without_na.drop('income', axis=1)
y = train_data_without_na['income']
#Transformin categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('model',LogisticRegression())]
pipeline = Pipeline(steps=steps)

stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)
print(f'Logistic regression AUC dropping missing values: {round(mean(scores),3)}')


# ## 5.2 Baseline imputing mising values

# Since our missing values are categorical variables:
# - workclass  
# - occupation    
# - native_country      
# 
# We will use simple imputer with strategy ='most_frequent' (mode).

# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('model',RandomForestClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'Random Forest AUC imputing missing values: {round(mean(scores),3)}')


# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('model',XGBClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'XGBClassifier AUC imputing missing values: {round(mean(scores),3)}')


# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('model',GradientBoostingClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'Gradient Boosting  AUC imputing missing values: {round(mean(scores),3)}')


# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('model',LogisticRegression())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5, shuffle=True)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'Logistic Regression AUC imputing missing values: {round(mean(scores),3)}')


# ## 5.2 Models handling imbalanced data

# Another problem we have encounter was our data was imbalanced: 
#        - 0    0.76%
#        - 1    0.24%
# Having an imbalanced data set can create model performance with classification problems. Our class is not very severe therefore we will still try to use: 
# 
# 
# - SMOTE (oversampling): creating more samples for the minority class.
# - UNDER : decreasing the majority class.
# 
# In both cases we can see that the models using SMOTE techinique the models performed worse than in UNDER technique. However, still, our data set was not that severly imbalanced and we can see that our model performed better without using any technique. 
# 

# ### 5.2.1 Using Ovesampling technique

# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('over', SMOTE()),('model',XGBClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'XGBClassifier AUC imputing missing values + SMOTE: {round(mean(scores),3)}')


# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('over', SMOTE()),('model',GradientBoostingClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'Gradient Boosting AUC imputing missing values + SMOTE: {round(mean(scores),3)}')


# ### 5.2.2 Using undersample  technique

# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('under', RandomUnderSampler(random_state=5)),('model',XGBClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'XGBClassifier AUC imputing missing values + UNDER: {round(mean(scores),3)}')


# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),('under', RandomUnderSampler(random_state=5)),('model',GradientBoostingClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5)

scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=stratified_kfold, n_jobs=3)

print(f'Gradient Boosting AUC imputing missing values + UNDER: {round(mean(scores),3)}')


# 
# #### Summary:
# 
# - Imputing missing values did helped us to increase  AUC in the models. However, the increase it is not huge.
# 
# - Using SMOTE and UNDER techniques did note helped our model to get better results
# 
# - Best model: XGBClassifier AUC imputing missing values: 0.928

# # 6. Tuning and testing

# While tuning our best parameters were as follows:
# 
# **The best Parameter for XGBClassifier were: {'model__learning_rate': 0.1, 'model__max_depth': 3, 'model__n_estimators': 500}**
# 
# 

# In[ ]:


X = train_data.drop('income', axis=1)
y = train_data['income']

#Transforming categorical columns
categorical_columns= X.select_dtypes(object).columns
# Using pipeline
steps = [('encoding', OneHotEncoder(cols=categorical_columns)),
         ('imputer',SimpleImputer(missing_values=np.NaN, strategy='most_frequent')),
         ('model',XGBClassifier())]
pipeline = Pipeline(steps=steps)
stratified_kfold = StratifiedKFold(n_splits = 5, random_state=5)

param_grid = {
    'model__n_estimators': [500],
    'model__learning_rate': [0.1],
    'model__max_depth': [3]

}


grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=stratified_kfold.split(X, y),
                    scoring='roc_auc',  n_jobs=5, verbose=True)


# In[ ]:


grid.fit(X,y) 


# In[ ]:


AUC = grid.best_score_
best_param = grid.best_params_
print(f'XGBClassifier AUC: {round(AUC,4)}')
print(f'The best Parameter for XGBClassifier is: {best_param}')


# In[ ]:


full_model = grid.best_estimator_
full_model


# ## 7. Shapply values - Feature importance

# In[ ]:


encoded_variables = full_model['encoding'].transform(X)
encoded_variables.head()


# In[ ]:


explainer = shap.TreeExplainer(full_model['model'])

shap_values = explainer.shap_values(encoded_variables)
shap.summary_plot(shap_values, encoded_variables)


# In[ ]:


shap.summary_plot(shap_values, encoded_variables, plot_type='bar')


# We can see that:
# - Marital_status_1 (which is Married-civ-spouse) impacts the result of our model on average of 1.0. This means that married civilians have higher probability to have income higher that 50K. This agrees with our Data exploration section where we mentioned that Married-civ-spouse tend to be more likely to have >50K income.
# - The second one is age. As explained above, the older one is, the more experience one has and therefore likely to have a higher salary.
# - Capital gain is the third most important feature and we deduct that having better access to capital is a key factor in improving income.
# Lastly, education is the fourth most important feature. Again, this is logical, as the more education you have, you more likely you are to have a higher salary.

# # 8. Prediction and Evaluation

# In[ ]:


test_data.shape


# In[ ]:


test_data.duplicated().sum()


# #### Dropping unnecesary columns

# In[ ]:


cols_to_drop = ['fnlwgt', 'education']
test_data.drop(cols_to_drop, inplace=True, axis=1)


# In[ ]:


test_data.head()


# In[ ]:


X_test = test_data.drop('income', axis=1)
y_test = test_data['income']


# In[ ]:


predictions = full_model.predict(X_test)
print(classification_report(y_test,predictions))


# In[ ]:


predicted_probabilities = full_model.predict_proba(X_test)

print(f'Test AUC score is : {roc_auc_score(y_test,predicted_probabilities[:,1])}')


# # 9. Conclusion

# Our ROC_AUC is almost the same as in our train set: 0.928 versus 0.926.
# 
# We can conclude that our model is not overfitted and we are very happy with the results.
