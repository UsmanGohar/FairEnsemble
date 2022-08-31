#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing 

# The aim here is to classify whether a client will subscribe to a term deposit plan or not. (Binary Classification)
# Dataset url: https://www.kaggle.com/henriqueyamahata/bank-marketing

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[1]:


## Performance 
### Hyper parameter tuning:

# xgboost : 0.9460779708783467
# RandomForestClassifier : 0.8956317519962423


### UP SAMPLING and One Hot encoding

# Catboost : 0.9428839830906529
# LightGBM : 0.9457022076092062


### No Up sampling and LabelEncoding

# Catboost : 0.9081063340991139
# LightGBM : 0.9043321299638989


### Voting Classifier ( Logistic Regression, KNN, Decision Tree Classifier)

## Accuracy : 0.9389384687646782


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# In[8]:


df = pd.read_csv(r'/kaggle/input/bank-marketing/bank-additional-full.csv', sep=';')


# In[9]:


df.drop_duplicates(keep=False, inplace=True) 


# In[10]:


columns = df.columns.tolist()
numerical = [feature for feature in columns if df[feature].dtypes!='O']
categorical = [feature for feature in columns if feature not in numerical]
polychotomus = [feature for feature in categorical if feature not in ['default','housing','loan','contact','y']]


# ## Preprocessing  

# ### Missing Values ( Unknown, 999 ) 

# The Dataset contains missing values in categorical varaibles as "Unknown" and in numerical variables as "999" 

# In[11]:


for feature in columns:
    df[feature] = np.where(df[feature]=='unknown',np.nan,df[feature])


# In[12]:


## Missing values in Categorical

def Missing(df,columns):
    missing = []
    for feature in columns:
        missing_val = np.round(df[feature].isna().sum()/len(df), 3)*100
        missing.append([feature, missing_val])
    miss_df = pd.DataFrame(missing, columns=['Feature', '% Missing'])
    miss_df = miss_df[miss_df['% Missing'] != 0]
    return miss_df


# In[13]:


Missing(df, columns)


# In[14]:


# Missing values in Numerical

count_999 = []
for feature in numerical:
    if (999 in df[feature].unique()):
        count_val = np.round((df[feature].value_counts()[999]/len(df[feature])),4)*100
        count_999.append([feature, count_val])
    else:
        pass
pd.DataFrame(count_999, columns=['Feature', '% Missing'])


# pdays and default have 96.32 and 20.9 percent missing values respectively hence need to drop these two columns. Also the rest of the columns having missing values, the rows have been dropped.

# In[15]:


df.dropna(axis=0,inplace=True)


# In[16]:


df.drop(['pdays','default'],axis=1, inplace=True)
columns.remove('pdays')
columns.remove('default')


# In[17]:


numerical = [feature for feature in columns if df[feature].dtypes!='O']
categorical = [feature for feature in columns if feature not in numerical]
polychotomus = [feature for feature in categorical if feature not in ['housing','loan','contact','y']]


# ### Unique Values  

# In[18]:


def unique(df, column):
    unique_vals = []
    for feature in column:
        val1 = df[feature].nunique()
        unique_vals.append([feature, val1])
    unique_df = pd.DataFrame(unique_vals, columns=['Feature', 'No. Unique'])

    return unique_df


# In[19]:


## Numerical Variables Describe
def describe_num(df,numerical):
    vals = []
    for feature in numerical:
        feat_des = [feature, df[feature].nunique(),np.round(df[feature].mean(),2), min(df[feature]),
                    np.quantile(df[feature],0.25),np.quantile(df[feature],0.5),np.quantile(df[feature],0.75),
                    max(df[feature])]
        vals.append(feat_des)
    desc_num = pd.DataFrame(vals, columns=['Feature','No. Unique','Mean','Min','Q1','Q2','Q3','Max'])
    return desc_num


# In[20]:


describe_num(df,numerical)


# In[21]:


unique(df, categorical)


# ## Exploratory Data Analysis 

# In[22]:


ax = sns.countplot(df['y'])
for p in ax.patches:
        ax.annotate('%{:.1f}'.format(np.round(p.get_height()/len(df),3)), (p.get_x()+0.3, p.get_height()+100))


# Dataset is highly imbalanced 9:1

# In[23]:


## Duration among the people who have and haven't agreed to the plan

plt.figure(figsize=(10,6))
sns.distplot(df[df['y']=='yes']['duration'])
sns.distplot(df[df['y']=='no']['duration'])
plt.legend(labels=['Yes','No'])
plt.show()


# Clients who have agreed for a term deposit plan tend to have a larger duration of conversation. Probably to get to know more about the details of the term deposit plan. On the other hand, the clients who are not interested donot attend the call for a long duration

# In[24]:


plt.figure(figsize=(15,6))


plt.subplot(1,2,1)
sns.distplot(df[df['y']=='yes']['campaign'])
sns.distplot(df[df['y']=='no']['campaign'])
plt.title('Campaign vs target')
plt.legend(labels=['Yes','No'])

plt.subplot(1,2,2)
sns.distplot(df[df['y']=='yes']['campaign'])
sns.distplot(df[df['y']=='no']['campaign'])
plt.title('Zoom to Peak')
plt.xlim([0,20])
plt.ylim([0.2,1.2])
plt.legend(labels=['Yes','No'])

plt.show()


# In[25]:


## Month 

plt.figure(figsize=(10,5))
sns.countplot(df['month'], order=['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], hue=df['y'])
plt.show()


# Most Employees have been contacted in May followed by July, August and then June. The percent of people agreeing to a Term deposit plan are also significantly large as compared in other months. 

# ## Feature Engineering

# ### Feature Creation 

# In[26]:


# Age Bin

def feat_creat(df):
    ## Age Binn
    age_bin = []
    for val in df['age']:
        if (val <= 32):
            age_bin.append(1)
        elif (val>32 and val<=38):
            age_bin.append(2)
        elif (val>38 and val<=47):
            age_bin.append(3)
        else:
            age_bin.append(4)
    df['Age_Bin'] = age_bin


# In[27]:


## Principal Component Analysis

pca = PCA(n_components=1)

pca.fit(df[numerical])
l1 = pca.transform(df[numerical])
df['PCA'] = l1

feat_creat(df)


# ### Encoding 

# In[28]:


## Housing, Loan : Dichotomus (yes, no)

map_dichot = {'yes':1,'no':0}
for feature in ['housing','loan','y']:
    df[feature] = df[feature].map(map_dichot)

## Contact : Dichotomus (Telephone, cellular)
df['contact'] = df['contact'].map({'telephone':1,'cellular':0})


# In[29]:


## df1 = label Encoding
## df2 = one hot encoding

df1 = df.copy()
df2 = df.copy()


# In[30]:


## Label Encoding Polychotomus Features

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[31]:


for feature in polychotomus:
    df1[feature] = le.fit_transform(df1[feature])


# In[32]:


## One Hot Encoding Categorical Features

df2 = pd.get_dummies(df, columns=polychotomus, drop_first=True)


# ### Upsampling using Smote 

# In[33]:


## Label Encoded variables

x1 = df1.drop(['y'],axis=1)
y1 = df1['y']


# In[34]:


## One Hot Encoded Variables

x2 = df2.drop(['y'],axis=1)
y2 = df2['y']


# In[35]:


from imblearn.over_sampling import SMOTE

SMOTE_OBJ = SMOTE()
xu,yu = SMOTE_OBJ.fit_sample(x2,y2)### Imp note hereee 


# ## Modelling 

# In[36]:


from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier 
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report


# In[37]:


### UP SAMPLING TRAIN TEST SPLIT

x_train,x_test,y_train,y_test = train_test_split(xu,yu,test_size=0.2,random_state=42)


# ### Random Forest  and XGBoost Hyperparameter tuning 

# #### XGBoost 

# In[38]:


## XGB

xgb = XGBClassifier()

param_grid = dict(learning_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                  boosting_type=['gbdt','dart','goss','rf'],
                 n_estimators=[100, 200, 350, 400])
xgb_ran = RandomizedSearchCV(xgb, param_grid, cv=5, n_jobs=-1)


# In[39]:


xgb_ran.fit(x_train, y_train)


# In[40]:


xgb_ran.score(x_test, y_test)


# In[41]:


print(xgb_ran.best_params_)


# In[43]:


pred_xgb = xgb_ran.predict(x_test)


# In[44]:


print(classification_report(y_test, pred_xgb))


# #### Random Forest 

# In[45]:


## RFC

random_grid = {'n_estimators': [100,200,400,300],
               'max_features': ['auto','sqrt'],
               'max_depth': [1, 3, 5],
               'min_samples_split': [5,10,15]}

rfc_ran = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, 
                               n_iter = 7, cv = 5, random_state=42, n_jobs = -1)


# In[46]:


rfc_ran.fit(x_train, y_train)


# In[47]:


print(rfc_ran.best_params_)


# In[49]:


rfc_ran.score(x_test, y_test)


# In[50]:


pred_rfc = rfc_ran.predict(x_test)
print(classification_report(y_test, pred_rfc))


# ### Upsampling and One Hot Encoding

# #### Catboost  

# In[51]:


## cat boost upsampling

cat = CatBoostClassifier()
cat.fit(x_train,y_train)


# In[52]:


cat.score(x_test, y_test)


# In[53]:


pred_cat = cat.predict(x_test)
print(classification_report(y_test, pred_cat))


# #### LGBM 

# In[54]:


lgb = LGBMClassifier()
lgb.fit(x_train, y_train)


# In[55]:


lgb.score(x_test, y_test)


# In[56]:


pred_lgb = lgb.predict(x_test)
print(classification_report(y_test, pred_lgb))


# In[57]:


lgb_featimp = pd.DataFrame(lgb.feature_importances_, index=x_train.columns).sort_values(by=0,ascending=False)


# In[58]:


## Feature importance LGBM

plt.figure(figsize=(10,12))
sort_lgb = lgb.feature_importances_.argsort()
plt.barh(x_train.columns[sort_lgb], lgb.feature_importances_[sort_lgb])
plt.show()


# #### Voting Classifier 

# In[59]:


estimator = [] 
estimator.append(('LR',  
                  LogisticRegression(max_iter = 10000))) 
estimator.append(('KNN', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 ))) 
estimator.append(('DTC', DecisionTreeClassifier())) 


# In[60]:


vot_soft = VotingClassifier(estimators = estimator, voting ='soft') 


# In[61]:


vot_soft.fit(x_train, y_train)


# In[62]:


vot_soft.score(x_test, y_test)


# In[63]:


pred_vot_soft = vot_soft.predict(x_test)
print(classification_report(y_test, pred_vot_soft))


# ### No upsampling technique used and Label Encoded

# In[64]:



x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,test_size=0.2,random_state=42)


# #### Catboost 

# In[65]:


cat.fit(x1_train, y1_train)


# In[66]:


print("Accuracy : ", cat.score(x1_test, y1_test),"\n")
print(classification_report(y1_test, cat.predict(x1_test)))


# #### LGBM

# In[67]:


lgb.fit(x1_train, y1_train)


# In[68]:


print("Accuracy : ", lgb.score(x1_test, y1_test),"\n")
print(classification_report(y1_test, lgb.predict(x1_test)))


# In[69]:


### Hyper parameter tuning:

# xgboost : 0.9460779708783467
# RandomForestClassifier : 0.8956317519962423


### UP SAMPLING and One Hot encoding

# Catboost : 0.9428839830906529
# LightGBM : 0.9457022076092062


### No Up sampling and LabelEncoding

# Catboost : 0.9081063340991139
# LightGBM : 0.9043321299638989


### Voting Classifier ( Logistic Regression, KNN, Decision Tree Classifier)

## Accuracy : 0.9389384687646782


# In[ ]:




