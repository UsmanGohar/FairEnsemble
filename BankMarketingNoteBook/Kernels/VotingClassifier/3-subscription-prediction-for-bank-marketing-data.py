#!/usr/bin/env python
# coding: utf-8

# # Subscription Prediction for bank Marketing Data

# In[ ]:


# import the important libraries.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')
import pandas_profiling as pp


# ### Load the dataset.

# In[ ]:


# data Loading
data=pd.read_csv('../input/bank-additional-full.csv', sep = ';')
data.sample(5)


# In[ ]:


print("Shape of the data:",data.shape)
print("Columns Names are:\n",data.columns)


# In[ ]:


print("General Information about the Data")
data.info()


# In[ ]:


print("Data Types for all the columns of the data: \n",data.dtypes)


# ### Numeric or categorical data

# In[ ]:


numeric_data = data.select_dtypes(include = np.number)
numeric_data.head()


# In[ ]:


numeric_data.columns


# In[ ]:


categorical_data = data.select_dtypes(exclude = np.number)
categorical_data.head()


# In[ ]:


categorical_data.columns


# In[ ]:


pp.ProfileReport(data)


# In[ ]:


print("Is there any null values in the data ? \n",data.isnull().values.any())


# In[ ]:


print("Total Null Values in the data = ",data.isnull().sum().sum())


# In[ ]:


total= data.isnull().sum()
percent_missing = data.isnull().sum()/data.isnull().count()
print(percent_missing)


# In[ ]:


data[data.duplicated(keep='first')]


# In[ ]:


data.drop_duplicates(keep='first',inplace=True)


# In[ ]:


print("Information about the dataframe : \n ")
data.info()


# #### Total numbers of missing values values in each column. ####

# In[ ]:


# Which columns have the most missing values?
def missing_data(df):
    total = df.isnull().sum()
    percent = total/df.isnull().count()*100
    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    missing_values['Types'] = types
    missing_values.sort_values('Total',ascending=False,inplace=True)
    return(np.transpose(missing_values))
missing_data(data)


# In[ ]:


print('Discrption of Numeric Data : ')
data.describe()


# In[ ]:


print('Discrption of Categorical Data : ')
data.describe(include='object')


# ### Check for class imbalance

# In[ ]:


print("Target values counts:\n",data['y'].value_counts())
data['y'].value_counts().plot.bar()
plt.show()


# In[ ]:


class_values = (data['y'].value_counts()/data['y'].value_counts().sum())*100
class_values


# That makes it highly unbalanced, the positive class account for 11.27% of all target.

# In[ ]:


print("Histogram for the numerical features :\n")
data.hist(figsize=(15,15),edgecolor='k',color='skyblue')
plt.tight_layout()
plt.show()


# ### Univariate Analysis for categorical features

# In[ ]:


cols = categorical_data.columns
for column in cols:
    plt.figure(figsize=(15,6))
    plt.subplot(121)
    data[column].value_counts().plot(kind='bar')
    plt.title(column)
    plt.tight_layout()


# In[ ]:



data.plot(kind='box',subplots=True,layout=(4,3),figsize=(15,15))
plt.tight_layout()


# In[ ]:


data.groupby(["contact"]).mean()


# In[ ]:


data.groupby("education").mean()


# In[ ]:


data.pivot_table(values="age",index="month",columns=["marital","contact"])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
cat_var=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week','poutcome','y']
for i in cat_var:
    data[i]=LE.fit_transform(data[i])
    
data.head()


# In[ ]:


X= data.iloc[:,:-1]
y= data.iloc[:,-1:]


# In[ ]:


#Now with single statement, you will be able to see all the variables created globally across the notebook, data type and data/information
get_ipython().run_line_magic('whos', '')


# # Classification Models

# #### Import machine learnig libraries

# In[ ]:


from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge,Lasso, ElasticNetCV
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,confusion_matrix, mean_squared_error,accuracy_score, f1_score,classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm

from imblearn.over_sampling import SMOTE 


# In[ ]:


sc=StandardScaler()
sc.fit_transform(X)


# In[ ]:


sm = SMOTE(random_state = 2)
X_sm, y_sm = sm.fit_sample(X, y)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X_sm,y_sm,test_size=0.25,random_state=2)


# ## Logistic Regression

# In[ ]:


lr=LogisticRegression(penalty = 'l1',solver = 'liblinear')
lr.fit(X_train,y_train)
pred_lr=lr.predict(X_test)
print(confusion_matrix(y_test,pred_lr))
score_lr= accuracy_score(y_test,pred_lr)
print("Accuracy Score is: ", score_lr)
print("F1 Score is: ", f1_score(y_test,pred_lr))
print(classification_report(y_test, pred_lr))


# ## KNN
# 

# In[ ]:


knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
print(classification_report(y_test,pred_knn))
confusion_matrix(y_test,pred_knn)


# In[ ]:


score_knn = cross_val_score(knn,y_test,pred_knn,cv=5,scoring = 'f1')
print(score_knn)
print("Mean of the cross validation scores:",score_knn.mean())


# ## Decision Tree Classifier

# In[ ]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred_dt=dt.predict(X_test)
confusion_matrix(y_test,pred_dt)


# In[ ]:


score_dt=cross_val_score(dt,y_test,pred_dt,cv=5)
print(score_dt)
print("Mean of the cross validation scores:",score_dt.mean())


# ## Random Forest Classifier

# In[ ]:


rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pred_rf=rf.predict(X_test)
confusion_matrix(y_test,pred_rf)


# In[ ]:


score_rf=cross_val_score(rf,y_test,pred_dt,scoring='f1',cv=5)
print(score_rf)
print("Mean of the cross validation scores:",score_rf.mean())


# ## XGBoost Classifier

# In[ ]:


xgb_clf= xgb.XGBClassifier()
xgb_clf.fit(X_train,y_train)
pred_xgb=xgb_clf.predict(X_test)
confusion_matrix(y_test,pred_xgb)


# In[ ]:


score_xgb = cross_val_score(xgb_clf,y_test,pred_xgb,scoring = 'f1',cv=5)
print(score_xgb)
print("Mean of the cross validation scores:",score_xgb.mean())


# In[ ]:


print('Feature importances:\n{}'.format(repr(xgb_clf.feature_importances_)))


# ## Comaparison Between Model performances

# In[ ]:


print("F1 Score of Logistic Regression",score_lr)
print("F1 Score of KNN",score_knn.mean())
print("F1 Score of Decision Tree",score_dt.mean())
print("F1 Score of Random Forest",score_rf.mean())
print("F1 Score of XGB",score_xgb.mean())


# In[ ]:



plt.bar(x=["LR","KNN","DT","RF","XGB"],height=[score_lr,score_knn.mean(),score_dt.mean(),score_rf.mean(),score_xgb.mean()])
plt.title( "Model Performances of Models",fontsize = 22)
plt.xlabel("Models",fontsize = 16)
plt.ylabel("F1 Score",fontsize=16)
plt.ylim(0,1)
plt.show()


# So after comparing the f1 score of the models, we can say that XGBoost classier has better performance than other models.

# ### Thankyou for visit the kernel. If you have any suggustion please comment.if you feel the kernel helpful,
# 
# ### please upvote.
