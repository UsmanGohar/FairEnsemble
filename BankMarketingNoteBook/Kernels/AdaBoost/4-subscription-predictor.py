#!/usr/bin/env python
# coding: utf-8

# # Subscription Predictor
# The provided data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution.
# 
# The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# ## Attribute Information:
# 
# ### Input variables:
# 
# #### bank client data:
# 1 - age (numeric)
# 
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# 
# 8 - contact: contact communication type (categorical: 'cellular','telephone')
# 
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# 
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
# 
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
# 
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 
# 20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
# ### Output variable (desired target):
# 
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# 
# *NOTE - Data from the site was in comma-seperated csv format it was changed to excell sheet manually*

# ## Applications of this Project  
# 
# 1. This Predictive Analysis of Data will help the user industry to plan there campaigns according to the previous data these predictions will help understand what the campaign should focus on and what should not to. 
# 
# 2. It will simplyfy the methods of Approaching the customers and will also give exact customers to whom company should approch what kind of people the should keep in target will nbe known by using this project.
# 
# 3. It also gives you details how much time company should focus on an XYZ pperson so that he is convinced to subscribe to there plans.
# 
# 4. Most Imp Use Of This Project is - It will Save TIME of Campaign as well as the company as, all work will be focused on correct targets.
# 
# 5. Who Can Use This Project Rather Than Banks? Answers is It has no limits, any company selling products can use this as it can use there data of calls of customers and predict will he be interested in there product.
# 
# SO LETS BEGIN!!!

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# importing basic but imp libraries.
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# reading data file as 'Data' 
Data = pd.read_csv('../input/bank-marketing/bank-additional-full.csv', sep = ';')


# In[4]:


Data.head(10)


# In[5]:


# 1st Problem here with this data was that it was having Yes/No
# as its values which was needed to be converted to 1/0 for Machine Learning purpose
Data.replace(('yes','no'),(1,0),inplace=True)


# In[6]:


Data.head(10)


# In[7]:


# 2nd Major issue here was the Data was having 'unknown' as value instead of 'NaN'
# Replacing 'unknown' to 'NaN'
Data.replace('unknown',np.nan,inplace=True)


# In[ ]:





# 

# In[8]:


Data.head(10)


# In[9]:


Data.groupby('job').count()


# In[10]:


Data.info()


# In[11]:


# Dropping "NaN's"
Data.dropna(inplace=True)
Data.info()


# ## Understanding and Manipulating Data

# In[12]:


# checking on 'default' values from the Data
sns.set_style('whitegrid')
sns.countplot(x='default',data=Data,palette='BuGn')


# In[13]:


# as we can see here all data is 0;
#dropping 'default' column from Data
Data.drop('default',axis=1,inplace=True)


# In[14]:


# Checking 'age' Values relation with 'y'
sns.jointplot(data=Data,x=Data['age'],y=Data['y'],kind='scatter')


# In[15]:


# as we can see Data is focused between age>20 and age<70
# so I added a restriction to Data for Age

indexNames = Data[Data['age']<20].index
Data.drop(indexNames, inplace=True)
indexNames = Data[Data['age']>70].index
Data.drop(indexNames, inplace=True)
sns.jointplot(data=Data,x=Data['age'],y=Data['y'],kind='scatter')


# In[16]:


Data.info()


# In[17]:


Data.head()


# Checking If Maritial Status had any effect on OutPut
# But to check soo,we first need to convert the data to its equallent format
# as 
# 
# Marital had 3 classes - 'divorced','married','single'
# Encoding it as 1,2,3

# In[18]:


Data.replace(('divorced','married','single'),(1,2,3),inplace=True)


# In[19]:


sns.jointplot(data=Data,x=Data['marital'],y=Data['y'],kind='scatter')


# In[20]:


sns.countplot(x='marital',data=Data,palette='BuGn')


# Marital colums has min effect on output so we can drop this column

# In[21]:


Data.drop('marital',inplace=True,axis=1)


# Similar to 'marital', 'job' and 'education' were also needed to be encoded 
# so encoding 'education' and then 'job'

# In[22]:


Data.replace(("basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree"),(1,2,3,4,5,6,7),inplace=True)
sns.jointplot(data=Data,x=Data['education'],y=Data['y'],kind='reg')


# In[23]:


Data.replace(("admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed"),(1,2,3,4,5,6,7,8,9,10,11),inplace=True)
sns.jointplot(data=Data,x=Data['job'],y=Data['y'],kind='reg')


# In[24]:


#checking relation og 'housing' with 'y'
sns.jointplot(data=Data,x=Data['housing'],y=Data['y'],kind='scatter')


# In[25]:


# checking 'loan' column
sns.countplot(x='loan',data=Data)


# In[26]:


sns.jointplot(data=Data,x=Data['loan'],y=Data['y'],kind='scatter')


# Method of Contact is no use for prediction so directly dropping it
# 

# In[27]:


Data.drop('contact',axis=1,inplace=True)


# In[28]:


#checking relation of 'duration of contact' with 'y'
sns.jointplot(data=Data,x=Data['duration'],y=Data['y'],kind='scatter')


# In[29]:


#checking relation of 'campaign' with 'y'
# campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
sns.jointplot(data=Data,x=Data['campaign'],y=Data['y'],kind='scatter')


# pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# not usefull so dropping

# In[30]:


Data.drop('pdays',axis=1,inplace=True)


# previous: number of contacts performed before this campaign and for this client (numeric)
# checking with 'previous' column

# In[31]:


sns.jointplot(data=Data,x=Data['previous'],y=Data['y'],kind='scatter')


# It has Some Relation so we will keep it 

# 
# Similar to 'education','job','marital', 'previous outcome' also needs  to be encoded 
# so encoding it and checking relation with 'y'
# 
# poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

# In[32]:


Data.replace(("failure","nonexistent","success"),(1,2,3),inplace=True)
sns.jointplot(data=Data,x=Data['poutcome'],y=Data['y'],kind='scatter')


# keeping it as it is and as well as checking relation of 'y' with -
# 1. emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 
# 2. cons.price.idx: consumer price index - monthly indicator (numeric)
# 
# 3. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# 
# 4. euribor3m: euribor 3 month rate - daily indicator (numeric)
# 
# 5. nr.employed: number of employees - quarterly indicator (numeric)

# In[33]:


sns.jointplot(data=Data,x=Data['emp.var.rate'],y=Data['y'],kind='scatter')


# In[34]:


sns.jointplot(data=Data,x=Data['cons.price.idx'],y=Data['y'],kind='scatter')


# In[35]:


sns.jointplot(data=Data,x=Data['cons.conf.idx'],y=Data['y'],kind='scatter')


# In[36]:


sns.jointplot(data=Data,x=Data['euribor3m'],y=Data['y'],kind='scatter')


# In[37]:


sns.jointplot(data=Data,x=Data['nr.employed'],y=Data['y'],kind='scatter')


# In[38]:


# Dropping Day of week And Month of Contact Irrelevant
Data.drop('day_of_week',axis=1,inplace=True)
Data.drop('month',axis=1,inplace=True)


# In[39]:


Data.info()


# Now Data has been Cleaned Lets move on to Data Splitting and Model Selection
# 
# # Data Splitting And Model Selection

# In[40]:


# Importing Train_Test_Split Model for data Splitting
from sklearn.model_selection import train_test_split


# In[41]:


X = Data.drop('y', axis=1)
Y = Data['y']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30,random_state=101)
# test_size indicate how much portion of data to include in test dataset
# random_state is the seed used by the random number generator


# In[42]:


# importing model selection functions
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV # cross Validation method
from sklearn.naive_bayes import GaussianNB

# All Classification Algorithms Used 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Scaleling functions used
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Methods used for Accuracy Check 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


# Trying Out All Algorithms on the data set for checking the cross validation score then picking the best algorithm for our task 

# In[43]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
n_splits = 5

for name, model in models:
    kfold = model_selection.KFold(n_splits=n_splits, shuffle=True,                                   random_state=5)
    cv_results = model_selection.cross_val_score(model, X_train,                                                  y_train, cv=kfold,                                                  scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %5.2f (%5.2f)" % (name, cv_results.mean()*100,                            cv_results.std()*100)
    print(msg)


# LR: 89.90 ( 0.27)
# 
# LDA: 89.90 ( 0.25)
# 
# KNN: 88.81 ( 0.20)
# 
# CART: 87.81 ( 0.32)
# 
# SVM: 87.44 ( 0.39)
# 
# NB: 82.12 ( 0.53)
# 
# As Here Max Accuracy is achieved by Linear Dicremenent Model(LDA) & Linear Regression(LR)
# 
# Lets Try This again on Scaled Data

# In[44]:


results_df = pd.DataFrame(results, index=names,columns='CV1 CV2 CV3 CV4 CV5'.split())
results_df


# In[45]:


results_df['CV Mean'] = results_df.iloc[:,0:n_splits].mean(axis=1)
results_df['CV Std Dev'] = results_df.iloc[:,0:n_splits].std(axis=1)
pd.set_option('precision',2)
results_df*100


# As We Can see the Precise Accuracy of Models
# 
# Trying out same with Scaled Data
# 
# 
# ### Using StandardScaler() method 
# This Method is used to centrallize and Scale the Data Points.
# 
# which basically helps to normalise the data within a particular range. Sometimes, it also helps in speeding up the calculations in an algorithm.

# In[46]:


pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
results = []
names = []
for name, model in pipelines:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=5)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)

results_df = pd.DataFrame(results, index=names,                           columns='CV1 CV2 CV3 CV4 CV5'.split())
results_df['CV Mean'] = results_df.iloc[:,0:n_splits].mean(axis=1)
results_df['CV Std Dev'] = results_df.iloc[:,0:n_splits].std(axis=1)
results_df.sort_values(by='CV Mean', ascending=False)*100


# As We Can See That Scaling of data has improved accuracy of few models but LDA remains same with 89.90 +- 0.28 accuracy

# #### Lets Try our Luck with the Ensembels
# ##### What are Ensembels ?
# Ensembels are Alogrithms that combine diverse set of learners (individual models) together to improvise on the stability and predictive power of the model.
# 
# which means it will take the score of all small models and then collectively learn to get a single best model
# 
# 
# SO LETS TRY!!
# 

# In[47]:


ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostClassifier())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier())])))  
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))
results = []
names = []
for name, model in ensembles:
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=5)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
#     print(msg)
    
results_df = pd.DataFrame(results, index=names,                           columns='CV1 CV2 CV3 CV4 CV5'.split())
results_df['CV Mean'] = results_df.iloc[:,0:n_splits].mean(axis=1)
results_df['CV Std Dev'] = results_df.iloc[:,0:n_splits].std(axis=1)
results_df.sort_values(by='CV Mean', ascending=False)*100


# ### As From This Method We Have Got A Best method for our problem with Accuracy of 90.84 +- .32
# 
# So Now Scaled Gradient Boosting Method Will be Used for or Machine Learning And Prediction 

# # Preparing And Training the Model
# #### (Gradient Boosting Classifier)
# 
# The statistical framework which use boosting as a numerical optimization problem where the objective is to minimize the loss of the model by adding weak learners using a gradient descent like procedure.
# 
# This class of algorithms were described as a stage-wise additive model. This is because one new weak learner is added at a time and existing weak learners in the model are frozen and left unchanged.
# The generalization allowed arbitrary differentiable loss functions to be used, expanding the technique beyond binary classification problems to support regression, multi-class classification and more.
# 
# ##### How Gradient Boosting Works
# Gradient boosting involves three elements:
# 
# A loss function to be optimized.
# 
# A weak learner to make predictions.
# 
# An additive model to add weak learners to minimize the loss function
# 
# -----------------------------------------------------------------------------
# 
# A gradient descent procedure is used to minimize the loss when adding trees.
# 
# Decision trees are used as the weak learner in gradient boosting.
# 
# Trees are added one at a time, and existing trees in the model are not changed.

# In[48]:


# Scaling of Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Training of Model 

# In[49]:


model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
# n_estimators - The number of boosting stages to perform.
# learning_rate - learning rate shrinks the contribution of each tree by value of learning_rate provided.
# max_features - The number of features to consider when looking for the best split.
# max_depth - The maximum depth limits the number of nodes in the tree.
# random_state - random_state is the seed used by the random number generator


# In[50]:


# Fitting The Model
model.fit(X_train, y_train)


# ## Prediction And Accuracy Of Model 

# In[51]:


# Predicting from The Model
predictions = model.predict(X_test)


print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report :")
print(classification_report(y_test, predictions))


# ### Confusion Matrix - 
# 
# [TP FP]
# 
# [FN TN]
# 
# True Positive: You predicted positive and it’s true.
# 
# True Negative: You predicted negative and it’s true.
# 
# False Positive: (Type 1 Error) You predicted positive and it’s false.
# 
# False Negative: (Type 2 Error) You predicted negative and it’s false.
# 
# 
# ### Classification Report - 
# class 0 - Not Subscribed
# 
# class 1 - Subscribed
# 
# precision - 
# Precision is the ability of a classiifer not to label an instance positive that is actually negative. For each class it is defined as as the ratio of true positives to the sum of true and false positives. Said another way, “for all instances classified positive, what percent was correct?”
# 
# recall - 
# Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives. Said another way, “for all instances that were actually positive, what percent was classified correctly?”
# 
# f1 score - 
# The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.
# 
# support - 
# Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. Support doesn’t change between models but instead diagnoses the evaluation process.

# In[52]:


score = model.score(X_test,y_test)
score


# #  Model is 90.28% Accurate and thats Not Bad!!!

# In[53]:


sns.jointplot(x=predictions,y=y_test)

