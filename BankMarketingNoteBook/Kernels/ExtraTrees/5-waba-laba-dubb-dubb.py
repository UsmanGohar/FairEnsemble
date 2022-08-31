#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


bank = pd.read_csv('../input/bank-additional-full.csv', sep = ';')
bank.head(10)


# In[ ]:


len(bank)


# In[ ]:


bank.info()


# In[ ]:


bank.columns


# In[ ]:


print(bank['job'].unique())
print(bank['marital'].unique())
print(bank['education'].unique())
print(bank['housing'].unique())
print(bank['loan'].unique())
print(bank['contact'].unique())
print(bank['month'].unique())
print(bank['day_of_week'].unique())
print(bank['campaign'].unique())


# In[ ]:


print('Min age: ', bank['age'].max())
print('Max age: ', bank['age'].min())
bank.isnull().any()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(30, 12)
sns.countplot(x = 'age',  palette="rocket", data = bank)
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Distribution', fontsize=15)
sns.despine()


# In[ ]:


### finding outliers based on age
# Method 1- Z score
threshold = 3
ys= bank['age']
mean_y = np.mean(ys)
stdev_y = np.std(ys)
fence_low  = mean_y-3*stdev_y
fence_high = mean_y+3*stdev_y
bank1= bank.loc[(bank['age'] > fence_low) & (bank['age'] < fence_high)]
print('Zscore lower bound and upper bound are', fence_low, 'and', fence_high, 'respectively')


# Method 2- IQR method
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    print('IQR lower bound and upper bound are', fence_low, 'and', fence_high, 'respectively')
    return df_out
bank2= remove_outlier(bank, 'age')

# we will use IQR here


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(30, 12)
sns.countplot(x = 'age',  palette="rocket", data = bank1)
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Distribution', fontsize=15)
sns.despine()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(30, 12)
sns.countplot(x = 'age',  palette="rocket", data = bank2)
ax.set_xlabel('Age', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Age Distribution', fontsize=15)
sns.despine()


# In[ ]:



#function to creat group of ages, this helps because we have 78 differente values here
bank2= bank2.copy()
def age(dataframe):
    q1 = dataframe['age'].quantile(0.25)
    q2 = dataframe['age'].quantile(0.50)
    q3 = dataframe['age'].quantile(0.75)
    dataframe.loc[(dataframe['age'] <= q1), 'age'] = 1
    dataframe.loc[(dataframe['age'] > q1) & (dataframe['age'] <= q2), 'age'] = 2
    dataframe.loc[(dataframe['age'] > q2) & (dataframe['age'] <= q3), 'age'] = 3
    dataframe.loc[(dataframe['age'] > q3), 'age'] = 4 
    print (q1, q2, q3)
    return dataframe
age(bank2);

bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'job',  palette="rocket", data = bank1)
ax.set_xlabel('Type of job', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('job Distribution', fontsize=15)
sns.despine()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[ ]:


labelencoder_X = LabelEncoder()


# In[ ]:


labelencoder_X.fit(bank2['job'])
bank2['job'] = labelencoder_X.transform(bank2['job'])
bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
sns.countplot(x = 'marital',  palette="rocket", data = bank2)
ax.set_xlabel('Marital status', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('Marital status', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['marital'])
bank2['marital'] = labelencoder_X.transform(bank2['marital'])
bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'education',  palette="rocket", data = bank2)
ax.set_xlabel('Highest education', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('Highest education', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['education'])
bank2['education'] = labelencoder_X.transform(bank2['education'])
bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'default',  palette="rocket", data = bank2)
ax.set_xlabel('Default on repayment', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('Default on repayment', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['default'])
bank2['default'] = labelencoder_X.transform(bank2['default'])
bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'loan',  palette="rocket", data = bank2)
ax.set_xlabel('Taken a loan?', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('loan', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['loan'])
bank2['loan'] = labelencoder_X.transform(bank2['loan'])
bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'housing',  palette="rocket", data = bank2)
ax.set_xlabel('Owns a house?', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('Housing', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['housing'])
bank2['housing'] = labelencoder_X.transform(bank2['housing'])
bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'contact',  palette="rocket", data = bank2)
ax.set_xlabel('How was the person contacted', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('contact', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['contact'])
bank2['contact'] = labelencoder_X.transform(bank2['contact'])
bank2.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'month',  palette="rocket", data = bank2)
ax.set_xlabel('Which month was the person contacted', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('Month', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['month'])
bank2['month'] = labelencoder_X.transform(bank2['month'])
bank2.tail(20)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'day_of_week',  palette="rocket", data = bank2)
ax.set_xlabel('On which day of the week was the person contacted', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('day_of_week', fontsize=25)
sns.despine()


# In[ ]:


labelencoder_X.fit(bank2['day_of_week'])
bank2['day_of_week'] = labelencoder_X.transform(bank2['day_of_week'])
bank2.tail(20)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(30, 12)
sns.countplot(x = 'duration',  palette="rocket", data = bank)
ax.set_xlabel('duration of call', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Duration Distribution', fontsize=15)
sns.despine()


# In[ ]:


def remove_outlier_upper(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > 0) & (df_in[col_name] < fence_high)]
    print('IQR lower bound and upper bound are 0 and', fence_high, 'respectively')
    return df_out
bank2= remove_outlier_upper(bank2, 'duration')


# In[ ]:


bank2['duration'].head()


# In[ ]:


#function to creat group of duration, this helps because we a lot of values
bank2= bank2.copy()
def duration(dataframe):
    q1 = dataframe['duration'].quantile(0.25)
    q2 = dataframe['duration'].quantile(0.50)
    q3 = dataframe['duration'].quantile(0.75)
    dataframe.loc[(dataframe['duration'] <= q1), 'duration'] = 1
    dataframe.loc[(dataframe['duration'] > q1) & (dataframe['duration'] <= q2), 'duration'] = 2
    dataframe.loc[(dataframe['duration'] > q2) & (dataframe['duration'] <= q3), 'duration'] = 3
    dataframe.loc[(dataframe['duration'] > q3), 'duration'] = 4 
    print (q1, q2, q3)
    return dataframe
duration(bank2)

bank2['duration'].head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
sns.countplot(x = 'campaign',  palette="rocket", data = bank2)
ax.set_xlabel('campaign', fontsize=25)
ax.set_ylabel('Count', fontsize=25)
ax.set_title('campaign', fontsize=25)
sns.despine()


# In[ ]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > 0) & (df_in[col_name] < fence_high)]
    print('IQR lower bound and upper bound are 0 and', fence_high, 'respectively')
    return df_out
bank2= remove_outlier(bank2, 'campaign')


# In[ ]:


len(bank2)


# In[ ]:


bank2.head()


# In[ ]:


bank2['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)


# In[ ]:


bank2.head()


# In[ ]:


print(bank2['pdays'].unique())


# In[ ]:


bank2.loc[(bank2['pdays'] == 999), 'pdays'] = 1
bank2.loc[(bank2['pdays'] > 0) & (bank2['pdays'] <= 10), 'pdays'] = 2
bank2.loc[(bank2['pdays'] > 10) & (bank2['pdays'] <= 20), 'pdays'] = 3
bank2.loc[(bank2['pdays'] > 20) & (bank2['pdays'] != 999), 'pdays'] = 4 
bank2.head()


# In[ ]:


print(bank2['emp.var.rate'].unique())


# In[ ]:


print(bank2['cons.price.idx'].unique())


# In[ ]:


print(bank2['cons.conf.idx'].unique())


# In[ ]:


print(bank2['euribor3m'].unique())


# In[ ]:


bank2.loc[(bank2['euribor3m'] < 1), 'euribor3m'] = 1
bank2.loc[(bank2['euribor3m'] > 1) & (bank2['euribor3m'] <= 2), 'euribor3m'] = 2
bank2.loc[(bank2['euribor3m'] > 2) & (bank2['euribor3m'] <= 3), 'euribor3m'] = 3
bank2.loc[(bank2['euribor3m'] > 3) & (bank2['euribor3m'] <= 4), 'euribor3m'] = 4
bank2.loc[(bank2['euribor3m'] > 4), 'euribor3m'] = 5
bank2.head(20)


# In[ ]:


bank_final= bank2.copy()
y = bank_final['y']
bank_final.drop(['y'],axis=1,inplace=True)
bank_final.shape


# In[ ]:


# Now lets start applying different models on our data
bank_final.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bank_final, y, test_size = 0.25, random_state = 0)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:


# Lets first standardize our data i.e. transform the data in a way that the variance is unitary and that the mean of the series is 0.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


# 1. KNN classifier
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

#Neighbors
neighbors = np.arange(0,25)

#Create empty list that will hold cv scores
cv_scores = []

#Perform 10-fold cross validation on training set for odd values of k:
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=123)
    scores = model_selection.cross_val_score(knn, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    print("k=%d %f" % (k_value, scores.mean()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %f" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)

print(accuracy_score(y_test, knnpred)*100)


# In[ ]:


# 2. Logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() 
logmodel.fit(X_train,y_train)
logpred = logmodel.predict(X_test)

print(accuracy_score(y_test, logpred)*100)


# In[ ]:


# 3. Linear classifiers: Support Vector Machines (kernel: sigmoid)

from sklearn.svm import SVC
for this_gamma in [.01, 1.0, 10.0]:
    svc= SVC(kernel = 'sigmoid', gamma= this_gamma)
    svc.fit(X_train, y_train)
    svcpred = svc.predict(X_test)
    print(accuracy_score(y_test, svcpred)*100)


# In[ ]:


# 4. Linear classifiers: Support Vector Machines (kernel: Radial Basis Function)

from sklearn.svm import SVC
for this_gamma in [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1, 1.0, 10.0]:
    svc= SVC(kernel = 'rbf', gamma= this_gamma)
    svc.fit(X_train, y_train)
    svcpred = svc.predict(X_test)
    print(this_gamma, accuracy_score(y_test, svcpred)*100)


# In[ ]:


# 5. Linear classifiers: Support Vector Machines (kernel: Linear)

from sklearn.svm import SVC
for this_gamma in [.01, 1.0, 10.0]:
    svc= SVC(kernel = 'linear', gamma= this_gamma)
    svc.fit(X_train, y_train)
    svcpred = svc.predict(X_test)
    print(accuracy_score(y_test, svcpred)*100)
# So we learned that value of gamma does not affect the accuracy of SVM when using a Linear kernel


# In[ ]:


# 6. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtree.fit(X_train, y_train)
dtreepred = dtree.predict(X_test)

print(accuracy_score(y_test, dtreepred)*100)


# In[ ]:


# 7. Random Forest Classifier(n=200)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)

print(accuracy_score(y_test, rfcpred)*100)


# In[ ]:


# 8. Random Forest Classifier(n=1000)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 1000)#criterion = entopy,gini
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)

print(accuracy_score(y_test, rfcpred)*100)


# In[ ]:


# 9. naive bayes classifier

from sklearn.naive_bayes import GaussianNB
gaussiannb= GaussianNB()
gaussiannb.fit(X_train, y_train)
gaussiannbpred = gaussiannb.predict(X_test)
probs = gaussiannb.predict(X_test)

print(accuracy_score(y_test, gaussiannbpred)*100)


# In[ ]:


# 10. Gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
gbkpred = gbk.predict(X_test)

print(accuracy_score(y_test, gbkpred)*100)


# In[ ]:


# 11. XGBoost Classifier

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)

print(accuracy_score(y_test, xgbprd)*100)


# In[ ]:


# Now lets find the cross validation scores for top 2 classifiers-
GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
XGB = (cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10).mean())
print(GBKCV,XGB)


# In[ ]:





# In[ ]:




