#!/usr/bin/env python
# coding: utf-8

# # Adult census income

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing


# In[ ]:


pd.set_option('display.max_columns',500)


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:



data1 = pd.read_csv("../input/adult-census-income/adult.csv")
x=data1.drop('income',axis=1)
y=data1['income']


# In[ ]:


x,y = train_test_split(data1,random_state=7)


# In[ ]:





# In[ ]:


x.head()


# In[ ]:


x.groupby('income').size()


# In[ ]:


x.info()


# In[ ]:


x.describe()


# In[ ]:


x.shape


# In[ ]:


(x=='?').sum()


# In[ ]:


((x=='?').sum()*100/32561).round(2)


# In[ ]:


((y=='?').sum()*100/32561).round(2)


# In[ ]:


#data[data[::] != '?']
x = x[(x['workclass']!='?')& (x['occupation']!='?') & (x['native.country']!='?')]


# In[ ]:


#data[data[::] != '?']
y = y[(y['workclass']!='?')& (y['occupation']!='?') & (y['native.country']!='?')]


# In[ ]:


(x=='?').sum()


# In[ ]:


(y=='?').sum()


# In[ ]:


x.info()


# In[ ]:


sns.pairplot(x)


# In[ ]:


correlation = x.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
#sns.heatmap(x.select_dtypes([object]), annot=True, annot_kws={"size": 7})


# In[ ]:



name = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']

for c in name:
    sns.boxplot(x=x[c],data=x)

    plt.show()


# In[ ]:


x.select_dtypes(['object']).head()


# In[ ]:


x['income'].unique()


# In[ ]:


x['workclass'].unique()


# In[ ]:


x['education'].unique()


# In[ ]:


x['occupation'].unique()


# In[ ]:


x['sex'].unique()


# In[ ]:


x['workclass'].unique()


# In[ ]:


x['native.country'].unique()


# In[ ]:


y['native.country'].unique()


# In[ ]:


y.replace(['South','Hong'],['South korea','Hong kong'],inplace=True)


# In[ ]:


x.replace(['South','Hong'],['South korea','Hong kong'],inplace=True)


# In[ ]:


x['native.country'].unique()


# In[ ]:


x['net_capital']=x['capital.gain']-x['capital.loss']
x.drop(['capital.gain','capital.loss'],1,inplace=True)


# In[ ]:


y['net_capital']=y['capital.gain']-y['capital.loss']
y.drop(['capital.gain','capital.loss'],1,inplace=True)


# In[ ]:


y.head()


# In[ ]:


x.head()


# In[ ]:



name = ['age','fnlwgt','education.num','net_capital','hours.per.week']
for c in name:
    sns.distplot(x[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[ ]:



name = ['age','fnlwgt','education.num','net_capital','hours.per.week']
for c in name:
    sns.distplot(y[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[ ]:


d = x.loc[:,['age','fnlwgt','education.num','net_capital','hours.per.week']]


# In[ ]:


d1 = y.loc[:,['age','fnlwgt','education.num','net_capital','hours.per.week']]


# In[ ]:


d.head()


# In[ ]:


d1.head()


# In[ ]:


from sklearn.preprocessing import Normalizer


# In[ ]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d=pd.DataFrame(pt.fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[ ]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d1=pd.DataFrame(pt.fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[ ]:



d=pd.DataFrame(Normalizer().fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[ ]:



d1=pd.DataFrame(Normalizer().fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[ ]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d=pd.DataFrame(pt.fit_transform(d),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# In[ ]:


pt = preprocessing.QuantileTransformer(output_distribution='normal')
d1=pd.DataFrame(pt.fit_transform(d1),columns=['age','fnlwgt','education.num','net_capital','hours.per.week'])


# quantile
# normalizer
# quantile

# In[ ]:


name = ['age','fnlwgt','education.num','net_capital','hours.per.week']

for c in name:
    sns.distplot(d[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[ ]:


name = ['age','fnlwgt','education.num','net_capital','hours.per.week']

for c in name:
    sns.distplot(d1[c], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.show()


# In[ ]:


sns.heatmap(x.corr(),annot = True)


# In[ ]:


sns.heatmap(y.corr(),annot = True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for c in x.select_dtypes(['object']).columns:
    
        x[c]=le.fit_transform(x[c])
        


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for c in y.select_dtypes(['object']).columns:
    
        y[c]=le.fit_transform(y[c])
        


# In[ ]:


d.head()


# In[ ]:


d1.head()


# In[ ]:


x.drop(['age','fnlwgt','education.num','net_capital','hours.per.week'],1,inplace=True)


# In[ ]:


y.drop(['age','fnlwgt','education.num','net_capital','hours.per.week'],1,inplace=True)


# In[ ]:


x=pd.merge(x,d,left_index=True,right_index=True)


# In[ ]:


y=pd.merge(y,d,left_index=True,right_index=True)


# In[ ]:


x.head()


# In[ ]:


x.shape


# In[ ]:


y.head()


# In[ ]:


#pca
#treebaseapproach
#rfe


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(x.corr(),annot = True)


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(y.corr(),annot = True)


# In[ ]:


x_train = x.drop('income',1)
y_train = x['income']


# In[ ]:


x_test = x.drop('income',1)
y_test = x['income']


# In[ ]:


from sklearn.feature_selection import RFECV


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


rfe = RFECV(estimator = DecisionTreeClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))


plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


rfe = RFECV(estimator = RandomForestClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


rfe = RFECV(estimator = AdaBoostClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


rfe = RFECV(estimator = GradientBoostingClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
 
# Feature importance values from Random Forests
rf = RandomForestClassifier(n_jobs=-1, random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = RandomForestClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[ ]:




rf = AdaBoostClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = AdaBoostClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[ ]:


rf = GradientBoostingClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = GradientBoostingClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[ ]:


rf = xgb.XGBClassifier(random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = xgb.XGBClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# # one hot

# In[ ]:


x['income']=le.fit_transform(x['income'])


# In[ ]:


x.head()


# In[ ]:


for c in x.select_dtypes(['object']).columns:
    cont = pd.get_dummies(x[c],prefix='Contract')
    x = pd.concat([x,cont],axis=1)
    x.drop(c,1,inplace=True)
    


# In[ ]:


for c in y.select_dtypes(['object']).columns:
    cont = pd.get_dummies(y[c],prefix='Contract')
    y = pd.concat([y,cont],axis=1)
    y.drop(c,1,inplace=True)
    


# In[ ]:


x.head()


# In[ ]:


x.shape


# In[ ]:


x_train = x.drop('income',1)
y_train = x['income']


# In[ ]:


x_test = x.drop('income',1)
y_test = x['income']


# In[ ]:


rfe = RFECV(estimator = DecisionTreeClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))


plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


rfe = RFECV(estimator = RandomForestClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


rfe = RFECV(estimator = AdaBoostClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


rfe = RFECV(estimator = GradientBoostingClassifier(random_state=1) , cv=4, scoring = 'accuracy')
rfe = rfe.fit(x_train,y_train)

col = x_train.columns[rfe.support_]

acc = accuracy_score(y_test,rfe.estimator_.predict(x_test[col]))

print('Number of features selected: {}'.format(rfe.n_features_))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,rfe.estimator_.predict(x_test[col])))

plt.figure()
plt.xlabel('k')
plt.ylabel('CV accuracy')
plt.plot(np.arange(1, rfe.grid_scores_.size+1), rfe.grid_scores_)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
 
# Feature importance values from Random Forests
rf = RandomForestClassifier(n_jobs=-1, random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = RandomForestClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[ ]:




rf = AdaBoostClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = AdaBoostClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[ ]:


rf = GradientBoostingClassifier( random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = GradientBoostingClassifier( random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))


# In[ ]:


rf = xgb.XGBClassifier(random_state=1)
rf.fit(x_train, y_train)
feat_imp = rf.feature_importances_

cols = x_train.columns[feat_imp >= 0.01]
est_imp = xgb.XGBClassifier(random_state=1)
est_imp.fit(x_train[cols], y_train)
 
# Test accuracy
acc = accuracy_score(y_test, est_imp.predict(x_test[cols]))
print('Number of features selected: {}'.format(len(cols)))
print('Test Accuracy {}'.format(acc))
print(confusion_matrix(y_test,est_imp.predict(x_test[cols])))

