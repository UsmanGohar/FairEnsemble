#!/usr/bin/env python
# coding: utf-8

# # **ADULT CENSUS INCOME PREDICTION**

# In this Notebook ,I am going to show my work on the Adult Census Income dataset and how boosting can help increase our accuracy(even by small amount) ,especially using the XGBoost algorithm.
# 
# I am a beginner ,so please keep in mind that I may have not gone into greater depths of analysis and model tuning.I would greatly appreciate any suggestions so as to improve my model.

# ## Overview

# *Prediction task is to determine whether a person makes over 50K or less in a year.*
#  
#  **Attributes**:
# 
#   *income*: >50K, <=50K
# 
#   *age*: continuous
# 
#   *workclass*: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov,
#   State-gov, Without-pay, Never-worked
# 
#   *fnlwgt*: continuous
# 
#   *education*: Bachelors, Some-college, 11th, HS-grad, Prof-school, 
#   Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, 
#   Doctorate, 5th-6th, Preschool
#   education-num: continuous
#   
#   *marital-status*: Married-civ-spouse, Divorced, Never-married, Separated, 
#   Widowed, Married-spouse-absent, Married-AF-spouse
# 
#   *occupation*: Tech-support, Craft-repair, Other-service, Sales, 
#   Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct,
#   Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, 
#   Protective-serv, Armed-Forces
# 
#   *relationship*: Wife, Own-child, Husband, Not-in-family, Other-relative, 
#   Unmarried
# 
#   *race*: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
# 
#   *sex*: Female, Male
# 
#   *capital-gain*: continuous
# 
#   *capital-loss*: continuous
# 
#   *hours-per-week*: continuous
# 
#   *native-country*: United-States, Cambodia, England, Puerto-Rico, 
#   Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, 
#   Greece, South, China, Cuba, Iran, Honduras, Philippines, 
#   Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland,
#   France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti,
#   Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand,
#   Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands

# ## Load libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import scikitplot as skplt


# ## Load Data

# In[ ]:


dataset=pd.read_csv("../input/adult.csv")


# ### *Check for null values and show the datatypes*

# In[ ]:


print(dataset.isnull().sum())
print(dataset.dtypes)


# ### *Look at data*

# In[ ]:


dataset.head()


# In[ ]:


#removing '?' containing rows
dataset = dataset[(dataset != '?').all(axis=1)]
#label the income objects as 0 and 1
dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1})


# In this dataset ,just by looking at the columns "education" and "education.num" you could say that they bot convey the same meaning,one just specifies the degree name and the other specifies a numerical value for that degree ,we could drop any one of these. Numerical data is preferrable so lets keep "education.num" and we can remove "education".The same could be said about "marital.status" and "relationship",here, generally one would assume income levels whether a person is married or not."relationship" indirectly conveys the same husband ,wife indirectly means the person is married others like child,etc says that person is single.Hence we can drop any one of these.
# 
# I will prove these in the following sections.

# ## Analyze data

# In[ ]:


sns.catplot(x='education.num',y='income',data=dataset,kind='bar',height=6)
plt.show()


# Higher the value ,higher the probability of income greater than 50k(Obviously!)

# In[ ]:


#explore which country do most people belong
plt.figure(figsize=(38,14))
sns.countplot(x='native.country',data=dataset)
plt.show()


# Here most people are from the USA,so we can drop this column as it creates unnecessary bias.

# In[ ]:


#marital.status vs income
sns.factorplot(x='marital.status',y='income',data=dataset,kind='bar',height=8)
plt.show()


# In[ ]:


#relationship vs income
sns.factorplot(x='relationship',y='income',data=dataset,kind='bar',size=7)
plt.show()


# Aha! You can clearly see that "relationship" and "marital.status", look similar i.e. tell us the same thing.
# 
# I will prove this now,first let us do some changes to data so it is simpler to understand.

# ## Feature Engineering

# In[ ]:


#we can reformat marital.status values to single and married
dataset['marital.status']=dataset['marital.status'].map({'Married-civ-spouse':'Married', 'Divorced':'Single', 'Never-married':'Single', 'Separated':'Single', 
'Widowed':'Single', 'Married-spouse-absent':'Married', 'Married-AF-spouse':'Married'})


#  ### *Label encoding*

# In[ ]:


for column in dataset:
    enc=LabelEncoder()
    if dataset.dtypes[column]==np.object:
         dataset[column]=enc.fit_transform(dataset[column])


# ### *Correlation using heatmap*

# In[ ]:


plt.figure(figsize=(14,10))
sns.heatmap(dataset.corr(),annot=True,fmt='.2f')
plt.show()


#  As we can see from the heatmap "education" and "education.num" are highly correlated, same can be said about the "marital.status" and "relationship" ,thus,we can drop "relationship" and "education".

# In[ ]:


dataset=dataset.drop(['relationship','education'],axis=1)


#  We can also drop "occupation" as "workclass" is sufficient.
# Furthermore,"fnlwgt" is not useful to us as it refers to only the sampling in the census conducted and has no practical effect on the label.Also we  drop the "native.country" as more are from single country(USA) which can cause bias.

# In[ ]:


dataset=dataset.drop(['occupation','fnlwgt','native.country'],axis=1)


#  Dataset after preprocessing

# In[ ]:


print(dataset.head())


# Split the dataset into predictors and target and make training and testing sets

# In[ ]:


X=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1]
print(X.head())
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,shuffle=False)


#  We use cross validation(CV) to select which model to use.In k-fold CV a model is trained using k-1  of the folds as training data.Then the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).

# In[ ]:


clf=GaussianNB()
cv_res=cross_val_score(clf,x_train,y_train,cv=10)
print(cv_res.mean()*100)


# In[ ]:


clf=DecisionTreeClassifier()
cv_res=cross_val_score(clf,x_train,y_train,cv=10)
print(cv_res.mean()*100)


# In[ ]:


clf=RandomForestClassifier(n_estimators=100)
cv_res=cross_val_score(clf,x_train,y_train,cv=10)
print(cv_res.mean()*100)


# We use random forest as it is known for its robustness and less sensitivity.It is a bagging algorithm.

# ## Model Tuning

#   Model Tuning is defined as tuning the parameters(hyperparameters) of our model so as to increase the performance of our classifier

# ### Gridsearch

#  Instead of manually changing each parameter and comparing results,sklearn provides us with Gridsearch which uses crossvalidation to internally check all the parameters and compare results and gives us the best parameters as output.
#  This is a time taking process.

# In[ ]:


'''
---USED GRIDSEARCH FOR HYPERPARAMETER TUNING-----
clf=RandomForestClassifier()
kf=KFold(n_splits=3)
max_features=np.array([1,2,3,4,5])
n_estimators=np.array([25,50,100,150,200])
min_samples_leaf=np.array([25,50,75,100])
param_grid=dict(n_estimators=n_estimators,max_features=max_features,min_samples_leaf=min_samples_leaf)
grid=GridSearchCV(estimator=clf,param_grid=param_grid,cv=kf)
gres=grid.fit(x_train,y_train)
print("Best",gres.best_score_)
print("params",gres.best_params_)

----------------OUTPUT------------------------
Best 0.810471100554236
params {'max_features': 5, 'min_samples_leaf': 50, 'n_estimators': 50}
'''


# #### Note:- This  cell is commented out as it takes long time to compute(15-20 mins)

# ## Finalize the Model

# ### *Fit the model with tuned parameters*

# In[ ]:


clf=RandomForestClassifier(n_estimators=50,max_features=5,min_samples_leaf=50)
clf.fit(x_train,y_train)


# ### *Make predictions*

# In[ ]:


pred=clf.predict(x_test)
pred


# ### *Evaluation metrics*

# In[ ]:


print("Accuracy: %f " % (100*accuracy_score(y_test, pred)))


# ## **XGBoost**

# Random forest follows the concept of bagging,the other method is boosting.In Boosting algorithms each classifier is trained on data, taking into account the previous classifiers’ success. After each training step, the weights are redistributed. Misclassified data increases its weights to emphasise the most difficult cases. In this way, subsequent learners will focus on them during their training.
# 
# Few boosting algorihtms are adaboost,gradientboosting,XGBoost.XGBoost is one of the most popular machine learning algorithm these days. 
# 
# [XGBoost](https://xgboost.readthedocs.io/en/latest/) (Extreme Gradient Boosting) belongs to a family of boosting algorithms and uses the gradient boosting (GBM) framework at its core.
# 
# To install XGBoost on your system using [conda](https://anaconda.org/conda-forge/xgboost).

# In[ ]:


import xgboost as xgb
xgb.__version__


# As we have already done feature engineering etc. We can move on to tune the hyperparameters.Parameters can be found in the XGboost documetation.
# 
# But before that we have to convert data into Dmatrix (XGBoost uses data only in this format).

# In[ ]:


dmat=xgb.DMatrix(x_train,y_train)
test_dmat=xgb.DMatrix(x_test)


# ## **Bayesian Optimization**

# Grid and random search are completely uninformed by past evaluations, and as a result, often spend a significant amount of time evaluating “bad” hyperparameters.But bayesain optimization are informed of previous evaluations.A very lucid explanation of it,that has helped me a lot, is given by William Koehrsen [here](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f).
# The library that i have used here is scikit-optimization([skopt](https://scikit-optimize.github.io/)).This library provides us with the BayesSearchCV method.

# In[ ]:


from skopt import BayesSearchCV 
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

params={'min_child_weight': (0, 10),
        'max_depth': (0, 30),
        'subsample': (0.5, 1.0, 'uniform'),
        'colsample_bytree': (0.5, 1.0, 'uniform'),
        'n_estimators':(50,100),
        'reg_lambda':(1,100,'log-uniform'),
        }

bayes=BayesSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic',eval_metric='error',eta=0.1),search_spaces=params,n_iter=50,scoring='accuracy',cv=5)
res=bayes.fit(x_train,y_train)
print(res.best_params_)
print(res.best_score_)


# XGBoost has an inbuilt cvmethod which helps us to find the rounds using early stopping to prevent overfitting

# In[ ]:


final_p={'colsample_bytree': 1.0, 'max_depth': 3, 'min_child_weight': 0,'subsample': 0.5,'reg_lambda': 100.0,'objective':'binary:logistic','eta': 0.1,'n_estimators':50, "silent": 1}
cv_res=xgb.cv(params=final_p,dtrain=dmat,num_boost_round=1000,early_stopping_rounds=100,metrics=['error'],nfold=5)
cv_res.tail()


# Now after finding the rounds required we train our final model with the tuned parameters and the rounds.We then validate it on our test set

# In[ ]:


final_clf=xgb.train(params=final_p,dtrain=dmat,num_boost_round=837)
pred=final_clf.predict(test_dmat)
print(pred)
pred[pred > 0.5 ] = 1
pred[pred <= 0.5] = 0
print(pred)
print(accuracy_score(y_test,pred)*100)


# As you can see we got a difference of  almost  0.6% ,this is very small but it may help in winning competitions and also I have not taken all parameters into the tuning process(gamma,alpha,etc.).Thus I can say with further tuning we can improve this model even more.I would VERY much appreciate any suggestions for improvments that I could make.
