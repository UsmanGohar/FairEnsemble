#!/usr/bin/env python
# coding: utf-8

# # Data Import

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from pprint import pprint
from scipy import stats
from sklearn.preprocessing import StandardScaler


# In[ ]:


dataset = pd.read_csv("../input/bank-marketing/bank-additional-full.csv", sep=';')
dataset.head()


# # Data Cleaning and Preprocessing

# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().sum()


# There are no missing values in the dataset.

# In[ ]:


dataset.describe()


# In[ ]:


dataset.hist(bins = 15, figsize = (10,10), xlabelsize = 0.1, ylabelsize = 0.1)
plt.show()


# In[ ]:


dataset.pdays.value_counts(normalize=True)


# pdays variable has the value 999 96% of the time. The variable gives no information since its variance is very low. It is better to drop this variable.

# In[ ]:


sns.catplot(x='default',hue='y',kind='count',data=dataset)


# In[ ]:


pd.crosstab(dataset['default'], dataset.y)


# There are only 3 customers which we know for sure that they have a loan in default. Again, this variable gives no information and it will be dropped in a later stage.

# In[ ]:


dataset.y.value_counts(normalize=True)


# Looking at the proportion of the classes we like to predict, we see that the dataset is imbalanced. We will take care of this problem with oversampling method.

# In[ ]:


colors = ["#0101DF", "#DF0101"]

sns.countplot('y', data=dataset, palette=colors)
plt.title('Deposit Distributions \n (0: No || 1: Yes)', fontsize=14)


# # Correlation Matrix

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(dataset.corr(),square=True,annot=True,cmap= 'twilight_shifted')


# There are highly correlated variables in the dataset. It would be reasonable to perform feature selection but most of the algorithms I use have their own feature selection, I omit to ddo that.

# ## Standardization

# In[ ]:


# make a copy of dataset to scaling
bank_scale=dataset.copy()

# remove 'pdays' and 'default' columns
bank_scale= bank_scale.drop(['pdays', 'default'], axis=1)

bank_scale.y.replace(('yes', 'no'), (1, 0), inplace=True)

# standardization for just numerical variables 
categorical_cols= ['job','marital', 'education',  'housing', 'loan', 'contact', 'month', 'day_of_week','poutcome','y']
feature_scale=[feature for feature in bank_scale.columns if feature not in categorical_cols]

scaler=StandardScaler()
scaler.fit(bank_scale[feature_scale])


# In[ ]:


scaled_data = pd.concat([bank_scale[categorical_cols].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(bank_scale[feature_scale]), columns=feature_scale)],
                    axis=1)

categorical_cols1= ['job','marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week','poutcome']
scaled_data= pd.get_dummies(scaled_data, columns = categorical_cols1, drop_first=True)
scaled_data.head()


# ## Train/Test Split

# In[ ]:


X = scaled_data.iloc[:,1:]
Y = scaled_data.iloc[:,-0]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2)


# # Benchmark Models

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
# Tuning parameter for RF ( tuning parameters are choosen based on best parameters of RandomizedSearchCV)
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]
min_samples_split = [5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
tuning_rf = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
tuning_rf.fit(X_train,y_train)
print('Best Parameter for Random Forest', tuning_rf.best_params_, tuning_rf.best_score_)

# Tuning parameter for Tree
param_dict= {"criterion": ['gini', 'entropy'],
            "max_depth": range(1,10),
            "min_samples_split": range(1,10),
            "min_samples_leaf": range(1,5)}
tuning_tree = GridSearchCV(DecisionTreeClassifier(random_state=12),  param_grid=param_dict, cv=10, verbose=1, n_jobs=-1)
tuning_tree.fit(X_train,y_train)
print('Best Parameter for Tree', tuning_tree.best_params_, tuning_tree.best_score_)

# Xgboost Parameters
param_xgb = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6],
 'gamma':[i/10.0 for i in range(0,5)]
}
tuning_xgb = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_xgb, scoring='roc_auc',n_jobs=4, cv=5)
tuning_xgb.fit(X_train,y_train)
print('Best Parameter for XGBoost', tuning_xgb.best_params_, tuning_xgb.best_score_)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Voting Classifier\nclf1 = DecisionTreeClassifier()\nclf2 = RandomForestClassifier(random_state=1)\nclf3 = GaussianNB()\nclf4= KNeighborsClassifier()\nclf5= LinearDiscriminantAnalysis()\nclf6= XGBClassifier()\n\n# Instantiate the classfiers and make a list\nclassifiers = [LinearDiscriminantAnalysis(),\n               KNeighborsClassifier(),\n               GaussianNB(), \n               SVC(kernel='linear'),\n               DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=9,min_samples_leaf=2, random_state=12),\n               RandomForestClassifier(n_estimators=155, max_features='auto', max_depth=45, min_samples_split=10, random_state=27),\n               XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5, min_child_weight=4, gamma=0.3, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),\n               VotingClassifier(estimators = [('DTree', clf1), ('rf', clf2), ('gnb', clf3),  ('knn', clf4),('lda', clf5), ('xgb', clf6)], voting ='soft')]\n\n# Define a result table as a DataFrame\nresult_table = pd.DataFrame(columns=['classifiers', 'fpr1','tpr1','fpr','tpr','train_accuracy','test_accuracy', 'train_auc', 'test_auc', 'f1_score', 'precision','recall','confusion matrix','Report'])\n\n# Train the models and record the results\nfor cls in classifiers:\n    model = cls.fit(X_train, y_train)\n    y_train_pred = model.predict(X_train)\n    y_test_pred = model.predict(X_test)\n    \n    train_accuracy= accuracy_score(y_train, y_train_pred)\n    test_accuracy= accuracy_score(y_test, y_test_pred)\n     \n    fpr, tpr, _ = roc_curve(y_test,  y_test_pred)\n    fpr1, tpr1, _ = roc_curve(y_train,  y_train_pred)\n    \n    train_auc = roc_auc_score(y_train, y_train_pred)\n    test_auc = roc_auc_score(y_test, y_test_pred)\n    \n    f1_score= metrics.f1_score(y_test, y_test_pred)\n    precision = metrics.precision_score(y_test, y_test_pred)\n    recall = metrics.recall_score(y_test, y_test_pred)\n    \n    conf_mat= confusion_matrix(y_test,y_test_pred)\n    report=classification_report(y_test,y_test_pred, digits=3, output_dict=True)\n    \n    result_table = result_table.append({'classifiers':cls.__class__.__name__,\n                                        'fpr1':fpr1,\n                                        'tpr1':tpr1,\n                                        'fpr':fpr, \n                                        'tpr':tpr, \n                                        'train_accuracy': train_accuracy,\n                                        'test_accuracy': test_accuracy,\n                                        'train_auc':train_auc,\n                                        'test_auc':test_auc,\n                                        'f1_score': f1_score,\n                                        'precision': precision,\n                                        'recall': recall,\n                                        'confusion matrix':conf_mat,\n                                        'Report':report}, ignore_index=True)\n\n# Set name of the classifiers as index labels\nresult_table.set_index('classifiers', inplace=True)")


# In[ ]:


result_table.rename(index={'VotingClassifier':'Model Ensemble'},inplace=True)
result_table


# In[ ]:


pd.DataFrame(result_table.iloc[0,12]).transpose()


# In[ ]:


fig = plt.figure(figsize=(15,10))

for i in range(result_table.shape[0]):
    plt.plot(result_table.iloc[i,]['fpr'], 
             result_table.iloc[i,]['tpr'], 
             label="{}, AUC={:.3f}".format(result_table.index[i], result_table.iloc[i,]['test_auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()


# # Oversampling - RandomOverSampler

# In[ ]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 

from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority') 
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
  
print('After OverSampling, the shape of X_train: {}'.format(X_train_over.shape)) 
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_over.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_over == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_over == 0))) 


# In[ ]:


# Tuning parameter for RF 
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]
min_samples_split = [5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
tuning_rf = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')
tuning_rf.fit(X_train_over,y_train_over)
print('Best Parameter for Random Forest', tuning_rf.best_params_, tuning_rf.best_score_)

# Tuning parameter for Tree
param_dict= {"criterion": ['gini', 'entropy'],
            "max_depth": range(1,10),
            "min_samples_split": range(1,10),
            "min_samples_leaf": range(1,5)}
tuning_tree = GridSearchCV(DecisionTreeClassifier(random_state=12),  param_grid=param_dict, cv=10, verbose=1, n_jobs=-1)
tuning_tree.fit(X_train_over,y_train_over)
print('Best Parameter for Tree', tuning_tree.best_params_, tuning_tree.best_score_)

# Xgboost Parameters
param_xgb = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6],
 'gamma':[i/10.0 for i in range(0,5)]}
tuning_xgb = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_xgb, scoring='roc_auc',n_jobs=4, cv=5)
tuning_xgb.fit(X_train_over,y_train_over)
print('Best Parameter for XGBoost', tuning_xgb.best_params_, tuning_xgb.best_score_)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Voting Classifier\nclf1 = DecisionTreeClassifier()\nclf2 = RandomForestClassifier(random_state=1)\nclf3 = GaussianNB()\nclf4 = KNeighborsClassifier()\nclf5= LinearDiscriminantAnalysis()\nclf6= XGBClassifier()\n\n# Instantiate the classfiers and make a list\nclassifiers = [LinearDiscriminantAnalysis(),\n               KNeighborsClassifier(),\n               GaussianNB(), \n               SVC(kernel='linear'),\n               DecisionTreeClassifier(criterion='gini', max_depth=9, min_samples_split=5,min_samples_leaf=1, random_state=12),\n               RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=45, min_samples_split=5, random_state=27),\n               XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=4, min_child_weight=6, gamma=0.4, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),\n               VotingClassifier(estimators = [('DTree', clf1), ('rf', clf2), ('gnb', clf3), ('knn', clf4), ('lda', clf5), ('xgb', clf6)], voting ='soft')]\n\n# Define a result table as a DataFrame\nresult_table1 = pd.DataFrame(columns=['classifiers', 'fpr1','tpr1','fpr','tpr','train_accuracy','test_accuracy', 'train_auc', 'test_auc', 'f1_score', 'precision','recall','confusion matrix','Report'])\n\n# Train the models and record the results\nfor cls in classifiers:\n    model = cls.fit(X_train_over, y_train_over)\n    y_train_pred = model.predict(X_train_over)\n    y_test_pred = model.predict(X_test)\n    \n    train_accuracy= accuracy_score(y_train_over, y_train_pred)\n    test_accuracy= accuracy_score(y_test, y_test_pred)\n     \n    fpr, tpr, _ = roc_curve(y_test,  y_test_pred)\n    fpr1, tpr1, _ = roc_curve(y_train_over,  y_train_pred)\n    \n    train_auc = roc_auc_score(y_train_over, y_train_pred)\n    test_auc = roc_auc_score(y_test, y_test_pred)\n    \n    f1_score= metrics.f1_score(y_test, y_test_pred)\n    precision = metrics.precision_score(y_test, y_test_pred)\n    recall = metrics.recall_score(y_test, y_test_pred)\n    \n    conf_mat= confusion_matrix(y_test,y_test_pred)\n    report=classification_report(y_test,y_test_pred, digits=3, output_dict=True)\n    \n    result_table1 = result_table1.append({'classifiers':cls.__class__.__name__,\n                                        'fpr1':fpr1,\n                                        'tpr1':tpr1,\n                                        'fpr':fpr, \n                                        'tpr':tpr, \n                                        'train_accuracy': train_accuracy,\n                                        'test_accuracy': test_accuracy,\n                                        'train_auc':train_auc,\n                                        'test_auc':test_auc,\n                                        'f1_score': f1_score,\n                                        'precision': precision,\n                                        'recall': recall,\n                                        'confusion matrix':conf_mat,\n                                        'Report':report}, ignore_index=True)\n\n# Set name of the classifiers as index labels\nresult_table1.set_index('classifiers', inplace=True)")


# In[ ]:


result_table1.rename(index={'VotingClassifier':'Model Ensemble'},inplace=True)
result_table1


# In[ ]:


fig = plt.figure(figsize=(15,10))

for i in range(result_table1.shape[0]):
    plt.plot(result_table1.iloc[i,]['fpr'], 
             result_table1.iloc[i,]['tpr'], 
             label="{}, AUC={:.3f}".format(result_table1.index[i], result_table1.iloc[i,]['test_auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()


# # Results

# In[ ]:


# Baseline Model
result_table.iloc[:,[4,5,6,7,8,9,10]]


# In[ ]:


# Oversampling with RandomOverSampler
result_table1.iloc[:,[4,5,6,7,8,9,10]]


# # Feature Importance and SHAP Values

# In[ ]:


xgb = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=4, min_child_weight=6, gamma=0.4, subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1,seed=27)
model_xgb = xgb.fit(X_train_over, y_train_over)
y_train_xgb = model_xgb.predict(X_train_over)
y_test_xgb = model_xgb.predict(X_test)

print(confusion_matrix(y_test,y_test_xgb))
print(classification_report(y_test,y_test_xgb, digits=3))

print('Train accuracy: %0.3f' % accuracy_score(y_train_over, y_train_xgb))
print('Test accuracy: %0.3f' % accuracy_score(y_test, y_test_xgb))

print('Train AUC: %0.3f' % roc_auc_score(y_train_over, y_train_xgb))
print('Test AUC: %0.3f' % roc_auc_score(y_test, y_test_xgb))


# In[ ]:


import shap
expl_xgb = shap.TreeExplainer(model_xgb)
shap_xgb = expl_xgb.shap_values(X_train_over)


# In[ ]:


shap.summary_plot(shap_xgb, X_train_over, plot_type="bar")


# In[ ]:


shap.summary_plot(shap_xgb, X_train_over)


# In[ ]:


shap.initjs()
shap.force_plot(expl_xgb.expected_value, shap_xgb[1050,:], X_train_over.iloc[1050,:], link='logit')


# In[ ]:


shap.initjs()
shap.force_plot(expl_xgb.expected_value, shap_xgb[4000,:], X_train_over.iloc[4000,:], link='logit')


# In[ ]:


# base value
y_train_over.mean()


# In[ ]:


X_train_over.iloc[4000,]

