#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing - term deposit prediction

# Data Set Information:
# 
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
# 
# There are four datasets:
# 1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
# 2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
# 3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
# 4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
# The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).
# 
# The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
# 
# Attribute Information:
# 
# Input variables:
# ### bank client data:
# * 1 - age (numeric)
# * 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# * 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# * 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# * 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# * 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# * 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# * related with the last contact of the current campaign:
# * 8 - contact: contact communication type (categorical: 'cellular','telephone')
# * 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# * 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# * 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after * the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# ### other attributes:
# * 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# * 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# * 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# * 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# ### social and economic context attributes
# * 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# * 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
# * 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# * 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# * 20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
# Output variable (desired target):
# * 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
# 
# Source:  http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# 

# ## Loading libraries

# In[ ]:


# Load libraries
import numpy
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ## Load dataset

# In[ ]:


# Load dataset
filename = "../input/bank-additional-full.csv"

# dataset_all = read_csv(filename, delim_whitespace=False, header=0, sep=";")
dataset = read_csv(filename, delim_whitespace=False, header=0, sep=";")


# ### drop column duration as suggested
# > duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

# In[ ]:


# drop column duration as suggested
dataset = dataset.drop(columns=["duration"])


# ## View dataset

# In[ ]:


# shape
print("Shape:")
print(dataset.shape)
print()
print("Dataset info:")
print()
print(dataset.info())


# ### Look at class distribution

# In[ ]:


# class distribution
print(dataset.groupby("y").size())


# The classes are imbalanced. The dataset needs to be resampled.

# ## Plotting some categorical variables

# In[ ]:


# show feature month - last contact month of year and count it
sns.catplot(x="month", kind="count", data=dataset,
            order=["mar", "apr", "may", "jun", "jul", "aug", "sep", "nov",
                   "dec"])
plt.show()


# In[ ]:


# count subscribed deposit per month
ax = sns.countplot(x="month", hue="y", data=dataset,
                   order=["mar", "apr", "may", "jun", "jul", "aug", "sep",
                          "nov", "dec"])

plt.show()


# In[ ]:


# pdays: number of days that passed by after the client was 
# last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# count y
ax = sns.countplot(x="pdays", hue="y", data=dataset)
plt.show()


# It seems to me that the variable "month" and "day_of_week" have no relevance for the prediction. For some reason, many calls were made in May.
# Most calls were made for the first time (999).

# ## Prepare data

# In[ ]:


# Prepare Data
dataset_enc = dataset.copy()

# remove columns month and day_of_week
dataset_enc = dataset_enc.drop(columns=["month", "day_of_week"])


# ### Encode categorical variable

# In[ ]:


# use scikit-learn LabelEncoder to encode labels
lb = LabelEncoder()
# Convert categorical variable
dataset_enc = pd.get_dummies(dataset_enc, columns=['job'], prefix=['job'])
dataset_enc = pd.get_dummies(dataset_enc,
                             columns=['marital'], prefix=['marital'])
dataset_enc = pd.get_dummies(dataset_enc,
                             columns=['education'], prefix=['education'])
dataset_enc = pd.get_dummies(dataset_enc,
                             columns=['default'], prefix=['default'])
dataset_enc = pd.get_dummies(dataset_enc,
                             columns=['housing'], prefix=['housing'])
dataset_enc = pd.get_dummies(dataset_enc, columns=['loan'], prefix=['loan'])

# binary transform of column contact categorical: "cellular","telephone"
dataset_enc['contact'] = lb.fit_transform(dataset['contact'])
dataset_enc = pd.get_dummies(dataset_enc,
                             columns=['poutcome'], prefix=['poutcome'])

# move y at end of dataset
dataset_enc['y_class'] = dataset['y']

# remove original y column
dataset_enc = dataset_enc.drop(columns=["y"])


# ### View dataset again

# In[ ]:


dataset_enc.info()


# ### Resample dataset - Down-sample Majority Class

# In[ ]:


# Resample
dataset_majority = dataset_enc[dataset_enc.y_class == "no"]
dataset_minority = dataset_enc[dataset_enc.y_class == "yes"]

# Downsample majority class
df_majority_downsampled = resample(dataset_majority, replace=False,
                                   n_samples=4640, random_state=123)

dataset_downsampled = pd.concat([df_majority_downsampled, dataset_minority])
dataset_downsampled.y_class.value_counts()


# **we now have a balanced dataset**

# ## Evaluate Algorithms

# In[ ]:


# Evaluate Algorithms
# Split-out validation dataset
array = dataset_downsampled.values
X = array[:, 0:46]
Y = array[:, 46]
validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation =     train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


# ### Spot Check some Algorithms

# In[ ]:


# Spot Check Algorithms
models = []
models.append(("LR", LogisticRegression()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,
                                 scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f"{name} {cv_results.mean():f} {cv_results.std():f}"
    print(msg)

# Compare Algorithms
sns.boxplot(x=names, y=results)
plt.show()


# Logistic regression has the better result

# ### Spot Check some ensemble Algorithms

# In[ ]:


# Compare Algorithms

pipelines = []
pipelines.append(("AB",
                  Pipeline([("AB", AdaBoostClassifier())])))
pipelines.append(("GBM",
                  Pipeline([("GBM", GradientBoostingClassifier())])))
pipelines.append(("RF",
                  Pipeline([("RF", RandomForestClassifier())])))
pipelines.append(("ET",
                  Pipeline([("ET", ExtraTreesClassifier())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,
                                 scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f"{name} {cv_results.mean():f} {cv_results.std():f}"
    print(msg)

# Compare Algorithms
sns.boxplot(x=names, y=results)
plt.show()


# GradientBoostingClassifier has the better result.

# ## Finalize Model

# In[ ]:


# Finalize Model
model = GradientBoostingClassifier()
# prepare the model
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("Accuracy score:", accuracy_score(Y_validation, predictions))
print("Classification report")
print(classification_report(Y_validation, predictions))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
conf_mat = confusion_matrix(Y_validation, predictions)
ax = plt.subplot()
sns.heatmap(conf_mat, annot=True, ax=ax, fmt='d')
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['no', 'yes'])
ax.yaxis.set_ticklabels(['no', 'yes'])
plt.show()


# In[ ]:




