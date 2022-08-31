#!/usr/bin/env python
# coding: utf-8

# # Problem statement
# 
# We have data from a Portuguese bank on details of customers related to selling a term deposit
# The objective of the project is to help the marketing team identify potential customers who are relatively more likely to subscribe to the term deposit and this increase the hit ratio

# # Data dictionary
# 
# **Bank client data**
# * 1 - age 
# * 2 - job : type of job 
# * 3 - marital : marital status
# * 4 - education 
# * 5 - default: has credit in default? 
# * 6 - housing: has housing loan? 
# * 7 - loan: has personal loan?
# * 8 - balance in account
# 
# **Related to previous contact**
# * 8 - contact: contact communication type
# * 9 - month: last contact month of year
# * 10 - day_of_week: last contact day of the week
# * 11 - duration: last contact duration, in seconds*
# 
# **Other attributes**
# * 12 - campaign: number of contacts performed during this campaign and for this client
# * 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign
# * 14 - previous: number of contacts performed before this campaign and for this client
# * 15 - poutcome: outcome of the previous marketing campaign
# 
# **Output variable (desired target):has the client subscribed a term deposit?**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# calculate accuracy measures and confusion matrix
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


#Read the dataset

bank_df = pd.read_csv("/kaggle/input/bank-marketing/bank-additional-full.csv",sep = ';')


# In[3]:


#Shape of the data

bank_df.shape


# In[4]:


#Reading the dataset

bank_df.head()


# In[5]:


#Info about the dataset
bank_df.info()


# In[6]:


#### this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model

bank_df.drop(['duration'], inplace=True, axis=1)


# #### Certain variables are more relevant if they are categorical variable than numerical variables. We will convert such categorical variables to numeric variabes

# In[7]:


bank_df['pdays']=bank_df['pdays'].astype('category')
bank_df['y']=bank_df['y'].astype('category')


# # Exploratory data analysis
# 

# ## Univariate analysis - boxplot / histogram for numerical variables

# In[8]:


sns.boxplot(x=bank_df['age'], data=bank_df)


# **Age column has some outliers. The median age is about 40 years. There are some customers above 90 years of age. This data might have to be checked**

# In[9]:


#histograms from the pair plots
sns.pairplot(bank_df)


# **The distribution of all numerical variables other than age is highly skewed - hence we might want to transform or bin some of these variables**

# **On similar lines, please perform univariate analysis of other numerical variables**

# ## Univariate analysis - countplot / value count for categorical variables
# 

# In[10]:


bank_df['job'].value_counts()


# In[11]:


sns.countplot(bank_df['marital'])


# In[12]:


plt.figure(figsize=(12,5))
sns.countplot(bank_df['education'])


# In[13]:


plt.figure(figsize=(12,5))
sns.countplot(bank_df['default'])


# **default - yes is a very very small % - we can consider deleting this column**[](http://)

# In[14]:


sns.countplot(bank_df['housing'])


# In[15]:


sns.countplot(bank_df['loan'])


# In[16]:


sns.countplot(bank_df['contact'])


# In[17]:


sns.countplot(bank_df['poutcome'])


# In[18]:


sns.countplot(bank_df['y'])


# In[19]:


bank_df['y'].value_counts(normalize=True)


# ### The response rate is only 11.6%. Hence the Y variable has a high class imbalance. Hence accuracy will not be a reliable model performance measure. 
# 
# ### FN is very critical for this business case because a false negative is a customer who will potentially subscribe for a loan but who has been classified as 'will not subscribe'. Hence the most relevant model performance measure is recall

# ## Bivariate analysis

# In[20]:


#Rename the dependant column from 'y ' to 'Target'
bank_df.rename(columns={'y':'Target'}, inplace=True)


# In[21]:


bank_df.columns


# In[22]:


#Group numerical variables by mean for the classes of Y variable
np.round(bank_df.groupby(["Target"]).mean() ,1)


# #### The mean balance is higher for customers who subscribe to the term deposit compared to those who dont
# 
# 
# #### number of days that passed by after the client was last contacted from a previous campaign is higher for people who have subscribed
# 
# #### number of contacts performed before this campaign is also higher for customers who subscribe

# ### All of the above facts indicate that customers with a higher balance and those who have been contacted frequently before the campaign tend to subscribe for the term deposit

# In[23]:


### Bivariate analysis using crosstab


# ### Bivariate analysis using crosstab

# In[24]:


pd.crosstab(bank_df['job'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# #### The highest conversion is for students (31%) and lowest is for blue-collar(7%)

# In[25]:


pd.crosstab(bank_df['marital'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[26]:


pd.crosstab(bank_df['education'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[27]:


print(pd.crosstab(bank_df['default'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False ))
print(bank_df['default'].value_counts(normalize=True))


# ### Since default - yes is only 0.073% of the data and the conversion is also comparitively lower for default - yes, we can remove this column

# In[28]:


bank_df.drop(['default'], axis=1, inplace=True)


# In[29]:


bank_df.columns


# In[30]:


pd.crosstab(bank_df['housing'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[31]:


pd.crosstab(bank_df['loan'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[32]:


pd.crosstab(bank_df['contact'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# In[33]:


pd.crosstab(bank_df['day_of_week'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )[0:10]


# In[34]:


pd.crosstab(bank_df['month'], bank_df['Target'], normalize='index').sort_values(by='yes',ascending=False )


# ### List out the high level findings from bivariate analysis that could provide pointers to feature selection
# 

# In[35]:


#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin


# In[36]:


#Binning campaign
cut_points = [2,3,4]
labels = ["<=2","3","4",">4"]
bank_df['campaign_range'] = binning(bank_df['campaign'], cut_points, labels)
bank_df['campaign_range'].value_counts()


# In[37]:


bank_df.drop(['campaign'], axis=1, inplace=True)
bank_df.columns


# In[38]:


X = bank_df.drop("Target" , axis=1)
y = bank_df["Target"]   # select all rows and the 17 th column which is the classification "Yes", "No"
X = pd.get_dummies(X, drop_first=True)


# In[39]:


test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[40]:


X_train.shape,X_test.shape


# In[41]:


#instantiating decision tree as the default model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[42]:


# Is the model an overfit model? 
y_pred = dt_model.predict(X_test)
print(dt_model.score(X_train, y_train))
print(dt_model.score(X_test , y_test))


# In[43]:


# Note: - Decision Tree is a non-parametric algorithm and hence prone to overfitting easily. This is evident from the difference
# in scores in training and testing

# In ensemble techniques, we want multiple instances (each different from the other) and each instance to be overfit!!!  
# hopefully, the different instances will do different mistakes in classification and when we club them, their
# errors will get cancelled out giving us the benefit of lower bias and lower overall variance errors.


# In[44]:


#Confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))


print(recall_score(y_test, y_pred,average="binary", pos_label="yes"))


# #### The recall score is relatively low and this has to be improves in the model
# 

# In[45]:


clf_pruned = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_pruned.fit(X_train, y_train)


# ## Visualizing the tree

# In[46]:


import graphviz
from sklearn.tree import export_graphviz

data = export_graphviz(clf_pruned,out_file=None,feature_names=feature_cols,class_names=['0','1'],   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
graph


# In[47]:


## Calculating feature importance
#feature_names=feature_cols
feat_importance = clf_pruned.tree_.compute_feature_importances(normalize=False)


feat_imp_dict = dict(zip(feature_cols, clf_pruned.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.sort_values(by=0, ascending=False)[0:10] #Top 10 features


# In[48]:


preds_pruned = clf_pruned.predict(X_test)
preds_pruned_train = clf_pruned.predict(X_train)


# In[49]:


acc_DT = accuracy_score(y_test, preds_pruned)
recall_DT = recall_score(y_test, preds_pruned, average="binary", pos_label="yes")


# In[50]:


#Store the accuracy results for each model in a dataframe for final comparison
resultsDf = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT, 'recall': recall_DT})
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf


# ### Overfitting is reduced after pruning, but recall has drastically reduced

# In[51]:


## Apply the Random forest model and print the accuracy of Random forest Model


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 50)
rfcl = rfcl.fit(X_train, y_train)





# In[52]:


pred_RF = rfcl.predict(X_test)
acc_RF = accuracy_score(y_test, pred_RF)
recall_RF = recall_score(y_test, pred_RF, average="binary", pos_label="yes")


# In[53]:


tempResultsDf = pd.DataFrame({'Method':['Random Forest'], 'accuracy': [acc_RF], 'recall': [recall_RF]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# In[54]:


## Apply Adaboost Ensemble Algorithm for the same data and print the accuracy.


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier( n_estimators= 200, learning_rate=0.1, random_state=22)
abcl = abcl.fit(X_train, y_train)





# In[55]:


pred_AB =abcl.predict(X_test)
acc_AB = accuracy_score(y_test, pred_AB)
recall_AB = recall_score(y_test, pred_AB, pos_label='yes')


# In[56]:


tempResultsDf = pd.DataFrame({'Method':['Adaboost'], 'accuracy': [acc_AB], 'recall':[recall_AB]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# In[57]:


## Apply Bagging Classifier Algorithm and print the accuracy


from sklearn.ensemble import BaggingClassifier

bgcl = BaggingClassifier(n_estimators=100, max_samples= .7, bootstrap=True, oob_score=True, random_state=22)
bgcl = bgcl.fit(X_train, y_train)






# In[58]:


pred_BG =bgcl.predict(X_test)
acc_BG = accuracy_score(y_test, pred_BG)
recall_BG = recall_score(y_test, pred_BG, pos_label='yes')


# In[59]:


tempResultsDf = pd.DataFrame({'Method':['Bagging'], 'accuracy': [acc_BG], 'recall':[recall_BG]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# In[60]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.1, random_state=22)
gbcl = gbcl.fit(X_train, y_train)





# In[61]:


pred_GB =gbcl.predict(X_test)
acc_GB = accuracy_score(y_test, pred_GB)
recall_GB = recall_score(y_test, pred_GB, pos_label='yes')


# In[62]:


tempResultsDf = pd.DataFrame({'Method':['Gradient Boost'], 'accuracy': [acc_GB], 'recall':[recall_GB]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'accuracy', 'recall']]
resultsDf
resultsDf


# ### Bagging gives overall best model performance. However, please note that the recall is still very low and will have to be improved
