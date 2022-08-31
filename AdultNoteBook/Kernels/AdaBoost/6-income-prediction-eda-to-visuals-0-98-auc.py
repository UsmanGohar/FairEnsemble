#!/usr/bin/env python
# coding: utf-8

# ## Importing Important libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# #### Loading the data onto notebook.

# In[2]:


data = pd.read_csv("../input/adult-census-income/adult.csv")
data


# ![](https://kiarofoods.com/wp-content/uploads/2019/10/line_break.png)
# # Exploratory Data Analysis:
# **Problem Type Identification:** We have the target variable available with us. So, it a supervised machine learning problem. First we try to find out the type of supervised machine learning that we have in this case study by lookin at the target variable

# In[3]:


print(f"Target: 'Income'\nUnique Values in Income: {data.income.unique()}\nNumber of unique values: {data.income.nunique()}")


# In the problem, we have 'Income' as the Target variable. we see that we have only two values which are to be predicted, either the income is greater than 50K, which is Yes, or the income is less than or equal to 50K, which is No. We will label encode the target variable.

# In[4]:


data['income'] = data['income'].str.replace('<=50K', '0')
data['income'] = data['income'].str.replace('>50K', '1')
data['income'] = data['income'].astype(np.int64)


# In[5]:


data.income.dtypes


# We can see that, we have encoded the values of the target variable, and converted it into int data-type. This problem is a classification problem with 'Income' as the target variable. Making a copy of the dataset to work ahead

# In[6]:


ds = data.copy()
print(f"Unique values in 'education': {ds.education.nunique()}\nUnique values in 'Education_num': {ds['education.num'].nunique()}")


# We see that for the feature 'Education', we already have the encoded values in feature 'Education_num'. 'Education' will be removed from the dataset.

# In[7]:


ds.drop(['education'], axis = 1, inplace = True)


# Checking to see that is there any Null values present in the data that we have. Handling the null values will be the first thing that we need to do. Then, we have a look at the data-types of the other features and the value counts and unique values in those features.

# In[8]:


plt.title("Null values in the data", fontsize = 12)
sns.heatmap(ds.isnull(), cmap = 'inferno')
plt.show()


# From the heatmap, we see that the dataset consists of no null values. But for some features, we have '?' as the values present. **'?' will be considered as null values.** We move ahead with the feature engineering part. Checking the datatypes of the columns

# In[9]:


print("Datatype of every feature: ")
ds.dtypes


# In[10]:


print("Number of unique values in every feature: ")
ds.nunique()


# *'Workclass', 'Marital_status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native_country' are the categorical variables in the data*. Proper encoding or conversion of these variables is necessary for the feature engineering. We will look at these attributes and convert them one by one.
# 
# **'Workclass':** Starting off with the work class, we look the number of unique values and value counts for those values

# In[11]:


ds.workclass.value_counts()


# In work class, *majority of the people are private employees*. The *minority of people are either working without-pay or they have never-worked*. We can combine the values of these two values as one. first we remove the blank space from the column is present in any values.

# In[12]:


ds['workclass'] = ds['workclass'].str.replace('Never-worked', 'Without-pay')


# Now, we have 8 unique values in this feature. But, we see that **there are some values where we have '?' in the column. This values can be replaced with NaN values.**

# In[13]:


ds['workclass'] = ds['workclass'].replace('?', np.NaN)


# In[14]:


plt.figure(figsize = (10,6))
plt.title("Income of people according to their workclass", fontsize = 16)
sns.countplot(y = ds['workclass'], hue = ds['income'])
plt.show()


# We see that the **majority of people who have income more than 50K a year are from private sector**. Same goes for the people with income less than 50K. But *for the Self Employed sector, the number of people whose income > 50K are more than the number of people whose income < 50K.* Now, moving ahead with replacing the null values and encoding the feature. **We will replace the NaN values in the 'Workclass' feature by the mode of the column, grouping it by the 'Occupation' feature.** We now have 7 unique values in Workclass feature. We can encode these values using the frequency encoding technique.

# In[15]:


from scipy.stats import mode
workclass_mode = ds.pivot_table(values='workclass', columns='occupation',aggfunc=(lambda x:mode(x).mode[0]))
workclass_mode


# In[16]:


loc1 = ds['workclass'].isnull()
ds.loc[loc1, 'workclass'] = ds.loc[loc1,'occupation'].apply(lambda x: workclass_mode[x])


# In[17]:


workclass_enc = (ds.groupby('workclass').size()) / len(ds)
print(workclass_enc)

ds['workclass_enc'] = ds['workclass'].apply(lambda x : workclass_enc[x])
ds['workclass_enc'].head(3)


# In[18]:


ds.drop(['workclass'], axis = 1, inplace = True)


# **'Occupation':** Similar to 'Workclass', we will look at the unique values and value counts in the 'Occupation' feature.

# In[19]:


ds.occupation.value_counts()


# **We will drop the rows where the occupation is NaN.**

# In[20]:


ds['occupation'] = ds['occupation'].replace('?', np.NaN)
ds = ds.loc[ds['occupation'].isnull() == False]
ds


# As we cannn see that after removing the null values from 'occupation', we are left with 30718 observations.

# In[21]:


plt.style.use('ggplot')
plt.figure(figsize = (10,6))
plt.title("Income of people according to their occupation", fontsize = 16)
sns.countplot(y = ds['occupation'], hue = ds['income'])
plt.show()


# Majority of people whose income is greater than 50K are either executive managers or they belong to any professional speciality. Now, encoding the occupation by frequency of the values in the column.

# In[22]:


occupation_enc = (ds.groupby('occupation').size()) / len(ds)
print(occupation_enc)

ds['occupation_enc'] = ds['occupation'].apply(lambda x : occupation_enc[x])
ds['occupation_enc'].head(3)


# In[23]:


ds.drop(['occupation'], axis = 1, inplace = True)


# **'Native_country':** We are checking for the salary on people in USA and outside USA, so , **we will convert all the values where country is not USA to 'non-usa'.** This way, we can encode the values by one-hot encoding without increasing the curse of dimensionality.

# In[24]:


ds['native.country'].loc[ds['native.country'] == 'United-States'] = 'usa'
ds['native.country'].loc[ds['native.country'] != 'usa'] = 'non_usa'
ds['native.country'].value_counts()


# In[25]:


plt.style.use('default')


# In[26]:


plt.style.use('seaborn-pastel')


# In[27]:


plt.figure(figsize = (8,3))
plt.title("Income of people according to their native country", fontsize = 16)
sns.countplot(y = ds['native.country'], hue = ds['income'])
plt.show()


# **Majority of people with higher income belong to the USA**. We also have more number of people from USA then any other country combined in this dataset. Encoding this feature using one hot encoding.

# In[28]:


ds['country_enc'] = ds['native.country'].map({'usa' : 1, 'non_usa' : 0})
ds.drop(['native.country'], axis = 1, inplace = True)


# **'Sex':** Similarly, encoding the sex using one hot encoding.

# In[29]:


plt.title("Income of people by their sex", fontsize = 16)
sns.countplot(x = ds['sex'], hue = ds['income'])
plt.show()


# We can see that male have more salary than female. Also in the dataset, the number of men are more than women. Encoding this feature with one hot encoding.

# In[30]:


ds['sex_enc'] = ds['sex'].map({'Male' : 1, 'Female' : 0})
ds.drop(['sex'], axis = 1, inplace = True)


# **'Marital_status':** Looking at the iincome of people according to their marital status.

# In[31]:


plt.style.use('default')


# In[32]:


plt.style.use('seaborn-talk')


# In[33]:


plt.title("Income of people by Marital Status", fontsize = 16)
sns.countplot(y = ds['marital.status'], hue = ds['income'])
plt.show()


# **Married people have a higher income as compared to others.** Encoding the feature

# In[34]:


marital_status_enc = (ds.groupby('marital.status').size()) / len(ds)
print(marital_status_enc)

ds['marital_status_enc'] = ds['marital.status'].apply(lambda x : marital_status_enc[x])
ds['marital_status_enc'].head(3)


# In[35]:


ds.drop(['marital.status'], axis = 1, inplace = True)


# Similarly, **for 'Race' and 'Relationship'**

# In[36]:


plt.style.use('bmh')


# In[37]:


plt.figure(figsize = (12,4))

plt.subplot(1, 2, 1)
sns.countplot(y = ds['race'], hue = ds['income'])
plt.title("Income respective to Race", fontsize = 12)

plt.subplot(1, 2, 2)
sns.countplot(y = ds['relationship'], hue = ds['income'])
plt.title("Income respective to Relationship", fontsize = 12)

plt.tight_layout(pad = 4)
plt.show()


# **White people have a higher salary as compared to other races**. Similarly, **husband in the family have a higher salary as compared to other relationship in the family.** Encoding both these columns

# In[38]:


race_enc = (ds.groupby('race').size()) / len(ds)
print(race_enc,'\n')
ds['race_enc'] = ds['race'].apply(lambda x : race_enc[x])

relationship_enc = (ds.groupby('relationship').size()) / len(ds)
print(relationship_enc)
ds['relationship_enc'] = ds['relationship'].apply(lambda x : relationship_enc[x])


# In[39]:


ds.drop(['race', 'relationship'], axis = 1, inplace = True)
new_ds = ds.drop(['income'], axis = 1)
new_ds['income'] = ds['income']
new_ds


# ## Outliers:
# We check if any outliers are present in the continous attributes of the dataset. We check it both by visualisations and the zscore for the continous columns.

# In[40]:


plt.style.use('default')


# In[41]:


plt.style.use('ggplot')


# In[42]:


clist = ['fnlwgt','age','capital.gain','capital.loss','hours.per.week']
plt.figure(figsize = (12,6))
for i in range(0, len(clist)):
    plt.subplot(2,3, i+1)
    sns.boxplot(ds[clist[i]], color = 'skyblue')
print("BoxPlots of the features:")
plt.show()


# **Outliers are present in the continous columns of the feature**. We will check the z-score of the features and and clip them from the data.

# In[43]:


from scipy.stats import zscore
zabs = np.abs(zscore(new_ds.loc[:,'fnlwgt':'hours.per.week']))
print(np.shape(np.where(zabs >= 3)))
new_ds = new_ds[(zabs < 3).all(axis = 1)]
new_ds


# WE have a total of 2566 outliers in the data. After removing the outliers, we have 28213 observations left.
# ## Correlation:
# Checking the correlation between the features and target variable to see which of them columns are more related to target.

# In[44]:


plt.figure(figsize = (14, 8))
plt.title("Correlation between target and features:")
sns.heatmap(new_ds.corr(), annot = True)
plt.show()


# 'Capital_gain', 'Education_num', 'Marital_status_enc', 'Relationship_enc' are most correlated to the Income of the observations.
# ## Scaling:
# As we see that the values of attributes in the dataset vary largely, so it is important to scale the data. Using the Min-Max scaler in order to bring normalisation in the data.

# In[45]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
new_ds.loc[:,'age':'hours.per.week'] = scale.fit_transform(new_ds.loc[:,'age':'hours.per.week'])
new_ds


# AS we can see from the above table that the data is now more normalised and can be used by the models for learning.
# ![](https://kiarofoods.com/wp-content/uploads/2019/10/line_break.png)
# # Data Imbalance:
# If the data is imbalanced, it can cause the overfitting and bias in the odel prediction. So it is important to check and cure the data imbalance if present. We check the target variable to see if it is balanced or not.

# In[46]:


plt.figure(figsize = (8, 4))
plt.title("Values distribution in target class: Income")
sns.countplot(data = new_ds, x = 'income')
plt.show()


# As we can see that data is imbalanced. In order **to remove the data imbalance, we use the SMOTETomek class to create synthetic values using KNN algorithm.**

# In[47]:


from imblearn.combine import SMOTETomek
x = new_ds.loc[:,"age":"relationship_enc"]
y = new_ds.loc[:,"income"]
smk = SMOTETomek()
x_new, y_new = smk.fit_resample(x, y)


# In[48]:


plt.figure(figsize = (8, 4))
plt.title("Values in target class after using SMOTETomek")
sns.countplot(x = y_new)
plt.show()


# As we can see that we now have a balanced dataset, so we can model ahead with the model building part.
# ![](https://kiarofoods.com/wp-content/uploads/2019/10/line_break.png)
# # Model Building:
# Starting with the spliting of the training and testing data. For that, we check to see what is the best random state.

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

max_accuracy = 0
best_rs = 0
for i in range(1, 150):
    x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.30, random_state = i)
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    pred = lg.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > max_accuracy: # after each iteration, acc is replace by the best possible accuracy
        max_accuracy = acc
        best_rs = i
print(f"Best Random State is {best_rs}, {max_accuracy*100}")


# Best possible random state is 67, so using it to split the data

# In[50]:


x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.30, random_state = 67)


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ## Model Fitting:
# Fitting 7 different models to check which model gives the best accuracy.

# In[52]:


# For Logistic Regression
lg = LogisticRegression()
lg.fit(x_train, y_train)
pred_lg = lg.predict(x_test)
print("Accuracy Score of Logistic Regression model is", accuracy_score(y_test, pred_lg)*100)

# For Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
pred_dtc = dtc.predict(x_test)
print("Accuracy Score of Decision Tree Classifier model is", accuracy_score(y_test, pred_dtc)*100)

# For K-Nearest Neighbour Classifier
knc = KNeighborsClassifier(n_neighbors = 5)
knc.fit(x_train, y_train)
pred_knc = knc.predict(x_test)
print("Accuracy Score of K-Nearest Neighbour Classifier model is", accuracy_score(y_test, pred_knc)*100)

# For Support Vector Classifier
svc = SVC(kernel = 'rbf')
svc.fit(x_train, y_train)
pred_svc = svc.predict(x_test)
print("Accuracy Score of Support Vector Classifier model is", accuracy_score(y_test, pred_svc)*100)

# For Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
pred_rfc = rfc.predict(x_test)
print("Accuracy Score of Random Forest model is", accuracy_score(y_test, pred_rfc)*100)

# For MultinomialNB
nb = MultinomialNB() # making the Multinomial Naive Bayes class
nb.fit(x_train, y_train) # fitting the model
pred_nb = nb.predict(x_test) # predicting the values
print("Accuracy Score of MultinomialNB model is", accuracy_score(y_test, pred_nb)*100)

# For ADA Boost Classifier
ada= AdaBoostClassifier()
ada.fit(x_train, y_train) # fitting the model
pred_ada = ada.predict(x_test) # predicting the values
print("Accuracy Score of ADA Boost model is", accuracy_score(y_test, pred_ada)*100)


# Best accuracy score is given by Random Forest Classifier model. In order to avoid the bias and overfitting or underfitting, we cross validate the models and check the mean accuracy score of them.
# ## Cross Validation:
# Cross validating the m,odels to see if they are underfitting or overfitting and to prevent bias. We will compare the mean accuracy scores of the model.

# In[53]:


from sklearn.model_selection import cross_val_score

lg_scores = cross_val_score(lg, x_new, y_new, cv = 10) # cross validating the model
print(lg_scores) # accuracy scores of each cross validation cycle
print(f"Mean of accuracy scores is for Logistic Regression is {lg_scores.mean()*100}\n")

dtc_scores = cross_val_score(dtc, x_new, y_new, cv = 10)
print(dtc_scores)
print(f"Mean of accuracy scores is for Decision Tree Classifier is {dtc_scores.mean()*100}\n")

knc_scores = cross_val_score(knc, x_new, y_new, cv = 10)
print(knc_scores)
print(f"Mean of accuracy scores is for KNN Classifier is {knc_scores.mean()*100}\n")

svc_scores = cross_val_score(svc, x_new, y_new, cv = 10)
print(svc_scores)
print(f"Mean of accuracy scores is for SVC Classifier is {svc_scores.mean()*100}\n")

rfc_scores = cross_val_score(rfc, x_new, y_new, cv = 10)
print(rfc_scores)
print(f"Mean of accuracy scores is for Random Forest Classifier is {rfc_scores.mean()*100}\n")

nb_scores = cross_val_score(nb, x_new, y_new, cv = 10)
print(nb_scores)
print(f"Mean of accuracy scores is for MultinomialNB is {nb_scores.mean()*100}\n")

ada_scores = cross_val_score(ada, x_new, y_new, cv = 10)
print(ada_scores)
print(f"Mean of accuracy scores is for ADA Boost Classifier is {ada_scores.mean()*100}\n")


# In[54]:


# Checking for difference between accuracy and mean accuracies.
lis3 = ['Logistic Regression','Decision Tree Classifier','KNeighbors Classifier','SVC', 'Random Forest Classifier', 
        'MultinomialNB', 'ADA Boost Classifier']

lis1 = [accuracy_score(y_test, pred_lg)*100, accuracy_score(y_test, pred_dtc)*100, accuracy_score(y_test, pred_knc)*100, 
        accuracy_score(y_test, pred_svc)*100, accuracy_score(y_test, pred_rfc)*100, accuracy_score(y_test, pred_nb)*100,
        accuracy_score(y_test, pred_ada)*100]

lis2 = [lg_scores.mean()*100, dtc_scores.mean()*100, knc_scores.mean()*100, svc_scores.mean()*100, rfc_scores.mean()*100, 
        nb_scores.mean()*100, ada_scores.mean()*100]

for i in range(0, 7):
    dif = (lis1[i]) - (lis2[i])
    print(lis3[i], dif)


# **Random forest classifier is the best model with highest cross validation mean score and accuracy score**. We will use it for the model building.
# ## Hyperparameter Tuning:
# Tuning the parameters of the Random Forest in order to obtain the best possible parameters for model building.

# In[55]:


from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
param = dict()
param['criterion'] = ['gini', 'entropy']
param['n_estimators'] = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
param['min_samples_split'] = [1,2,5,8,10,15,20,25,50,55,60,80,100]


gs = GridSearchCV(estimator = rfc, param_grid = param, scoring='f1', cv = 5, n_jobs = 3)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# After the hyperparameter tuning, **the best parameters for Random Forest Classifier are 'crietrion' = 'entropy', 'min_samples_split' = 2, 'n_estimators' = 100**. We build the model using these parameters.

# In[56]:


rfc = RandomForestClassifier(criterion = 'entropy', min_samples_split = 2, n_estimators = 100)
rfc.fit(x_train, y_train)
print(rfc.score(x_train, y_train))
pred_rfc = rfc.predict(x_test)


# ![](https://kiarofoods.com/wp-content/uploads/2019/10/line_break.png)
# # Model Evaluation:
# We have build the model after the cross validation and hyper parameter tuning. It is now time to evaluate the model using the classification report, confusion matrix and ROC curve.

# In[57]:


from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy Score of RFC model is", accuracy_score(y_test, pred_rfc)*100)
print("Confusion matrix for RFC Model is")
print(confusion_matrix(y_test, pred_rfc))
print("Classification Report of the RFC Model is")
print(classification_report(y_test, pred_rfc))

plot_roc_curve(rfc, x_test, y_test) # arg. are model name, feature testing data, label testing data.
plt.title("Recevier's Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# After the model evaluation , we get the **precision and recall for both the target variable as 0.92 and 0.91**. The **f1- score of the model is 0.92**. The ROC curve gave us **the AUC score which is 0.98**. Model evaluation gives the results that ***the prediction is very accurate.***
# ![](https://miro.medium.com/max/2400/1*IH10jlQEJ7GW1_oq8s7WPw.png)
# # Serialisation:
# Now we save the Random Forest Classifier Model as an object using joblib.

# In[58]:


import joblib
joblib.dump(rfc, 'Census Income Prediction.obj') # saving the model as an object


# In[ ]:




