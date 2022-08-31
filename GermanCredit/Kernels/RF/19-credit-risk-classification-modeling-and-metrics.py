#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().system('pip install plotly')
import plotly.offline as py 
import plotly.graph_objs as go
import plotly.express as px
from collections import Counter  
from subprocess import call
from IPython.display import Image
############################################################################################
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Data Ingestion :**
# 
# In the beginning , I start by loading data and checking it

# In[ ]:


credit=pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")
print("The dataset is {} credit record".format(len(credit)))


# In[ ]:


credit.head(2)


# According to the data dictionnary provided to detail each of the columns:
# * Age (numeric)
# * Sex (text: male, female)
# * Job (numeric: 0 — unskilled and non-resident, 1 — unskilled and resident, 2 — skilled, 3 — highly skilled)
# * Housing (text: own, rent, or free)
# * Saving accounts (text — little, moderate, quite rich, rich)
# * Checking account (numeric, in DM — Deutsch Mark)
# * Credit amount (numeric, in DM)
# * Duration (numeric, in month)
# * Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

# **Data checks:**
# 
# The function info() helps to get a concise summary of a DataFrame by providning data types per column

# In[ ]:


credit=credit.iloc[:, 1:]


# In[ ]:


credit.info()


# The data sanity shows that two columns contains NaN values which will be handled later.

# In[ ]:


credit.describe()


# **Descriptive analysis:**
# 
# Exploratory data analysis in a data science project is a mandatory step in order to understand the way some of the attributes are distributed. In this chapter, I focus on drawing some charts in order to find out and demonstrate insights. To this end, I use the Plotly’s Python graphing library to create graphs which makes interactive, publication-quality graphs. For further detail please check plotly website.

# *** Sex Vs Age Cross tabulation:**
# 
# A box plot is a statistical representation of numerical data through their quartiles. The ends of the box represent the lower and upper quartiles, while the median (second quartile) is marked by a line inside the box.

# In[ ]:


credit['Sex'].value_counts()


# In[ ]:


SA = credit.loc[:,['Sex','Age']]
fig = px.box(SA, x="Sex", y="Age", points="all",color="Sex")
fig.update_layout(
    title={
          'text':"Sex Vs Age Cross tabulation",
        'y':.95,
        'x':.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Sex",
    yaxis_title="Age",
   
)
fig.show()


# In[ ]:


SC =credit.loc[:,['Sex','Credit amount']]
fig = px.box(SC, x="Sex", y="Credit amount", points="all", color="Sex")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.update_layout(
    title={
          'text':"Sex Vs Credit Amount Cross tabulation",
        'y':.95,
        'x':.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Sex",
    yaxis_title="Age",
   
)
fig.show()


# * **Purpose distribution:**
# 
# A histogram is a representation of the distribution of numerical data, where the data are binned and the count for each bin is represented.

# In[ ]:


Purpose = credit['Purpose']
fig = px.histogram(credit, x="Purpose", color="Purpose")
fig.update_layout(
    title={
          'text':"Purpose breakdown",
        'y':.95,
        'x':.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
   
)
fig.show()


# The histogram chart shows that most of credit purpose is related to car prurchase , followed by radio/TV one.

# *** Purpose Vs Credit Amount Cross tabulation**

# In[ ]:


SC =credit.loc[:,['Purpose','Credit amount']]
fig = px.box(SC, x="Purpose", y="Credit amount", color="Purpose")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.update_layout(
    title={
          'text':"Purpose Vs Credit Amount Cross tabulation",
        'y':.95,
        'x':.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Purpose",
    yaxis_title="Credit amount",
   
)
fig.show()


# **Transformations and feature engineering:**
# 
# Part of the data cleansing step involves:
# * hypothesizing about the features I think I need
# * preparing to integrate them in my model
# 
# A machine learning model unfortunately cannot deal with categorical variables (except for some models ). Therefore, I have to find a way to encode these variables as numbers before handling them within the model. There are two main ways to carry out this process:
# 
# * Label Encoding: is the concept of assigning each unique category in a categorical variable with an integer. No new columns are created.
# 
# => It is only recommended for two unique categories since it gives the categories an arbitrary ordering
# 
# * One-Hot Encoding: is the concept of creating a new column for each unique category in a categorical variable. Each observation receives a value of “1” in the column for its corresponding category and a value “0” in all other new columns.
# 
# The Risk is what I would like to predict: either a 0 for the loan presenting no risk and will be repaid on time, or a 1 indicating that the loan presents a risk and the client will have some payment difficulties.
# To this end, I have two unique categories that’s why I use the map function for Label encoding.

# In[ ]:


credit['Risk'] = credit['Risk'].map({'bad':1, 'good':0})


# When the time comes to build the machine learning model, I have to fill in these missing values (known as imputation) identified during the data checks phase.
# In my case I have at my disposal a small dataset which oblige me to keep all my rows that’s why I have introduced a new category value called “Others” for both Saving account and Checking account columns.

# In[ ]:


credit['Saving accounts'] = credit['Saving accounts'].fillna('Others')
credit['Checking account'] = credit['Checking account'].fillna('Others')


# I create then a checkpoint:

# In[ ]:


credit_clean=credit.copy()


# The Second step consists of transforming the data into dummy variable which is a part of One-hot encoding:

# In[ ]:


cat_features = ['Sex','Housing', 'Saving accounts', 'Checking account','Purpose']
num_features=['Age', 'Job', 'Credit amount', 'Duration','Risk']
for variable in cat_features:
    dummies = pd.get_dummies(credit_clean[cat_features])
    df1= pd.concat([credit_clean[num_features], dummies],axis=1)

Risk= df1['Risk']          
df2=df1.drop(['Risk'],axis=1)


# Since the data is ready to be integrated and fit into the model, I can start by splitting it into training and testing sets.

# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(df2,Risk,test_size=0.20,random_state = 30)


# * **Model building process:**
# 
# The risk prediction is a standard supervised classification task:
# * Supervised: The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features
# * Classification: The label is a binary variable, 0 (no risk and loan will be on time), 1 (risky loan will have difficulty repaying loan)
# I use RandomForestClassifier from scikit-learn with the familiar Scikit-Learn modeling syntax: I first create the baseline model which will be tuned in order to seek the best hyperparameters.

# In[ ]:


random_forest = RandomForestClassifier( random_state = 100)


# * **Model Optimization:**
# 
# Hyperparameters are model-specific parameters whose values are set before the learning process begins. In my case, I am using random forest classifier, hyperparameters include for example the number of trees in the forest (n_estimators) and the maximum depth of the tree (max_depth) as described within the model specifications.
# Hyperparameters Tuning is a measure of how much performance can be gained by tuning them and searching for the right set of hyperparameter to achieve high precision and accuracy.
# There are several parameter tuning techniques, but two of the most widely-used parameter optimizing techniques are :
# 
# * Grid Search : The concept behavior is similar to the grid, where all the values are placed in the form of a matrix. Each combination of parameters is taken into consideration.
# 
# * Random Search : The concept tries random combinations of the hyperparameters to find the best solution for the built model based on the defined scoring.
# 
# 
# I try to adjust the following RF set of hyperparameters using Random search:
# * n_estimators = number of trees in the forest
# * max_features = max number of features considered for splitting a node
# * max_depth = max number of levels in each decision tree
# * min_samples_split = min number of data points placed in a node before the node is split
# * min_samples_leaf = min number of data points allowed in a leaf node
# * bootstrap = method for sampling data points (with or without replacement)

# In[ ]:


#Standardization
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]# Number of trees in random forest
max_features = ['auto', 'sqrt']# Number of features to consider at every split
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]# Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]# Minimum number of samples required at each leaf node
bootstrap = [True, False]# Method of selecting samples for training each tree

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

random_forest = RandomForestClassifier(random_state = 100)
rf_random = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=4, scoring='recall', random_state=42, n_jobs = -1)
rf_random.fit(X_train_std, Y_train)
rf_random.best_params_


# In[ ]:


Y_test_pred = rf_random.predict(X_test_std)


# **Model metrics:**
# 
# *** Confusion matrix:**
# 
# It is a performance metric widely-used for machine learning classification tasks where output can be two or more classes. It is an array with 4 different combinations of predicted and actual values as shown below:

# In[ ]:


confusion_matrix= confusion_matrix(Y_test, Y_test_pred)
confusion_matrix


# In[ ]:


y_true = ["bad", "good"]
y_pred = ["bad", "good"]
df_cm = pd.DataFrame(confusion_matrix, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
df_cm.dtypes

plt.figure(figsize = (8,5))
plt.title('Confusion Matrix')
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# To summarize the confusion matrix :
# 
# * TRUE POSITIVES (TP)= 122,
# * TRUE NEGATIVES (TN)= 18,
# * FALSE POSITIVES (FP)= 14,
# * FALSE NEGATIVES (FN)= 46.
# 
# The confusion matrix is extremely useful for measuring:
# * Recall(Sensitivity): also called the True Positive Rate is defined as the proportion of loan with an associated risk which will have a positive result. In other words, a highly sensitive test is one that correctly identifies credit with risk.
# * Specificity: also called the True Negative Rate is defined as the proportion of loans without an associated risk which will have a negative result. In other words, a highly sensitive test is one that correctly identifies credit without risk.
# * Accuracy: is the number of correctly predicted risks out of all the data points.
# * Precision: is the number of loans with risk which were actually correct

# The formulae for the evaluation metrics are as follows :

# In[ ]:


total=sum(sum(confusion_matrix))

sensitivity_recall = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity_recall : ',sensitivity_recall )

Specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity: ', Specificity)

precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('Precision: ', precision)

accuracy =(confusion_matrix[0,0]+confusion_matrix[1,1])/(confusion_matrix[0,0]+confusion_matrix[0,1]+
                                                         confusion_matrix[1,0]+confusion_matrix[1,1])
print('Accuracy: ', accuracy)


# **The roc curve:**
# An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:
# 
# * True Positive Rate
# * False Positive Rate
# 
# 
# The area covered by the curve is the area between the blue line (ROC) and the axis. This area covered is AUC. The bigger the area covered, the better the machine learning model is at distinguishing the given classes.

# In[ ]:


fpr, tpr, thresholds = roc_curve(Y_test, Y_test_pred)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12

plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

print("\n")
print ("Area Under Curve: %.2f" %auc(fpr, tpr))
print("\n")

