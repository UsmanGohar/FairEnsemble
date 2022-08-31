#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing the Required Libraries

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
from IPython.core.display import display, HTML

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
import warnings; warnings.simplefilter('ignore')
display(HTML("<style>.container { width:75% !important; }</style>"))


# In[ ]:


# Importing the Data Set
df_adult_eda = pd.read_csv("/kaggle/input/adult-census-income/adult.csv")
df_adult_eda.head()


# In[ ]:


df_adult_eda.info()


# In[ ]:


# Removing any space in the names of the columns
df_adult_eda.columns = df_adult_eda.columns.str.replace(' ', '')
df_adult_eda.columns


# In[ ]:


df_adult_eda['income'].value_counts()


# In[ ]:


print("Initial shape of the dataset : ", df_adult_eda.shape)

# Dropping the duplicate Rows
df_adult_eda = df_adult_eda.drop_duplicates(keep = 'first')
print ("Shape of the dataset after dropping the duplicate rows : ", df_adult_eda.shape)


# In[ ]:


df_adult_eda.head()


# In[ ]:


df_adult_eda['age'].nunique()


# In[ ]:


df_adult_eda['workclass'].unique()


# In[ ]:


# Checking the null values in the columns
df_adult_eda.isnull().sum(axis = 0)


# In[ ]:


df_adult_eda[df_adult_eda['native.country'] == '?'].shape


# In[ ]:


# This Code will Count the occuring of the '?' in all the columns
for i in df_adult_eda.columns:
    t = df_adult_eda[i].value_counts()
    index = list(t.index)
    print ("The Value Counts of ? in", i)
    for i in index:
        temp = 0
        if i == '?':
            print (t['?'])
            temp = 1
            break
    if temp == 0:
        print ("0")


# In[ ]:


# Dropping the rows whose occupation is '?' 
df_adult_eda = df_adult_eda[df_adult_eda.occupation != '?']

df_adult_eda['occupation'].value_counts()


# In[ ]:


# The minimum age of the person
df_adult_eda.at[df_adult_eda['age'].idxmin(),'age']


# ## Exploratory Data Analysis

# In[ ]:


# This distribution plot shows the distribution of Age of people across the Data Set
plt.rcParams['figure.figsize'] = [12, 8]
sns.set(style = 'whitegrid')

sns.distplot(df_adult_eda['age'], bins = 90, color = 'mediumslateblue')
plt.ylabel("Distribution", fontsize = 15)
plt.xlabel("Age", fontsize = 15)
plt.margins(x = 0)

print ("The maximum age is", df_adult_eda['age'].max())
print ("The minimum age is", df_adult_eda['age'].min())


# In[ ]:


# Distribution of Different Features of the Dataset
distribution = df_adult_eda.hist(edgecolor = 'black', linewidth = 1.2, color = 'c')
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.show()


# In[ ]:


# Checking the Difference between the values of the mean and median to get an idea about the amount of outliers
print (df_adult_eda['hours.per.week'].median())
print (df_adult_eda['hours.per.week'].mean())


# In[ ]:


# This heatmap shows the Correlation between the different variables
plt.rcParams['figure.figsize'] = [10,7]
sns.heatmap(df_adult_eda.corr(), annot = True, color = 'blue', cmap = 'YlGn');


# In[ ]:


# This shows the hours per week according to the education of the person
sns.set(rc={'figure.figsize':(12,8)})
sns_grad = sns.barplot(x = df_adult_eda['education'], y = df_adult_eda['hours.per.week'], data = df_adult_eda)
plt.setp(sns_grad.get_xticklabels(), rotation=90);


# In[ ]:


# This bar graph shows the difference of hours per week between male and female 
sns.set(style = 'whitegrid', rc={'figure.figsize':(8,6)})
sns.barplot(x = df_adult_eda['sex'], y = df_adult_eda['hours.per.week'], data = df_adult_eda,
            estimator = mean, hue = 'sex', palette = 'winter');


# In[ ]:


# Creating Pandas Series for the workclasses whose income is higher than 50K 
df_ = df_adult_eda.loc[df_adult_eda['income'] == '>50K',['workclass']]
workclass_types = df_['workclass'].value_counts()
labels = list(workclass_types.index)
aggregate = list(workclass_types)

# This Pie chat shows the Percentage of different workclass who earns more than 50K
plt.pie(aggregate, labels = labels, autopct='%1.2f%%', shadow=True)
plt.legend(labels, loc = 'best')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[ ]:


# Grouping people by their education
education_size = df_adult_eda.groupby('education').size()

# Grouping people who earns more than 50K by their education
more_income = df_adult_eda.loc[df_adult_eda['income'] == '>50K', ['education']].groupby('education').size()

sns.set(style = 'dark')
plt.rcParams['figure.figsize'] = [15, 9]
fig, ax = plt.subplots(1,2)

# Setting axes Labels and Titles
ax[0].set_ylabel("Education")
ax[0].set_xlabel("No. of People")
ax[1].set_xlabel("No. of People")
ax[0].title.set_text("People grouped by their Education")
ax[1].title.set_text("People who're earning more than 50K")

# Barplot for the people grouped by their education
sns_ed_1 = sns.barplot(x = list(education_size), y = list(education_size.index), palette = 'winter',
                       order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'Bachelors', 'Doctorate',
                                'Assoc-acdm', 'Assoc-voc', 'HS-grad', 'Masters', 'Prof-school', 'Some-college'], ax = ax[0])

# Barplot for the people who earns more than 50K grouped by their education
sns_ed_2 = sns.barplot(x = list(more_income), y = list(more_income.index), palette = 'winter',
                       order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'Bachelors', 'Doctorate',
                                'Assoc-acdm', 'Assoc-voc', 'HS-grad', 'Masters', 'Prof-school', 'Some-college'], ax = ax[1])

#plt.setp(sns_ed_1.get_xticklabels(), rotation = 90);
#plt.setp(sns_ed_2.get_xticklabels(), rotation = 90);


# In[ ]:


df_adult_eda['occupation'].unique()


# In[ ]:


# Grouping people according to their country and their income
df_adult_eda_ = df_adult_eda[df_adult_eda['native.country'] != '?']
native_more = df_adult_eda_.loc[df_adult_eda_['income'] == '>50K',['native.country']].groupby('native.country').size()
native_less = df_adult_eda_.loc[df_adult_eda_['income'] == '<=50K',['native.country']].groupby('native.country').size()

index_more = list(native_more.index)
index_less = list(native_less.index)

# Checking if the Countries in both aspects are same or not
print(index_more)
print(len(index_more))
print(index_less)
print(len(index_less))


# In[ ]:


# Checking which Countries are not in the list
[country for country in index_less if country not in index_more]


# In[ ]:


# Making DataFrames of the Data
df_more = pd.DataFrame({'Countries' : index_more, '>50K' : list(native_more) })
df_less = pd.DataFrame({'Countries' : index_less, '<=50K' : list(native_less) })

# Adding the entries of the missing countries
df_more.loc[40] = 'Holand-Netherlands', 0
df_more.loc[41] = 'Outlying-US(Guam-USVI-etc)', 0

df_more


# In[ ]:


# Merging both the Data Frames to be used for plotting
df_fin = pd.merge(df_less, df_more, on = 'Countries')

df_fin


# In[ ]:


sns.set(style = 'whitegrid')
plt.rcParams['figure.figsize'] = [20,10]
# Dropping the United States Row as there's a disparity between US and other Countries
df_fin = df_fin.drop([38])

# This Bar plot shows which country's people after US make more than 50K a year

sns_ = sns.barplot(x = df_fin['Countries'], y = df_fin['>50K'], data = df_fin, palette = 'winter')
sns_.title.set_text("People who're earning more than 50K")

plt.setp(sns_.get_xticklabels(), rotation = 90);


# In[ ]:


# This Bar plot shows which country's people after US make less than 50K a year

sns__ = sns.barplot(x = df_fin['Countries'], y = df_fin['<=50K'], data = df_fin, palette = 'winter')
sns__.title.set_text("People who're earning less than 50K")

plt.setp(sns__.get_xticklabels(), rotation = 90);


# In[ ]:


# Setting Parameters
plt.rcParams['figure.figsize'] = [15,15]
sns.set(style = 'darkgrid')

# This Violin plot show how capital gain, loss, hours per week and education vary with the race of the people
plt.subplot(2,2,1)
sns.violinplot(x = df_adult_eda['race'], y = df_adult_eda['capital.gain'], data = df_adult_eda);
plt.subplot(2,2,2)
sns.violinplot(x = df_adult_eda['race'], y = df_adult_eda['capital.loss'], data = df_adult_eda);
plt.subplot(2,2,3)
sns.violinplot(x = df_adult_eda['race'], y = df_adult_eda['hours.per.week'], data = df_adult_eda);
plt.subplot(2,2,4)
sns.violinplot(x = df_adult_eda['race'], y = df_adult_eda['education.num'], data = df_adult_eda);


# In[ ]:


# Setting Parameters
plt.rcParams['figure.figsize'] = [15,8]
fig, ax = plt.subplots(1,2)

# Setting axes Labels and Titles
ax[0].set_ylabel("No. of People")
ax[0].set_xlabel("Relationship Status")
ax[1].set_xlabel("Relationship Status")
ax[0].title.set_text("People who're earning less than 50K")
ax[1].title.set_text("People who're earning more than 50K")

# Grouping people according to their Income and Relationship Status 
rel_less = df_adult_eda.loc[df_adult_eda['income'] == '<=50K',['relationship']].groupby('relationship').size()
rel_more = df_adult_eda.loc[df_adult_eda['income'] == '>50K',['relationship']].groupby('relationship').size()

# This barplot shows the No.of people earning more or less than 50K according to their Relationship Status
sns_rel_1 = sns.barplot(x = list(rel_less.index), y = list(rel_less), ax = ax[0])
sns_rel_2= sns.barplot(x = list(rel_more.index), y = list(rel_more), ax = ax[1])

plt.setp(sns_rel_1.get_xticklabels(), rotation = 60);
plt.setp(sns_rel_2.get_xticklabels(), rotation = 60);


# In[ ]:


# Setting axes Labels and Titles 
fig, ax = plt.subplots(1,2)
ax[0].set_xlabel('Race')
ax[1].set_xlabel('Race')
ax[0].set_ylabel('No. of People')
ax[0].title.set_text("People who're earning less than 50K")
ax[1].title.set_text("People who're earning more than 50K")

# Grouping People according to their race and income
race_less = df_adult_eda.loc[df_adult_eda['income'] == '<=50K'].groupby('race').size()
race_more = df_adult_eda.loc[df_adult_eda['income'] == '>50K'].groupby('race').size()

# This barplot shows the no.of people earning more or less than 50K according to their races
sns_race_1 = sns.barplot(x = list(race_less.index), y = list(race_less), ax = ax[0],
                         order = ['White', 'Black','Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'])
sns_race_2 = sns.barplot(x = list(race_more.index), y = list(race_more), ax = ax[1],
                        order = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'])

plt.setp(sns_race_1.get_xticklabels(), rotation = 90);
plt.setp(sns_race_2.get_xticklabels(), rotation = 90);


# ## Preprocessing

# In[ ]:


# Copying the eda adult dataFrame and reseting the index
df_adult = df_adult_eda.copy()

df_adult = df_adult.reset_index(drop = True)
df_adult.head()


# In[ ]:


df_adult.describe()


# In[ ]:


# Removing the unkown occupations
df_adult = df_adult[df_adult.occupation != '?']

print (df_adult['occupation'].value_counts())


# In[ ]:


df_adult.head()


# In[ ]:


# Changing the income column into Numerical Value
df_adult['income'] = df_adult['income'].map({'<=50K':0, '>50K':1})


# In[ ]:


df_adult['income'].value_counts()


# In[ ]:


# Changing the Categorical Values to Numerical values using the sklearns Label Encoder
from sklearn.preprocessing import LabelEncoder

categorical_features = list(df_adult.select_dtypes(include=['object']).columns)
label_encoder_feat = {}
for i, feature in enumerate(categorical_features):
    label_encoder_feat[feature] = LabelEncoder()
    df_adult[feature] = label_encoder_feat[feature].fit_transform(df_adult[feature])

df_adult.head()


# The label encoders are saved for each of the feature converted so that they can be decoded at the end. <br>
#  -- for feature, encoder in label_encoder_feat.items(): <br>
#     -- df_adult[feature] = encoder.inverse_transform(df_adult[feature])

# In[ ]:


# Shuffling the Data Set
from sklearn.utils import shuffle
df_adult = shuffle(df_adult)

# Splitting the data set into train and test set
from sklearn.model_selection import train_test_split

features_ = df_adult.drop(columns = ['income', 'education.num'])
target = df_adult['income']
X_train, X_test, y_train, y_test = train_test_split(features_, target, test_size = 0.3,random_state = 0)

print ("Train data set size : ", X_train.shape)
print ("Test data set size : ", X_test.shape)


# In[ ]:


# Plotting the feature importances using the Boosted Gradient Descent
from xgboost import XGBClassifier
from xgboost import plot_importance

# Training the model
model = XGBClassifier()
model_importance = model.fit(X_train, y_train)

# Plotting the Feature importance bar graph
plt.rcParams['figure.figsize'] = [14,12]
sns.set(style = 'darkgrid')
plot_importance(model_importance);


# ## Machine Learning Models

# Model-1 Logistic Regression

# In[ ]:


# Training the model_1
logistic = LogisticRegression(C = 0.5, max_iter = 500)
model_1 = logistic.fit(X_train, y_train)

# Predictions
pred_1 = model_1.predict(X_test)

print ("The accuracy of model 1 : ",accuracy_score(y_test, pred_1))
print ("The f1 score of model 1 : ", f1_score(y_test, pred_1, average = 'binary'))


# Model-2 Random Forest Classifier

# In[ ]:


# Training the model_2
R_forest = RandomForestClassifier(n_estimators = 200)
model_2 = R_forest.fit(X_train, y_train)

# Predictions
pred_2 = model_2.predict(X_test)

print ("The accuracy of model 2 : ",accuracy_score(y_test, pred_2))
print ("The f1 score of model 2 : ", f1_score(y_test, pred_2, average = 'binary'))


# Model-3 Boosted Gradient Descent

# In[ ]:


# Training the model 3
boosted_gd = XGBClassifier(learning_rate = 0.35, n_estimator = 500)
model_3 = boosted_gd.fit(X_train, y_train)

# Predictions
pred_3 = model_3.predict(X_test)

print ("The accuracy of model 3 : ",accuracy_score(y_test, pred_3))
print ("The f1 score of model 3 : ", f1_score(y_test, pred_3, average = 'binary'))


# Model-4 Bernoulli NB

# In[ ]:


# Training the model 4
NB = BernoulliNB(alpha = 0.3)
model_4 = NB.fit(X_train, y_train)

# Predictions
pred_4 = model_4.predict(X_test)

print ("The accuracy of model 4 : ",accuracy_score(y_test, pred_4))
print ("The f1 score of model 4 : ", f1_score(y_test, pred_4, average = 'binary'))


# Model-5 Support Vector Classifier

# In[ ]:


# Training the model 5
svc = SVC(kernel = 'rbf', max_iter = 1000, probability = True)
model_5 = svc.fit(X_train, y_train)

# Predictions
pred_5 = model_5.predict(X_test)

print ("The accuracy of model 5 : ",accuracy_score(y_test, pred_5))
print ("The f1 score of model 5 : ", f1_score(y_test, pred_5, average = 'binary'))


# ## Analysis of the model performances

# #### Classification Reports

# In[ ]:


list_pred = [pred_1, pred_2, pred_3, pred_4, pred_5]
model_names = ["Logistic Regression", "Random Forest Classifier", "Boosted Gradient Descent", "Bernoulli NB", "SVC"]

for i, predictions in enumerate(list_pred) :
    print ("Classification Report of ", model_names[i])
    print ()
    print (classification_report(y_test, predictions, target_names = ["<=50K", ">50K"]))


# #### Consfusion Matrix for the Classifier

# In[ ]:


for i, pred in enumerate(list_pred) :
    print ("The Confusion Matrix of : ", model_names[i])
    print (pd.DataFrame(confusion_matrix(y_test, pred)))
    print ()


# In[ ]:


# ROC Curve for the classification models

models = [model_1, model_2, model_3, model_4, model_5]

# Setting the parameters for the ROC Curve
plt.rcParams['figure.figsize'] = [10,8]
plt.style.use("bmh")

color = ['red', 'blue', 'green', 'fuchsia', 'cyan']
plt.title("ROC CURVE", fontsize = 15)
plt.xlabel("Specificity", fontsize = 15)
plt.ylabel("Sensitivity", fontsize = 15)
i = 1

for i, model in enumerate(models) :
    prob = model.predict_proba(X_test)
    prob_positive = prob[:,1]
    fpr, tpr, threshold = roc_curve(y_test, prob_positive)
    plt.plot(fpr, tpr, color = color[i])
    plt.gca().legend(model_names, loc = 'lower right', frameon = True)

plt.plot([0,1],[0,1], linestyle = '--', color = 'black')
plt.show()

