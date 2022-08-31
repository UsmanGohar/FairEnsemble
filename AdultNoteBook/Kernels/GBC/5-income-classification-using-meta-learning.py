#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income Classification using Meta Learning

# In[ ]:


#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from numpy import mean, std


# In[ ]:


#reading the dataset and converting it to dataframe
df = pd.read_csv("../input/adult-census-income/adult.csv")


# In[ ]:


#Viewing the top 5 rows of our dataset
df.head()


# ## Exploratory Data Analysis

# **Income - Target column**

# In[ ]:


sns.countplot(df.income)


# *As we can see, there is a **class imbalance**. The ">50K" class is comparatively very less. So, we will do **Random Over-Sampling** during preprocessing.*
# 

# **Age**

# In[ ]:


sns.distplot(df[df.income=='<=50K'].age, color='g')
sns.distplot(df[df.income=='>50K'].age, color='r')


# *We can observe a rough margin **around 30**. We will divide age into 2 parts ie. under 30 and over 30. We need to check if its useful for our model during testing.*

# **Workclass**

# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.workclass, hue=df.income, palette='tab10')


# *Majority of the data falls under **Private**. So, we will convert this into Private and not-Private.*

# **fnlwgt**

# In[ ]:


sns.distplot(df[df.income=='<=50K'].fnlwgt, color='r')
sns.distplot(df[df.income=='>50K'].fnlwgt, color='g')


# *This is a very **ambiguous** attribute. Will check during testing.*

# **Education**

# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.education, hue=df.income, palette='muted')


# **education.num**

# In[ ]:


sns.countplot(df["education.num"], hue=df.income)


# **marital.status**

# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df['marital.status'], hue=df.income)


# *We observe that the majority of ">50K" class is **Married-civ-spouse**. So we ll make it 1 and others 0*

# **occupation**

# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.occupation, hue=df.income, palette='rocket')


# **relationship**

# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.relationship, hue=df.income, palette='muted')


# **race**

# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.race, hue=df.income, palette='Set2')


# **sex**

# In[ ]:


plt.xticks(rotation=90)
sns.countplot(df.sex, hue=df.income)


# **capital.gain**

# In[ ]:


df['capital.gain'].value_counts()


# **capital.loss**

# In[ ]:


df['capital.loss'].value_counts()


# **hours.per.week**

# In[ ]:


sns.distplot(df[df.income=='<=50K']['hours.per.week'], color='b')
sns.distplot(df[df.income=='>50K']['hours.per.week'], color='r')


# **native.country**

# In[ ]:


df['native.country'].value_counts()


# ## Preprocessing

# ### Finding and Handling Missing Data
# 
# *Observing the dataset, I found that the null values are marked as "?", So, we will now convert them to numpy.nan(null values).*

# In[ ]:


df[df.select_dtypes("object") =="?"] = np.nan
nans = df.isnull().sum()
if len(nans[nans>0]):
    print("Missing values detected.\n")
    print(nans[nans>0])
else:
    print("No missing values. You are good to go.")


# In[ ]:


#majority of the values are "Private". Lets fill the missing values as "Private".
df.workclass.fillna("Private", inplace=True)

df.occupation.fillna(method='bfill', inplace=True)

#majority of the values are "United-States". Lets fill the missing values as "United-States".
df['native.country'].fillna("United-States", inplace=True)

print("Handled missing values successfully.")


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d

class MyLabelEncoder(LabelEncoder):

    def fit(self, y, arr=[]):
        y = column_or_1d(y, warn=True)
        if arr == []:
            arr=y
        self.classes_ = pd.Series(arr).unique()
        return self

le = MyLabelEncoder()


# ### Feature Engineering and Encoding the columns

# In[ ]:


# age_enc = pd.cut(df.age, bins=(0,25,45,65,100), labels=(0,1,2,3))
df['age_enc'] = df.age.apply(lambda x: 1 if x > 30 else 0)

def prep_workclass(x):
    if x == 'Never-worked' or x == 'Without-pay':
        return 0
    elif x == 'Private':
        return 1
    elif x == 'State-gov' or x == 'Local-gov' or x == 'Federal-gov':
        return 2
    elif x == 'Self-emp-not-inc':
        return 3
    else:
        return 4

df['workclass_enc'] = df.workclass.apply(prep_workclass)

df['fnlwgt_enc'] = df.fnlwgt.apply(lambda x: 0 if x>200000 else 1)

le.fit(df.education, arr=['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th', 
                                             'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate'])
df['education_enc'] = le.transform(df.education)


df['education.num_enc'] = df['education.num'].apply(lambda x: 1 if x>=9 else 0)

df['marital.status_enc'] = df['marital.status'].apply(lambda x: 1 if x=='Married-civ-spouse' or x == 'Married-AF-spouse' else 0)

def prep_occupation(x):
    if x in ['Prof-specialty', 'Exec-managerial', 'Tech-support', 'Protective-serv']:
        return 2
    elif x in ['Sales', 'Craft-repair']:
        return 1
    else:
        return 0

df['occupation_enc'] = df.occupation.apply(prep_occupation)

df['relationship_enc'] = df.relationship.apply(lambda x: 1 if x in ['Husband', 'Wife'] else 0)

df['race_enc'] = df.race.apply(lambda x: 1 if x=='White' else 0)

df['sex_enc'] = df.sex.apply(lambda x: 1 if x=='Male' else 0)

df['capital.gain_enc'] = pd.cut(df["capital.gain"], 
                                bins=[-1,0,df[df["capital.gain"]>0]["capital.gain"].median(), df["capital.gain"].max()], labels=(0,1,2)).astype('int64')

df['capital.loss_enc'] = pd.cut(df["capital.loss"], 
                                bins=[-1,0,df[df["capital.loss"]>0]["capital.loss"].median(), df["capital.loss"].max()], labels=(0,1,2)).astype('int64')

# hpw_enc = pd.cut(df['hours.per.week'], bins= (0,30,40,53,168), labels=(0,1,2,3))
df['hours.per.week_enc'] = pd.qcut(df['hours.per.week'], q=5, labels=(0,1,2,3), duplicates='drop').astype('int64')

df['native.country_enc'] = df['native.country'].apply(lambda x: 1 if x=='United-States' else 0)

df['income_enc'] = df.income.apply(lambda x: 1 if x==">50K" else 0)

print("Encoding complete.")


# In[ ]:


df.select_dtypes("object").info()


# In[ ]:


#dropping encoded columns - education, sex, income
df.drop(['education', 'sex', 'income'], 1, inplace=True)


# ### Label Encoding without Feature Engineering

# In[ ]:


for feature in df.select_dtypes("object").columns:
    df[feature]=le.fit_transform(df[feature])


# ### Feature Selection

# In[ ]:


df.info()


# In[ ]:


#Visualizing the pearson correlation with the target class
pcorr = df.drop('income_enc',1).corrwith(df.income_enc)
plt.figure(figsize=(10,6))
plt.title("Pearson Correlation of Features with Income")
plt.xlabel("Features")
plt.ylabel("Correlation Coeff")
plt.xticks(rotation=90)
plt.bar(pcorr.index, list(map(abs,pcorr.values)))


# From the pearson correlation plot, we can see that correlation of few columns are very **low** with the target column, so, we ll drop them.

# In[ ]:


df.drop(['workclass', 'fnlwgt','occupation', 'race', 'native.country', 'fnlwgt_enc', 'race_enc', 'native.country_enc'], 1, inplace=True)


# In[ ]:


sns.heatmap(df.corr().apply(abs))


# **Dropping redundant features**

# We can see that **education_enc, education.num_enc and education.num** as well as **relationship_enc and marital.status_enc** have **high correlation**. So, we will only keep one of them based on their correlation with income_enc.
# 
# We also have some redundant feautres as we have engineered features from them(age, capital.gain, etc.).

# In[ ]:


df.drop(['age', 'education.num_enc', 'education_enc', 'marital.status_enc', 'capital.gain', 'capital.loss', 'hours.per.week'], 1, inplace = True)


# In[ ]:


df.info()


# In[ ]:


X = df.drop('income_enc', 1)
y = df.income_enc


# ### Train Test Split (3:1)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# In[ ]:


print("No. of rows in training data:",X_train.shape[0])
print("No. of rows in testing data:",X_test.shape[0])


# ### Random Over Sampling

# *We can see the class imbalance in our target. This results in models that have poor predictive performance, specifically for the minority class. So, we need to random over sampling*

# In[ ]:


oversample = RandomOverSampler(sampling_strategy=0.5) #50% oversampling
X_over, y_over = oversample.fit_resample(X_train, y_train)


# In[ ]:


y_over.value_counts()


# ## Model Preparation

# In[ ]:


#Model Imports
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[ ]:


seed= 42


# In[ ]:


models = {
    'LR':LogisticRegression(random_state=seed),
    'SVC':SVC(random_state=seed),
    'AB':AdaBoostClassifier(random_state=seed),
    'ET':ExtraTreesClassifier(random_state=seed),
    'GB':GradientBoostingClassifier(random_state=seed),
    'RF':RandomForestClassifier(random_state=seed),
    'XGB':XGBClassifier(random_state=seed),
    'LGBM':LGBMClassifier(random_state=seed)
    }


# In[ ]:


# evaluate a give model using cross-validation
def evaluate_models(model, xtrain, ytrain):
    cv = StratifiedKFold(shuffle=True, random_state=seed)
    scores = cross_val_score(model, xtrain, ytrain, scoring='accuracy', cv=cv, error_score='raise')
    return scores

def plot_scores(xval,yval,show_value=False):
    plt.ylim(ymax = max(yval)+0.5, ymin = min(yval)-0.5)
    plt.xticks(rotation=45)
    s = sns.barplot(xval,yval)
    if show_value:
        for x,y in zip(range(len(yval)),yval):
            s.text(x,y+0.1,round(y,2),ha="center")


# In[ ]:


# evaluate the models and store results for 100% oversampled minority class
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_models(model, X_train, y_train) 
    results.append(scores) 
    names.append(name) 
    print('*%s %.3f (%.3f)' % (name, mean(scores), std(scores)))


# In[ ]:


plt.boxplot(results, labels=names, showmeans=True)
plt.show() 


# In[ ]:


param_grids = {
    'LR':{'C':[0.001,0.01,0.1,1,10]},
    'SVC':{'gamma':[0.01,0.02,0.05,0.08,0.1], 'C':range(1,8)},
    
    'AB':{'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [100, 200, 500]},
    
    'ET':{'max_depth':[5,8,10,12], 'min_samples_split': [5,9,12],
          'n_estimators': [100,200,500,800]},
    
    'GB':{'learning_rate': [0.05, 0.1, 0.2], 'max_depth':[3,5,9],
          'min_samples_split': [5,7,9], 'n_estimators': [100,200,500],
          'subsample':[0.5,0.7,0.9]},
    
    'RF':{'max_depth':[3,5,9,15], 'n_estimators': [100, 200, 500, 1000],
          'learning_rate': [0.05, 0.1, 0.2], 'min_samples_split': [5,9,12]},
    
    'XGB':{'max_depth':[3,5,7,9], 'n_estimators': [100, 200, 500],
           'learning_rate': [0.05, 0.1, 0.2], 'subsample':[0.5,0.7,0.9]},
    
    'LGBM':{'n_estimators': [100,200,500],'learning_rate': [0.05, 0.1, 0.2],
            'subsample':[0.5,0.7,0.9],'num_leaves': [25,31,50]}
}


# In[ ]:


# !pip install sklearn-deap
# from evolutionary_search import EvolutionaryAlgorithmSearchCV


# In[ ]:


# evaluate the models and store results
# best_params = []
# names= []
# for name, param_grid, model in zip(param_grids.keys(), param_grids.values(), models.values()):
#     eascv = EvolutionaryAlgorithmSearchCV(model, param_grid, verbose=3, cv=3)
#     eascv.fit(X_train,y_train)
#     names.append(name)
#     best_params.append(eascv.best_params_)
#     print(name)
#     print("best score:",eascv.best_score_)
#     print("best params:",eascv.best_params_)


# In[ ]:


best_params=[
    {'C': 10},
    {'gamma': 0.1, 'C': 2},
    {'learning_rate': 0.1, 'n_estimators': 500},
    {'max_depth': 12, 'min_samples_split': 9, 'n_estimators': 100},
    {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_split': 9, 'n_estimators': 200, 'subsample': 0.9},
    {'max_depth': 9, 'n_estimators': 200, 'min_samples_split': 5},
    {'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.1, 'subsample': 0.9},
    {'n_estimators': 100, 'learning_rate': 0.05, 'subsample': 0.9, 'num_leaves': 25}
            ]


# In[ ]:


models = [
    ('LR',LogisticRegression(random_state=seed)),
    ('SVC',SVC(random_state=seed)),
    ('AB',AdaBoostClassifier(random_state=seed)),
    ('ET',ExtraTreesClassifier(random_state=seed)),
    ('GB',GradientBoostingClassifier(random_state=seed)),
    ('RF',RandomForestClassifier(random_state=seed)),
    ('XGB',XGBClassifier(random_state=seed)),
    ('LGBM',LGBMClassifier(random_state=seed))
]


# In[ ]:


for model, param in zip(models, best_params):
    model[1].set_params(**param)


# In[ ]:


models.append(('MLModel',StackingClassifier(estimators = models[:-1])))


# In[ ]:


scores=[]
preds=[]
for model in models:
    model[1].fit(X_train,y_train)
    print(model[0],"trained.")
    scores.append(model[1].score(X_test,y_test))
    preds.append(model[1].predict(X_test))
print("Results are ready.")


# ## Using Classification Based on Assocation

# In[ ]:


get_ipython().system('pip install pyarc==1.0.23')
get_ipython().system('pip install pyfim')
from pyarc import CBA, TransactionDB


# In[ ]:


txns_train = TransactionDB.from_DataFrame(X_train.join(y_train))
txns_test = TransactionDB.from_DataFrame(X_test.join(y_test))


cba = CBA(support=0.15, confidence=0.5, algorithm="m1")
cba.fit(txns_train)


# In[ ]:


cba_score = cba.rule_model_accuracy(txns_test) 
scores.append(cba_score)
models.append(["CBA"])


# In[ ]:


model_names= [i[0] for i in models]
scores = list(map(lambda x: x*100, scores))


# In[ ]:


plot_scores(model_names, scores, True)

