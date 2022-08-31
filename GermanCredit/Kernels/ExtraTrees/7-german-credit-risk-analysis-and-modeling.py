#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex

sns.set_style('whitegrid')

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 0. Loading and Understanding the Dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv", index_col=0)
df.head()


# In[ ]:


df.info()


# There are missing values in columns "Saving accounts" and "Checking accounts".

# In[ ]:


display(Markdown("#### Explore the Values of Text Columns:"))
cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']
for col in cols:
    line = "**" + col + ":** "
    for v in df[col].unique():
        line = line + str(v) + ", "
    display(Markdown(line))


# From above exploration:
# - Columns "Housing", "Saving accounts" and "Checking accounts" are **Ordinal** data.    
# - Columns "Sex", "Purpose" and "Risk" are **Categorical** data.    

# ## 1. Data Pre-processing For Ordinal Columns

# Known from Content in the dataset page, column "Job" is **Ordinal** data that:
# - 0 - unskilled and non-resident, 
# - 1 - unskilled and resident, 
# - 2 - skilled, 
# - 3 - highly skilled
# 
# SO, apply the save logic to other **Ordinal** columns "Housing", "Saving accounts" and "Checking accounts".
# 
# For "Saving accounts" and "Checking accounts":
# - 0 - missing value, as UNKNOWN 
# - 1 - little
# - 2 - moderate
# - 3 - quite rich
# - 4 - rich
# 
# For "Housing":
# - 0 - free
# - 1 - rent
# - 2 - own

# In[ ]:


# label encode account quality and fill NaN with 0
def SC_LabelEncoder(text):
    if text == "little":
        return 1
    elif text == "moderate":
        return 2
    elif text == "quite rich":
        return 3
    elif text == "rich":
        return 4
    else:
        return 0

df["Saving accounts"] = df["Saving accounts"].apply(SC_LabelEncoder)
df["Checking account"] = df["Checking account"].apply(SC_LabelEncoder)


# In[ ]:


# label encode account quality and fill NaN with 0
def H_LabelEncoder(text):
    if text == "free":
        return 0
    elif text == "rent":
        return 1
    elif text == "own":
        return 2

df["Housing"] = df["Housing"].apply(H_LabelEncoder)


# ## 2. EDA

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.histplot(df, x='Age', bins=30, hue="Sex", ax=ax[0]).set_title("Age/Sex Distribution");
sns.boxplot(data=df, x="Sex", y="Age", ax=ax[1]).set_title("Age/Sex Distribution");

fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(data=df, x='Risk', y='Age', ax=ax[0]).set_title("Age Distribution with Risk");
sns.countplot(data=df, x="Sex", hue="Risk", ax=ax[1]).set_title("Sex Distribution with Risk");


# **Analysis:** 
# - Age does not affect the risk rating much. 
# - Males take more count of credit from Bank.
# - Males have lower percentage of bad rating than woman.

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.histplot(df, x='Credit amount', bins=30, ax=ax[0]).set_title("Credit Amount (in Deutsch Mark) Distribution");
sns.histplot(df, x='Duration', bins=30, ax=ax[1]).set_title("Duration (in month) Distribution");

fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(data=df, x='Risk', y='Credit amount', ax=ax[0]).set_title("Credit Amount (in Deutsch Mark) Distribution with Risk");
sns.boxplot(data=df, x='Risk', y='Duration', ax=ax[1]).set_title("Duration (in month) Distribution with Risk");


# **Analysis:** The higher credit amount and longer duration means higher risk to the bank.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,5))
sns.countplot(data=df, x="Job", hue="Risk", ax=ax[0]).set_title("Job Distribution with Risk");
sns.countplot(data=df, x="Housing", hue="Risk", ax=ax[1]).set_title("Housing Distribution with Risk");


# **Analysis:** 
# - Most of people in records have job skill level 2, but the job skill level does not affect the risk rating much.
# - People who own a house means low risk and good rating to the bank.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,5))
sns.countplot(data=df, x="Saving accounts", hue="Risk", ax=ax[0]).set_title("Saving Account Quality Distribution with Risk");
sns.countplot(data=df, x="Checking account", hue="Risk", ax=ax[1]).set_title("Checking Account Quality Distribution with Risk");


# **Analysis** (since 0 means unknown, only discuss quality level 1 to 4):
# - The person with more saving means less risk to the bank, but most people in the records have little saving (not rich!)
# - About half of people who have little checking account are considered as bad rating in risk.
# - About 20% of people who have moderate checking account are considered as bad rating in risk.

# In[ ]:


sns.pairplot(df[['Age', 'Job', 'Housing', 'Saving accounts', 
                 'Checking account', 'Credit amount', 'Duration', "Risk"]], hue="Risk");


# In[ ]:


corr = df[['Age', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration']].corr()
sns.set(rc={'figure.figsize':(11,7)})
sns.heatmap(corr,linewidths=.5, annot=True, cmap="YlGnBu",mask=np.triu(np.ones_like(corr, dtype=np.bool)))    .set_title("Pearson Correlations Heatmap");


# **Analysis:** The Credit Amount is HIGHLY and POSITIVELY related to the Duration.

# ## 3. Data Pre-processing For Discrete Categorical Columns

# In[ ]:


# use LabelEncoder() to encode other categorical columns:
for col in ["Sex", "Purpose", "Risk"]:
    le = LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])
df.head()


# ## 4. Clustering

# In[ ]:


cdf = df.drop("Risk", axis=1)


# ### 4.1 Find the Best Number of Clusters for K-Means and Analysis

# #### Start with applying Elbow Method.

# In[ ]:


inertias = []

for i in range(2,16):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(cdf)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
plt.title('Inertias v.s. N_Clusters')
plt.plot(np.arange(2,16),inertias, marker='o', lw=2);


# **Analysis:** The "elbow" in above chart is indicated  at 4. The number of clusters chosen should therefore be 4. 
# #### With 4 Clusters:

# In[ ]:


km = KMeans(n_clusters=4, random_state=0)
clusters = km.fit_predict(cdf)


# In[ ]:


df_clustered = cdf[['Age', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration']]
df_clustered["Cluster"] = clusters
sns.pairplot(df_clustered[['Age', 'Job', 'Housing', 'Saving accounts', 
                 'Checking account', 'Credit amount', 'Duration', "Cluster"]], hue="Cluster");


# **Analysis:** Compare the matrix in EDA, this 4-clustered matrix plot show a clearer grouping boundaries than the given Good/Bad Risk rating.

# ### 4.2 Cluster the data to Two Group and Compare with Given Good/Bad Risk Rating
# Use K-means to cluster people in the records into 2 group and check if the result closed to given two risk groups.

# In[ ]:


km = KMeans(n_clusters=2, random_state=0)
clusters = km.fit_predict(cdf)


# In[ ]:


display(Markdown("In encoded Risk column, good = 1 and bad = 0, but the predicted clusters 0 and 1 does not have the same meaning. Thus, whether the predicted clusters is equal or opposite to the given risk, the higher TRUE percentage will be the accuracy rate."))
acc = max((sum(clusters == df["Risk"]) / len(df)), (sum(clusters != df["Risk"]) / len(df)))
display(Markdown("The accuracy rate of 2-Means clustering is " + str(acc)))


# ## 5. Predicting the Risk
# **Based on given Risk column**

# In[ ]:


X, y = df.drop("Risk", axis=1), df["Risk"]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=0)


# ### 5.1 K-Nearest Neighbors Classification

# In[ ]:


max_score = 0
max_k = 0
for k in range(1, 100):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train,y_train)
    score = f1_score(y_test, neigh.predict(X_test))
    if score > max_score:
        max_k = k
        max_score = score

display(Markdown("If use K-Nearest Neighbors Classification, the k should be " + str(max_k) + " to get best prediction, and then the  mean accuracy is " + str(max_score)))


# ### 5.2 Modeling by Other Classifiers
# Since KNN algorithm cost lots of memory and time for prediction, this section want to try some more classifiers.
# #### Model Selection with Cross Validate

# In[ ]:


# define models
Models = {
    "SVC": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GaussianNaiveBayes": GaussianNB()
}


# In[ ]:


cv_results = pd.DataFrame(columns=['model', 'train_score', 'test_score'])
for key in Models.keys():
    cv_res = model_selection.cross_validate(Models[key], X_train, y_train, 
                                             return_train_score=True,
                                             scoring="f1",
                                             cv=5, n_jobs=-1)
    res = {
        'model': key, 
        'train_score': cv_res["train_score"].mean(), 
        'test_score': cv_res["test_score"].mean(),
        'fit_time': cv_res["fit_time"].mean(),
        'score_time': cv_res["score_time"].mean(),
        }
    cv_results = cv_results.append(res, ignore_index=True)
    print("CV for model:", key, "done.")
cv_results


# #### Evaluate Model on Testing Set
# - Random Forest Classifier gives a good result on both train_score and test_score.
# - SVC and Gaussian Naive Bayes show the less over-fiting.
# - Gaussian Naive Bayes Classifier has least runtime.
# - Random Forest Classifier would tell feature importances, while SVC only return coef_ in the case of a linear kernel, which will be too slow.
# 
# **Taking all this into consideration, Random Forest Classifier is chose to evaluate on testing set:**

# In[ ]:


rf = Models["RandomForest"].fit(X_train, y_train)
print('f1_score:', f1_score(y_test, rf.predict(X_test)))


# #### Feature Importance Discussion

# In[ ]:


feature_importance = pd.DataFrame()
feature_importance["feature"] = X_train.columns
feature_importance["importance"] = rf.feature_importances_
feature_importance = feature_importance.sort_values("importance", ascending=False)
feature_importance

