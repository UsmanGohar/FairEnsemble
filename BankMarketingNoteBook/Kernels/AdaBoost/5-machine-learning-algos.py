#!/usr/bin/env python
# coding: utf-8

# # 10 Machine learning algorithms explained
# 
# This kernel gives an overview of different machine learning algorithms and explains how they work along with the python code. Mostly all the models are tuned with optimal hyperparameters.
# 
# ## Table of contents
# - [Basic EDA and Data preprocessing](#Basic-EDA-and-Data-preprocessing)
# - [Let's start with the algos](#Let's-start-with-the-algos)
#     1. [Regression](#1.-Regression)
#         - [Linear Regression](#Linear-Regression)
#         - [SGD Regression](#SGD-Regression)
#         - [Logistic Regression](#Logistic-Regression)
#     2. [Support Vector Machines](#2.-Support-Vector-Machines)    
#     3. [Naive Bayes](#3.-Naive-Bayes)
#     4. [K-Nearest Neighbors](#4.-K-Nearest-Neighbors)    
#     5. [Decision Tree](#5.-Decision-Tree)
#     6. [RandomForest](#6.-RandomForest)
#     7. [Adaptive Boosting](#7.-Adaptive-Boosting)    
#     8. [Gradient Boosting](#8.-Gradient-Boosting)
#         - [XGBoost](#XGBoost)
#         - [LightGBM](#LightGBM)
#         - [CatBoost](#CatBoost)
#     9. [Dimensionality reduction](#9.-Dimensionality-reduction)    
#         - [Principle Component Analysis](#Principle-Component-Analysis)
#         - [t-Distributed Stochastic Neighbor Embedding](#t-Distributed-Stochastic-Neighbor-Embedding)
#     10. [K-Means Clustering](#10.-K-Means-Clustering)

# In[1]:


get_ipython().system('pip install pydotplus')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 0)

import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact
from IPython.display import display, Image

from sklearn.externals.six import StringIO  
import pydotplus

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


# ## Basic EDA and Data preprocessing

# In[3]:


df = pd.read_csv('../input/bank-additional-full.csv', sep=';')
print(df.shape)
df.head()


# In[4]:


df['y'].value_counts(normalize=True)


# In[5]:


df.replace('unknown', np.nan, inplace=True)


# In[6]:


cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

for c in cols:
    df[c] = pd.factorize(df[c])[0]

df[cols] = df[cols].replace(-1, np.nan)    

df['y'] = df['y'].map({'no': 0, 'yes': 1})


# In[7]:


df.hist(figsize=(20, 20));


# In[8]:


df['default'].value_counts(normalize=True)


# In[9]:


df.drop('default', axis=1, inplace=True)


# In[10]:


cols = ['marital', 'job', 'housing', 'loan', 'education']

@interact
def plot_count(c=cols):
    plt.figure(figsize=(15, 35))
    plt.subplot(6, 2, 1)
    sns.countplot(c, data=df)
    plt.xticks(rotation=90)    
    plt.subplot(6, 2, 2)
    sns.countplot(c, data=df, hue='y')
    plt.xticks(rotation=90)


# In[11]:


nan_cols = df.isnull().sum() / df.shape[0]
nan_cols = nan_cols[nan_cols > 0].sort_values()
nan_cols


# In[12]:


cols = ['marital', 'job', 'housing', 'loan', 'education']

for c in cols:
    df[f'{c}_na'] = df[c].isnull().astype(np.int32)
    df[c] = df[c].fillna(df[c].median())

df.isnull().sum().any()    


# In[13]:


cont_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
             'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']         

cat_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week',
            'poutcome', 'marital_na', 'job_na', 'housing_na', 'loan_na', 'education_na']


# In[14]:


def print_metric(model, df, y, scaler=None):
    X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, random_state=7, stratify=y)
    mets = [accuracy_score, precision_score, recall_score, f1_score] 
    
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_met = pd.Series({m.__name__: m(y_train, train_preds) for m in mets})
    val_met = pd.Series({m.__name__: m(y_val, val_preds) for m in mets})
    
    if hasattr(model, 'predict_proba'):
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        train_met['roc_auc'] = roc_auc_score(y_train, train_probs)
        val_met['roc_auc'] = roc_auc_score(y_val, val_probs)
    
    met_df = pd.DataFrame()
    met_df['train'] = train_met
    met_df['valid'] = val_met
    
    display(met_df)


# In[15]:


def get_cross_val(model, train_X, train_y, cv):
    cv_scores = cross_val_score(model, train_X, train_y, scoring='roc_auc', cv=cv, n_jobs=2)
    print(f'CV Mean: {cv_scores.mean():.4f} \t CV Std: {cv_scores.std():.4f}')
    print(cv_scores)


# In[16]:


def grid_search(model, params, train_X, train_y, cv):
    grid_search = GridSearchCV(model, param_grid=grid_params, scoring='f1', n_jobs=-1, cv=cv, verbose=1)
    grid_search.fit(df, y)
    print(grid_search.best_score_)
    print(grid_search.best_params_)

    return grid_search.best_estimator_


# In[17]:


y = df.pop('y')
skf = StratifiedKFold(5, shuffle=True, random_state=7)


# ## Let's start with the algos

# ## 1. Regression
# 
# This is the most basic algorithm. There is no magic in it. All the concept is borrowed from the field of statistics. The main idea is to find the relationship between dependent and independent variables. Consider you have 2 variables x(independent) and y(dependent) we need to find the following relationship: 
# $$ y = B_0x + B_1 $$
# here $B_0$ and $B_1$ are learnable parameters.
# 
# Logistic regression is basically *LinearRegression + CrossEntropyLoss*. So in order to understand LogisticRegression, we must know what LinearRegression is.
# 
# ### Linear Regression
# `LinearRegression` from `Scikit Learn` uses covariance, mean and variance statistics to calculate $B_0$ and $B_1$. The basic formula is:
# $$ B_0 = {cov(x, y) \over variance(x)} $$
# 
# $$ B_1 = mean(y) - B_1 * mean(x) $$
# 
# <img src="https://4.bp.blogspot.com/-tWOGQiH9nUk/XDIJYqAQqkI/AAAAAAAADFs/f-8EnXVslIgyeXO1IlRHI3XzPu95gkrVgCLcBGAs/s1600/residuals.gif"></img>
# 
# 
# ### SGD Regression
# Usually, Linear regression uses gradient-descent to calculate gradients to update the parameters in order to reduce the loss. Here the loss is the sum of the distance between the points and the optimal regression line. This algorithm is available with `SGDRegressor` in Scikit-Learn. Each variable is assigned a random weight $W_i$ and this weight is changed with each gradient-descent step. The steps are as follows:
# 1. Calculate the predictions with current values of $W_1, W_2, W_3, ...$
# 2. Calculate the loss
# 3. Calculate the gradient of $W_i$ with respect to the loss.
# 4. Update the weights $W_i$ using gradient descent.
# 
# Gradient Descent step is defined as:
# $$ W_i = W_i - \alpha*dW_i $$
# here, alpha is learning-rate which is basically the step-size contributing the amount of gradient we want to consider in each step. <br>
# Check out [this video](https://www.youtube.com/watch?v=vMh0zPT0tLI) to understand gradient-descent visually.
# 
# ### Logistic Regression
# Now, in case of Logistic-Regression we use cross-entropy loss. Entropy defines the degree of chaos. First we need to squish our predictions between 0 and 1 to make it as the predicted probability and then this value is used to calculate the cross_entropy loss. The formulas are:
# $$ p_i = {1 \over {1 + e^{-W^Tx}}} $$
# $$ loss = -\sum_{i=0}^n y_ilog(p_i) + (1-y_i)log(1-p_i) $$
# here, $y_i$ is the $i_{th}$ target value and $W^Tx$ is the prediction. <br>
# Here is a [reference playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe) to understand this algorithm visually.

# In[18]:


model = LogisticRegression(C=4.6416, class_weight='balanced', n_jobs=-1, random_state=7)
get_cross_val(model, df, y, skf)
print_metric(model, df, y, scaler=StandardScaler())


# ### 2. Support Vector Machines
# 
# Support vector machines(SVM) are mostly used for classification problems although they can be used for regression too. The main idea is to find a hyperplane within the n-dimensional space in order to maximize the distance between hyperplane and data-points. These data-points are called support vectors and the distance is called the margin.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png" width="300"></img>
# 
# In the above case, the data is spread across the dimension space so a linear hyperplane can be drawn easily, but sometimes it's not possible to do that. In such cases, we use an RBF kernel.
# 
# <img src="https://blog-c7ff.kxcdn.com/blog/wp-content/uploads/2017/02/kernel.png" width="500"></img>

# In[19]:


model = SVC(random_state=7)
get_cross_val(model, df, y, skf)


# ## 3. Naive Bayes
# Naive Bayes uses **Bayes theorem** to predict a value given some data. Naive Bayes is called *naive* because it assumes that each input variable is independent of each other. It calculate the current hypothesis and uses prior knowledge given to it by the data to train the model and make predictions. The bayes theorem is as follows:
# $$ P({c \over d}) =  {{p({d \over c}) p(c)} \over {p(d)}} $$
# here,
# - $p({c \over d})$: probability that $d$ belongs to class $c$
# - $p({d \over c})$: probability of data of the class $c$
# - $p({c})$: probability of class $c$
# - $p({d})$ predictive prior probability i.e probability of data irrespective of the class.

# In[20]:


model = GaussianNB()
get_cross_val(model, df, y, skf)
print_metric(model, df, y, scaler=StandardScaler())


# ## 4. K-Nearest Neighbors
# 
# KNN is the simplest one. New test data is assigned the values which the nearest K neighbors possess. First, it finds it's K-nearest neighbors using either Euclidean distance or Hamming distance and then considers only those points to predict a value. In the case of the regression, the predicted value can be the average or weight average of the target values of those points and in case of classification, it takes the majority class among its peers.
# 
# <img src="https://www.fromthegenesis.com/wp-content/uploads/2018/09/K_NN_Ad.jpg" width="400"></img>

# In[21]:


model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
get_cross_val(model, df, y, skf)
print_metric(model, df, y, scaler=StandardScaler())


# ## 5. Decision Tree
# 
# Decision trees are basically similar to binary trees. Each node has 2 child nodes except the leaf nodes. We select a feature everytime we make a split at some node using some evaluation criterion. The commonly used criterion for regression is based on variance. Here, we iterate over all the features and in each feature we iterate over the possible points between the lower-bound and upper-bound of the feature values and calculate information gain(IG). To calculate information we first calculate the initial entropy and final entropy after we make a split based on the currently selected feature value. For example, let's say we split on a leaf based on some feature 'age'. Data with age < 10 goes towards left and others on right.
# 
# Now for each new leaf, we calculate entropy and further calculate information gain using its parent node. Since entropy means the degree of chaos and decrease in entropy is means increase in information gain after the split so we take that feature and its value as the best split point.
# $$ Entropy(S_O) = -\sum_{i=1}^N {p_i log_2(p_i)}$$
# 
# $$ IG = S_O - \sum_{i=1}^2 {N_i \over N}S_i $$
# 
# <img src="https://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/img/topic3_credit_scoring_entropy.png"></img>
# 
# Check out [this video](https://www.youtube.com/watch?v=7VeUPuFGJHk) to understand decision trees better.

# In[22]:


model = DecisionTreeClassifier(max_depth=3, max_features=0.5, random_state=7)
get_cross_val(model, df, y, skf)
print_metric(model, df, y)


# **Let's visualize the decision tree**

# In[23]:


dot_data = StringIO()
export_graphviz(model, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## 6. RandomForest
# 
# RandomForest is just an ensemble of the decision trees. A single decision tree is not enough to predict a value so RandomForest creates an ensemble of decision trees randomly. The random keyword specifies certain parameters, which adds randomness either during initialization or during training. One of them is `max_features`, which chooses a sample of features randomly with each decision tree to reduce overfitting and introduce stable and robust training. RandomForest by default doesn't choose all the data points with each iteration of training but uses a subsample of it. Also, it has `oob_score` which calculates the metric score on the data left out during subsampling.
# 
# <img src="https://cdn-images-1.medium.com/max/592/1*i0o8mjFfCn-uD79-F1Cqkw.png" width="500"></img>

# In[24]:


model = RandomForestClassifier(class_weight='balanced', max_depth=12, max_features=0.7,
                               n_estimators=50, n_jobs=-1, random_state=7)
get_cross_val(model, df, y, skf)
print_metric(model, df, y)


# ## 7. Adaptive Boosting
# 
# Adaptive boosting algorithms take a greedy approach. It creates an ensemble of small weak models called priors and uses them to make final predictions. The steps are as follows:
# 1. First, it builds a simple classifier usually a decision tree and creates initial predictions.
# 2. Using these predictions it re-weigh the input data in which the incorrectly predicted instances are assigned a higher weight and correctly predicted instances are assigned lower weights.
# 3. Then it forms a new data frame in which the instances are selected based on the assigned weights.
# 4. Again repeat step 1 with this new data.
# 5. Finally, after all the iterations, final predictions are made using the linear combinations of these stumps.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/0*paPv7vXuq4eBHZY7.png" width="600"></img>
# 
# Check out [this video](https://www.youtube.com/watch?v=LsK-xG1cLYA) for step-wise explanation.

# In[25]:


model = AdaBoostClassifier(random_state=7)
get_cross_val(model, df, y, skf)
print_metric(model, df, y)


# ## 8. Gradient Boosting
# 
# This algorithm is somewhat similar to an adaptive boosting algorithm but not exactly. It uses a gradient-based method to make final predictions and create new trees. Also, the trees here are usually larger than the stumps in AdaBoost. The steps are as follows:
# 1. Make an initial prediction which is basically a leaf.
# 2. Create residuals using the actual value and the initial prediction.
# 3. Build a tree to predict these residuals.
# 4. Take the average of the residuals corresponding to the data point and make it as the new predictions.
# 5. Repeat steps 2, 3 and 4 until the number of trees is formed as assigned in the algorithm instance.
# 6. At last take ensemble of predictions of each tree by multiplying each prediction by a learning rate.
# 
# For better accuracy, it is preferred to use a large number of trees and lower a learning-rate but this might end up slow convergence and take a lot of time.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*2fGb3jTF85XyHtnpJYA8ug.png" width="500" height="400"></img>
# 
# Check out [this video](https://www.youtube.com/watch?v=3CC4N4z3GJc) for more.

# ### XGBoost
# [XGBoost](https://xgboost.readthedocs.io/en/latest/) comes with the implementation of gradient-boosting trees with some other relevant features. XGBoost is faster than Scikit-Learn default GradientBoosting API. It uses level-wise growth while constructing a tree.

# In[26]:


model = XGBClassifier(n_jobs=-1, random_state=7)
get_cross_val(model, df, y, skf)
print_metric(model, df, y)


# ### LightGBM
# [LightGBM](https://lightgbm.readthedocs.io/en/latest) is another implementation of gradient boosting algorithm. It grows tree leaf-wise i.e. it grows vertically as compared to other algorithms which grow horizontally(level-wise growth). As compared to XGBoost it is much faster and has a lot more parameters to play with. It also handles categorical features in a different way. To make it aware of categorical columns you need to either pass them or change the data-type of those columns to `category`.

# In[27]:


X = df.copy()
for c in cat_cols:
    X[c] = X[c].astype('category')

model = LGBMClassifier(bagging_fraction=0.5, feature_fraction=0.5, num_leaves=20, n_jobs=-1, random_state=7)
get_cross_val(model, X, y, skf)
print_metric(model, X, y)


# ### CatBoost
# [Catboost](https://tech.yandex.com/catboost/) is created by [Yandex](https://yandex.com) which is basically LightGBM plus it handles categorically features more efficiently. It uses some special algorithm to handle categorical features which is similar to mean-encoding but follows a bit different approach to avoid overfitting. To work with categorical features with Catboost you need to pass the indexes of categorical columns without performing any one-hot encoding.

# In[28]:


cat_ixs = [df.columns.tolist().index(c) for c in cat_cols]

model = CatBoostClassifier(random_state=7, cat_features=cat_ixs, silent=True)
get_cross_val(model, df, y, skf)
print_metric(model, df, y)


# ## 9. Dimensionality reduction
# 
# What if we have a lot of data and not all of them is relevant or maybe we can extract some features out of it and reduce the dimensionality by maintaining the efficiency? It's possible through statistics. There are several of them. Some are statistical such as PCA, LDA, and TSNE while some are based on neural-nets such as AutoEncoders. I will explain PCA and TSNE here.

# ### Principle Component Analysis
# 
# Consider that each data point is represented in an N-dimensional space. Each dimension is a principle component here. Now in this N-dimensional space, PCA tries to find a direction which has the highest variance and takes the hyperplane perpendicular to it. Each data point is mapped on this hyperplane and the new points are the new coordinates of these data points. The number of components is chosen based on the variance. Check out this amazing [video](https://www.youtube.com/watch?v=FgakZw6K1QQ) for step-wise explanation. 
# 
# Theoritically the PCA components are calculated as follows:
# 1. Normalize the data.
# 2. Shift the data points such that the mean of the data becomes 0.
# 3. Get the first principle component which is basically a line on which the perpendicular projection distance of each data point is minimum.
# 4. Draw more principle components such the each new component is perpendicular to the previous one. We can have maximum n components, where n is total number of features.
# 5. Draw the variance graph(skree plot) and choose the optimal number of principle components.
# 6. Take the choosen principle components and project the data points on each of them.
# 7. Rotate the components to make it p-dimensional, where p is the number of choosen components.
# 8. Finally, project each data point in p-dimensional space using the projections made step-6 from each component.
# 
# Mathematically the PCA components are calculated as follows:
# 1. Normalize the data
# 2. Calculate the covariance matrix
# 3. Calculate eigenvalues and eigenvectors
# 4. Eigenvector matrix will be of shape $n*m$. Take the first n_columns of this matrix. These will be the reduced components for each data-point.
# 
# <img src="https://blog.bioturing.com/wp-content/uploads/2018/11/Blog_pca_6b.png" width="300"></img>

# Now to choose the components we can check the variance by visualizing the Components vs Cumulative-Variance plot and choose the number of components which covers almost 99% of the total variance.

# In[29]:


X = StandardScaler().fit_transform(df)
pca = PCA(n_components=24).fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');


# In[30]:


X_r = PCA(n_components=18).fit_transform(X)
model = LogisticRegression(C=0.02637, class_weight='balanced', n_jobs=-1, random_state=7)
get_cross_val(model, X_r, y, skf)
print_metric(model, X_r, y)


# ### t-Distributed Stochastic Neighbor Embedding
# 
# PCA is good but it's linear so it can't interpret complex polynomial relations among independent variables. 
# TSNE is a non-linear technique for dimensionality reduction which maps multi-dimensional data to two or more dimensions suitable for human observation. TNSE is well-suited for the visualization of high-dimensional data. TSNE is a probabilistic technique. Workflow defined in the [original paper](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) is as follows:
# 1. First, it calculates the conditional probability of similarity of points in high-dimensional space and points in low-dimensional space. Conditional probability describes the similarity between two points that the first point will choose the second point in proportion to their probability density under a Gaussian centered at first point.
# 2. Then it tries to reduce the difference in the conditional probabilities in high-dimensional space and low-dimensional space.
# 3. Finally uses gradient descent method t-SNE minimizes the sum of Kullback-Leibler divergence of these data points.
# 
# Check out [this video](https://www.youtube.com/watch?v=NEaUSP4YerM) for more.

# In[31]:


X = StandardScaler().fit_transform(df)
X_r = TSNE(n_components=2).fit_transform(X)
model = LogisticRegression(C=0.0278, class_weight='balanced', n_jobs=-1, random_state=7)
get_cross_val(model, X_r, y, skf)
print_metric(model, X_r, y)


# ## 10. K-Means Clustering
# 
# K-Means clustering is an unsupervised method which uses only the independent variables to cluster them in N-dimensional space. Usually, number of clusters is unknown because it's unsupervised learning algorithm but in this case, we know the number of clusters so we can just add that to the model initialization. In case when we have unlabeled data, we usually use elbow-method to determine the number of clusters. KNN chooses random initial points or initial centroids so the algorithm doesn't guarantee an optimal result. Although it uses `k-means++` to initialize the points still it is random. The steps for this algorithm are as follows:
# 1. Initialize K-points.
# 2. For each data, point finds the closest centroid and assign the class of the centroid to it.
# 3. Find new centroids among all the clusters formed by averaging the independent variables.
# 4. Repeat steps 2-3 until new centroids are the same as the previous ones.
# 5. For each new data point in testing assign the label of the centroid it is closest to.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*tWaaZX75oumVwBMcKN-eHA.png" width="600"></img>
# 
# Check out [this video](https://www.youtube.com/watch?v=4b5d3muPQmA) for more.

# In[32]:


model = KMeans(n_clusters=2, n_jobs=-1, random_state=7)
print_metric(model, df, y)


# # End
