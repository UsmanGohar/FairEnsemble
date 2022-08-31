#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report, f1_score, plot_roc_curve


# In[2]:


adult = pd.read_csv('../input/adult-census-income/adult.csv')
adult.sample()


# In[3]:


adult.info()


# *In this info detail, indicate that there is no missing value at all. But if you see the whole data carefully, you will find **missing value with '?'**.*

# # PreProcessing

# *Preprocessing scheme:*
# * Encode all columns
# * Drop education because it's already encoded on education.num
# * Drop fnlwgt because it's unique

# *Handling Missing Value In Pipeline*

# In[4]:


binary_encoder_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'NC', missing_values = '?')),
    ('binary', ce.BinaryEncoder())
])

transformer = ColumnTransformer([
    ('one hot', OneHotEncoder(drop = 'first'), ['relationship', 'race', 'sex']),
    ('binary', binary_encoder_pipe, ['workclass', 'marital.status', 'occupation', 'native.country'])],
    remainder = 'passthrough')


# *Splitting Data*

# In[5]:


adult['income'].value_counts()


# Income is the target data and **indicated with imbalance data**. I define **income with 1 if income is >50K and 0 if income is <50K**.

# In[6]:


X = adult.drop(['fnlwgt', 'education', 'income'], axis = 1)
y = np.where(adult['income'] == '>50K', 1, 0)


# In[7]:


X.shape


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,
                                                    test_size = 0.3, random_state = 1212)


# I use 0.3 as default score for test_size and X.shape for random_state so the data will be devided equally.

# # Define Model

# I use 3 Boosting Algorithms Models:
# * Ada Boost Classifier
# * Gradient Boosting Classifier
# * XGB Classifier

# In[9]:


adaboost = AdaBoostClassifier(DecisionTreeClassifier(), random_state = 1212)
pipe_ada = Pipeline([
    ('transformer', transformer),
    ('adaboost', adaboost)])

gradboost = GradientBoostingClassifier(random_state = 1212)
pipe_grad = Pipeline([
    ('transformer', transformer),
    ('gradboost', gradboost)])

XGBOOST = XGBClassifier(random_state = 1212)
pipe_XGB = Pipeline([
    ('transformer', transformer),
    ('XGBOOST', XGBOOST)])


# # Cross Validation

# *Model Evaluation*

# In[10]:


def model_evaluation(model, metric):
    skfold = StratifiedKFold(n_splits = 5)
    model_cv = cross_val_score(model, X_train, y_train, cv = skfold, scoring = metric)
    return model_cv

pipe_ada_cv = model_evaluation(pipe_ada, 'f1')
pipe_grad_cv = model_evaluation(pipe_grad, 'f1')
pipe_XGB_cv = model_evaluation(pipe_XGB, 'f1')


# *Fitting Data*

# In[11]:


for model in [pipe_ada, pipe_grad, pipe_XGB]:
    model.fit(X_train, y_train)


# *Summary*

# In[12]:


score_mean = [pipe_ada_cv.mean(), pipe_grad_cv.mean(), pipe_XGB_cv.mean()]
score_std = [pipe_ada_cv.std(), pipe_grad_cv.std(), pipe_XGB_cv.std()]
score_f1 = [f1_score(y_test, pipe_ada.predict(X_test)),
            f1_score(y_test, pipe_grad.predict(X_test)), 
            f1_score(y_test, pipe_XGB.predict(X_test))]
method_name = ['Ada Boost Classifier', 'Gradient Boost Classifier ',
              'XGB Classifier']
summary = pd.DataFrame({'method': method_name, 'mean score': score_mean,
                        'std score': score_std, 'f1 score': score_f1})
summary


# From these scores, **XGB Classifier is the best one** with the highest f1 score and mean score, also the lowest std score. Let's cross-check with the important features, see if the model is correct.

# In[13]:


plot_roc_curve(pipe_XGB, X_test, y_test)


# # Importance Features

# In[14]:


features = list(pipe_ada[0].transformers_[0][1].get_feature_names()) + pipe_ada[0].transformers_[1][1][1].get_feature_names() + ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']


# In[15]:


imptab_ada = pd.DataFrame(pipe_ada[1].feature_importances_, columns = ['imp'], index = features)
imptab_ada.sort_values('imp').plot(kind = 'barh', figsize = (15,8))
plt.title('Importance Table For Ada Boost Classifier Model')
plt.show()


# In[16]:


imptab_grad = pd.DataFrame(pipe_grad[1].feature_importances_, columns = ['imp'], index = features)
imptab_grad.sort_values('imp').plot(kind = 'barh', figsize = (15,8))
plt.title('Importance Table For Gradient Boost Classifier Model')
plt.show()


# In[17]:


imptab_XGB = pd.DataFrame(pipe_XGB[1].feature_importances_, columns = ['imp'], index = features)
imptab_XGB.sort_values('imp').plot(kind = 'barh', figsize = (15,8))
plt.title('Importance Table For XGB Classifier Model')
plt.show()


# From Importance Features Table, the **XGB Classifier can boost almost all the features**. It's has a consistency with the cross validation result. Now, see if the HyperParameter Tuning process can boost until getting the maximum score.

# # HyperParameter Tuning

# In[18]:


XGBOOST = XGBClassifier(random_state = 1212)
estimator = Pipeline([('transformer', transformer), ('XGBOOST', XGBOOST)])

hyperparam_space = {
    'XGBOOST__learning_rate': [0.1, 0.05, 0.01, 0.005],
    'XGBOOST__n_estimators': [50, 100, 150, 200],
    'XGBOOST__max_depth': [3, 5, 7, 9]
}

random = RandomizedSearchCV(
                estimator,
                param_distributions = hyperparam_space,
                cv = StratifiedKFold(n_splits = 5),
                scoring = 'f1',
                n_iter = 10,
                n_jobs = -1)

random.fit(X_train, y_train)


# In[19]:


print('best score', random.best_score_)
print('best param', random.best_params_)


# After HyperParameter Tuning, the best score is 0.6996, which getting lower. N estimator is 150, Max depth is 5, and Learning rate is 0.1. Let's compare the result.

# # Before VS After Tuning Comparison

# In[20]:


estimator.fit(X_train, y_train)
y_pred_estimator = estimator.predict(X_test)
print(classification_report(y_test, y_pred_estimator))


# In[21]:


random.best_estimator_.fit(X_train, y_train)
y_pred_random = random.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_random))


# In[22]:


score_list = [f1_score(y_test, y_pred_estimator), f1_score(y_test, y_pred_random)]
method_name = ['XGB Classifier Before Tuning', 'XGB Classifier After Tuning']
best_summary = pd.DataFrame({
    'method': method_name,
    'f1 score': score_list
})
best_summary


# After all, HyperParameter Tuning doesn't work good in this data. So if I have to choose, I pick the **XGB Classifier score Before Tuning, which is 0.71**. I know the number isn't good enough either because the data is imbalance and I don't process any resampling on it.
