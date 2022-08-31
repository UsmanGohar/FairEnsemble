#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)

import missingno as msno
from sklearn.preprocessing import LabelEncoder

#ignore warnings 
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inlinethe current session')


# In[ ]:


df = pd.read_csv('../input/bank-marketing/bank-additional-full.csv', sep=';')
df.head()


# In[ ]:


df.columns


# In[ ]:


for col in df.columns : 
    msg = 'columnn : {:>10}\t count of NaN value : {:.0f}'.format(col, 100 * (df[col].isnull().sum() ))
    print(msg)


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


int_column = df.dtypes[df.dtypes =='int64'].index |  df.dtypes[df.dtypes =='float64'].index


# In[ ]:


for col in int_column : 
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    sns.distplot(df[col])
    plt.xlabel(col)
    plt.ylabel('Density')
    
    plt.subplot(1,2,2)
    sns.boxplot(x='y', y = col, data =df, showmeans = True)
    plt.xlabel('Target')
    plt.ylabel(col)
    
    plt.show()


# In[ ]:


obj_column = df.dtypes[df.dtypes == 'object'].index
df[obj_column[2]].unique()


# In[ ]:


for i in range(0, len(obj_column)) :
    print(obj_column[i])
    print(df[obj_column[i]].unique())
    print()


# In[ ]:


for i in range(0, len(obj_column)) :
    fig, ax = plt.subplots(figsize=(15,4))

    sns.countplot(x = obj_column[i], data = df)
    sns.set(font_scale=1)

    ax.set_title('{} Count Distribution'.format(obj_column[i]))


# In[ ]:


labelencoder_X = LabelEncoder() 


# In[ ]:


df["job"] = labelencoder_X.fit_transform(df["job"])
df["marital"] = labelencoder_X.fit_transform(df["marital"])
df["education"] = labelencoder_X.fit_transform(df["education"])
df["default"] = labelencoder_X.fit_transform(df["default"])
df["housing"] = labelencoder_X.fit_transform(df["housing"])
df["loan"] = labelencoder_X.fit_transform(df["loan"])
df["contact"] = labelencoder_X.fit_transform(df["contact"])
df["month"] = labelencoder_X.fit_transform(df["month"])
df["day_of_week"] = labelencoder_X.fit_transform(df["day_of_week"])
df["poutcome"] = labelencoder_X.fit_transform(df["poutcome"])
df["y"] = labelencoder_X.fit_transform(df["y"])


# In[ ]:


pd.set_option('max_columns', None)
df.tail()


# In[ ]:


df.shape


# In[ ]:


heatmap_data = df

colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y = 1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True, annot_kws={'size':16})

del heatmap_data


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('y',axis=1),
                                                    df['y'],
                                                    test_size=.3, random_state = 42,
                                                    stratify= df['y'])


# In[ ]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=df.drop('y',axis=1).columns)
X_test = pd.DataFrame(X_test, columns=df.drop('y',axis=1).columns)


# In[ ]:


models = [LogisticRegression(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          XGBClassifier()]

names = [ 'LogisticRegression',
         'DecisionTreeClassifier',
          'RandomForestClassifier',
          'XGBClassifier']

for model,name in zip(models,names):
    m = model.fit(X_train,y_train)
    print(name, 'report:')
    print('Train score',model.score(X_train,y_train))
    print('Test score',model.score(X_test,y_test))
    print()
    print("Train confusion matrix:\n",confusion_matrix(y_train, model.predict(X_train)),'\n')
    print("Test confusion matrix:\n",confusion_matrix(y_test, model.predict(X_test)))
    print('*'*50)


# In[ ]:


model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

from sklearn.tree import plot_tree
plt.figure(figsize=(20,15))
plot_tree(model,
          feature_names= df.drop('y', axis=1).columns,  
          class_names= ['yes','no'],
          filled=True)
plt.show()


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score


# In[ ]:


m = RandomForestClassifier().fit(X_train,y_train)
pred_y = m.predict(X_test)
print('*'*50)
print('Report')
print('model : RandomForestClassifier')
print('Train score',model.score(X_train,y_train))
print('Test score',model.score(X_test,y_test))
print()
print("accuracy: %.2f" %accuracy_score(y_test, pred_y))
print("Precision : %.3f" % precision_score(y_test, pred_y))
print("Recall : %.3f" % recall_score(y_test, pred_y))
print("F1 : %.3f" % f1_score(y_test, pred_y))
print()
print("Train confusion matrix:\n",confusion_matrix(y_train, model.predict(X_train)),'\n')
print("Test confusion matrix:\n",confusion_matrix(y_test, model.predict(X_test)))
print('*'*50)


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

print('Train score',model.score(X_train,y_train))
print('Test score',model.score(X_test,y_test))

lrCoef = LogisticRegression().fit(X_train,y_train).coef_
print(lrCoef)


# In[ ]:


print("Coefficient of Logistic Regression")
for i in range(0, len(lrCoef[0])) :
    print('{} : {}'.format(X_train.columns[i], lrCoef[0][i]))


# In[ ]:


coefdf = pd.DataFrame(data=X_train.columns, index=range(0, len(lrCoef[0])), columns=['Feature'])
coefdf['Coef'] = lrCoef[0]
coefdf['Absuolute num of Coef'] = abs(lrCoef[0])
coefdf = coefdf.sort_values(by='Absuolute num of Coef', ascending=False).reset_index(drop=True)
coefdf


# In[ ]:


bcd = {'age', 'job', 'marital', 'education', 'default', 'housing', 'loan'}
lc = {'contact', 'month', 'day_of_week', 'duration'}
oth = {'campaign', 'pdays', 'previous', 'poutcome'}
sec = {'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'}

coefdf['Category'] = 0
for i in range(0,coefdf.shape[0]) : 
    if coefdf['Feature'][i] in bcd : 
        coefdf['Category'][i] = 'Bank Client'
    elif coefdf['Feature'][i] in lc : 
        coefdf['Category'][i] = 'Last Contact'
    elif coefdf['Feature'][i] in oth : 
        coefdf['Category'][i] = 'Other'
    else : 
        coefdf['Category'][i] = 'Social Economic'    
coefdf.sort_values(by='Absuolute num of Coef', ascending=False)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=coefdf, y=coefdf['Feature'], x=coefdf['Coef'])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.title('Coefficient of Logistic Regression\n(score : 91%)', fontsize=20)
plt.xlabel('Coefficient')

plt.savefig('Coefficient of Logistic Regression.png')
plt.show()


# In[ ]:


import shap


# In[ ]:


RFmodel = RandomForestClassifier().fit(X_train, y_train)

explainer = shap.TreeExplainer(RFmodel)
shap_values = explainer.shap_values(X_test)

# fig, ax = plt.subplots(figsize=(10,5))
shap.summary_plot(shap_values, X_test, plot_size=(10,5), show=False)
plt.title('SHAP of Random Forest Classifier\n(score : 92%)')
plt.show()


# In[ ]:


# from sklearn.inspection import permutation_importance

# result = permutation_importance(RFmodel, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
# sorted_idx = result.importances_mean.argsort()

# plt.figure(figsize=(10,5))
# plt.title('Permutation Importance of Random Forest Classifier\n(score : 92%)')

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.boxplot(result.importances[sorted_idx].T,
#             vert=False, labels=X.columns[sorted_idx]);


# In[ ]:


XGBmodel = XGBClassifier().fit(X_train, y_train)

explainer = shap.TreeExplainer(XGBmodel)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_size=(10,5), show=False)
plt.title('SHAP of XGBoost Classifier\n(score : 88%)')
plt.figsize=(10,5)
plt.show()


# **Duration** is the most important feature which should be discarded because it is not known before a call is performed. Still, it indicates that making a call longer could help a lot to increase the subscription. 
# 
# The higher cons.price.idx and euribor3m, the lower emp.var.rate and nr.employed, which are all social and economic context attributes, the more likely to subscribe the term deposit.
# The social and economic context attributes were the most helpful features to predict. 
# 
# Among Bank Clients Data, default was a significant factor but not that much as Social Economic was. Other Bank Clients such as Job, Martial, Education were not an influential factor.   
# 
# When it comes to attributes which are related with the last contact of the current campaign, coefficient of Contact and Month were higher than any other Bank Clients Data attributes.
# 
# In conclusion, subscription mostly depends on social and economic situation. Thus, to make more cost efficient to increase subcribition to a term deposit, it is needed to concentrate marketing budget on certain time when euribor rate, comsumer price rate are high, employment variation rate is low. 
