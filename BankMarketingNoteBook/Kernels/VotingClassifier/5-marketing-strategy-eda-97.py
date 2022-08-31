#!/usr/bin/env python
# coding: utf-8

# The [data](https://www.kaggle.com/henriqueyamahata/bank-marketing) is related with direct marketing campaigns of a Portuguese banking institution. 
#    The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
#    in order to access if the product (bank term deposit) would be (or not) subscribed. 
# 
# > Number of Instances: 45211
# 
# > Number of Attributes: 16 + output attribute.
# 
# 
# The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 

# # Attribute information:
#    Input variables:
#   
#    ## bank client data:
#    
#    1 - age (numeric)
#   
#    2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
#                                        "blue-collar","self-employed","retired","technician","services") 
#    
#    3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
#   
#    4 - education (categorical: "unknown","secondary","primary","tertiary")
#    
#    5 - default: has credit in default? (binary: "yes","no")
#   
#    6 - balance: average yearly balance, in euros (numeric) 
#   
#    7 - housing: has housing loan? (binary: "yes","no")
#   
#    8 - loan: has personal loan? (binary: "yes","no")
#   
#    ## related with the last contact of the current campaign:
#    
#    9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
#   
#   10 - day: last contact day of the month (numeric)
#   
#   11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#   
#   12 - duration: last contact duration, in seconds (numeric)
#   
#    ## other attributes:
#   
#   13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#   
#   14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
#   
#   15 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
#   16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
# 
# # Output variable (desired target):
# 
#   17 - y - has the client subscribed a term deposit? (binary: "yes","no")

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


path = '../input/bank-marketing/bank-additional-full.csv'
df  = pd.read_csv(path,sep=';')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.y.value_counts()


# #EDA 

# In[ ]:


int_column = df.dtypes[df.dtypes == 'int64'].index | df.dtypes[df.dtypes == 'float64'].index


# In[ ]:



for column in int_column:
    plt.figure(figsize=(16,4))

    plt.subplot(1,3,1)
    sns.distplot(df[column])
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'{column}  Distribution')

    plt.subplot(1,3,2)
    sns.boxplot(x='y', y=column, data =df, showmeans=True )
    plt.xlabel('Target')
    plt.ylabel(column)
    plt.title(f'{column}  Distribution')

    plt.subplot(1,3,3)
    counts, bins = np.histogram(df[column], bins=20, normed=True)
    cdf = np.cumsum (counts)
    plt.plot (bins[1:], cdf/cdf[-1])
    #plt.xticks(range(15,100,5))
    plt.yticks(np.arange(0,1.1,.1))
    plt.title(f'{column}  cdf')
    plt.show()
    print()


# In[ ]:


# Quantiles
for column in int_column:
    print(f'For {column}:')

    print('Min:', df[column].quantile(q = 0))
    print('1º Quartile:', df[column].quantile(q = 0.25))
    print('2º Quartile:', df[column].quantile(q = 0.50))
    print('3º Quartile:', df[column].quantile(q = 0.75))
    print('Max:', df[column].quantile(q = 1.00),'\n')


# In[ ]:


df.drop(df[df.age>60].index, inplace=True)
df.drop(df[df.campaign>10].index, inplace=True)
df.drop(df[df.duration>1000].index, inplace=True)
df.drop('pdays', axis=1, inplace=True)


# ##For object type

# In[ ]:


dfgrouped = df.groupby('y')


# In[ ]:


def plot_barh(array,incrementer, bias, text_color ='blue', palette_style = 'darkgrid',palette_color = 'RdBu'):

    sns.set_style(palette_style)
    sns.set_palette(palette_color)

    plt.barh(array.index, width = array.values, height = .5)
    plt.yticks(np.arange(len(array)))
    plt.xticks( range(0, round(max(array)) +bias, incrementer ))

    for index, value in enumerate(array.values):
        plt.text(value +.5, index, s= '{:.1f}%'.format(value), color = text_color)

    #plt.show()
    return plt


# In[ ]:


def feature_perc(feature,groupby= 'yes'):

    count = dfgrouped.get_group(groupby)[feature].value_counts()
    total_count = df[feature].value_counts()[count.index]

    perc = (count/total_count)*100
    return perc 


# In[ ]:


obj_column = df.dtypes[df.dtypes == 'object'].index
obj_column


# In[ ]:


for column in obj_column[:-1]:

    yes_perc = feature_perc(column, groupby='yes')
    no_perc = feature_perc(column, groupby='no')

    plt.figure(figsize=(16,6))

    plt.subplot(1,2,1)
    plt.title(f'Success rate by  {column}')
    plot_barh(yes_perc.sort_values(),5,10)

    plt.subplot(1,2,2)
    plt.title(f'Failure rate by  {column}')
    plot_barh(no_perc.sort_values(),5,10)
    plt.show()
    print()


# ##Modeling

# In[ ]:


df1 = df.copy()
df1['y'] = df1.y.apply(lambda x:0 if x=='no' else 1)


# In[ ]:


df1.y.value_counts()


# In[ ]:


from sklearn.utils import resample

# Separate majority and minority classes
df1_majority = df1[df1.y==0]
df1_minority = df1[df1.y==1]
 
# Upsample minority class
df1_minority_upsampled = resample(df1_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=36962,    # to match majority class
                                 random_state=42) # reproducible results
 
# Combine majority class with upsampled minority class
df = pd.concat([df1_majority, df1_minority_upsampled])
 
# Display new class counts
df.y.value_counts()


# In[ ]:


df


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


obj_column = df.dtypes[df.dtypes == 'object'].index
mapingdf = pd.DataFrame()

for column in obj_column:
    labelencoder = LabelEncoder()
    df[column] = labelencoder.fit_transform(df[column])
    mapingdf[column] = df[column]
    mapingdf['_'+column] =  labelencoder.inverse_transform(df[column])


# In[ ]:


#for reference
mapingdf


# In[ ]:


df.head()


# In[ ]:


df.corr().y.sort_values()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('y',axis=1),
                                                    df['y'],
                                                    test_size=.3, random_state = 42,
                                                    stratify= df['y'])


# In[ ]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


models = [DecisionTreeClassifier(),
          RandomForestClassifier(),
          XGBClassifier()]

names = [ 'DecisionTreeClassifier',
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
          feature_names= df1.drop('y', axis=1).columns,  
          class_names= ['yes','no'],
          filled=True)
plt.show()

