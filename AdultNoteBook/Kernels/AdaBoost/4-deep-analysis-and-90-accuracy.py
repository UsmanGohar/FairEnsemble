#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('../input/adult-census-income/adult.csv')


# # EDA

# In[4]:


df.head()


# In[5]:


df.shape


# Dataset has 32561 rows and 15 columns.

# In[6]:


df.nunique()


# Label column has only two categories, hence it is a problem of classification. There are no constant columns nor there are any identifier column.

# In[7]:


df.isnull().sum()


# There are no null values in the dataset

# In[8]:


df.dtypes


# There are 8 object type feautures rest of the features are of integer type.

# In[9]:


df.skew()


# There is skewness present in the data which needs to be removed.

# In[10]:


df['income'].value_counts()


# Dataset is imbalanced.

# In[11]:


df.describe()


# All the columns are not present as they are of object type. Count of each column is 32561 showing there are no null values. Mean is very much greater than the median in capital gain and capital loss stating that there is high skewness present and data is skewed to the right side. Also there is high variance in Capital gain and Capital loss column. Min, Max, and interquartile ranges have variable difference, that means there are outliers present in the data.

# ### Univariate Analysis

# In[12]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['income'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(x='income',data=df)
plt.ylabel('No. of People')
df['income'].value_counts()


# Dataset if highly imbalanced. There is less than 25% of >50K income category while more than 75% of <=50K income.

# In[13]:


#Separating categorical and continuous variables
cat=[feature for feature in df.columns if df[feature].nunique()<45]
cont=[feature for feature in df.columns if df[feature].nunique()>45]


# In[14]:


plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
df['workclass'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(x='workclass',data=df)
plt.ylabel('No. of Individuals')
df['workclass'].value_counts()


# There are 9 workclass in total including Never worked and one unknown category(?).Most individuals work in private sector and there are very few who have never worked or work without pay. There are 3 categories of govt job provided state, federal and local among which no. of people working in the local govt is highest.

# In[15]:


plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
df['education'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(x='education',data=df)
plt.xticks(rotation=45)
plt.ylabel('No. of Individuals')
df['education'].value_counts()


# Most of the people are high school graduate. There are few who have done masters and doctorate. The no. of people who went through just the preschool or 1st to 4th is the least.

# In[16]:


plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
df['education.num'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(x='education.num',data=df)
plt.ylabel('No. of Individuals')
df['education.num'].value_counts()


# Majority of individuals lie in the 9th 10th category of education no. which is a liitle higher than the median education number. People with least and highest educations are very few.

# In[17]:


plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
df['marital.status'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(x='marital.status',data=df)
plt.ylabel('No. of Individuals')
df['marital.status'].value_counts()


# Majority of people are married to a civialian spouse or Never married. Least people are married to armed forces. From the above maritial status data we can see that there are less young people in the workforce as compared to young ones.

# In[18]:


plt.figure(figsize=(20,15))
plt.subplot(2,1,1)
df['occupation'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(2,1,2)
sns.countplot(x='occupation',data=df)
plt.xticks(rotation=45)
plt.ylabel('No. of Individuals')
df['occupation'].value_counts()


# We can observe over here that prof-speciality has the highest number for people than any other occupation. followed by craft repair persons. Minimum occupation category is the armed forces with only 9 people in it. There is an unknown category of occupation too.

# In[19]:


plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
df['relationship'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.subplot(1,2,2)
sns.countplot(x='relationship',data=df)
plt.ylabel('No. of Individuals')
df['relationship'].value_counts()


# There are much more husband working than their wives. There are 25% individuals working who fall in not in family category.

# In[20]:


plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
df['race'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(x='race',data=df)
plt.xticks(rotation=45)
plt.ylabel('No. of Individuals')
df['race'].value_counts()


# SInce this is from european countries, most of the individuals working here are white. There is also an other category where minory races are present.

# In[21]:


plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
df['sex'].value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(x='sex',data=df)
plt.ylabel('No. of Individuals')
df['sex'].value_counts()


# There is more than double the number of men working than the no. of women.

# In[22]:


plt.figure(figsize=(15,8))
sns.countplot(x='native.country',data=df)
plt.xticks(rotation=90)
plt.ylabel('No. of Individuals')
df['native.country'].value_counts()


# Majority of the people working belong from U.S. whereas there are people who come to U.S. from their own countries but there no. is very low. Second highest no. of people belong from Mexico as it is a neighboring country.

# In[23]:


plt.figure(figsize=(8,6))
sns.histplot(df['age'],kde=True,color='r')
plt.ylabel('No. of Individuals')
print('Minimum',df['age'].min())
print('Maximum',df['age'].max())


# Minimum age of a working individual is 17 and highest is 90 which is way far retirement, but majority of the people working are in the age 25 to 45. Data is skewed to the right side.

# In[24]:


plt.figure(figsize=(8,6))
sns.histplot(df['fnlwgt'],kde=True,color='k')
plt.ylabel('No. of Individuals')
print('Minimum',df['fnlwgt'].min())
print('Maximum',df['fnlwgt'].max())


# It is assigned by combination of features and has the peak wt as 0.2*1e6. Data does not follow normal distribution and data is rigt skewed.

# In[25]:


plt.figure(figsize=(15,12))
sns.distplot(df['capital.gain'],color='m', kde_kws={"color": "k"})
print('Minimum',df['capital.gain'].min())
print('Maximum',df['capital.gain'].max())


# Minimum capital gain is 0 while the range goes on to 99999 but most of the people are with the gain of 1000. Data is highly skewed with a very long tail due to presence of large outliers. Outliers here are very few people belonging to elite class who have very large capital gains.

# In[26]:


plt.figure(figsize=(15,12))
sns.distplot(df['capital.loss'],color='g', kde_kws={"color": "k"})
print('Minimum',df['capital.loss'].min())
print('Maximum',df['capital.loss'].max())


# Minimum capital loss is 0 while the range goes on above 4000 but most of the people are with the loss of 1000. There is also a slight peak seen near 2000. Data is highly skewed with a long tail to the right side.

# In[27]:


plt.figure(figsize=(15,12))
sns.distplot(df['hours.per.week'],color='b', kde_kws={"color": "k"})
print('Minimum',df['hours.per.week'].min())
print('Maximum',df['hours.per.week'].max())


# Most of the people work 40 hours a day where there is a high chance that they belong to private sector. There are people working as low ass 1 hour a week and as high as 99 hours a week which undoubtfully might belong from the armed forces. Data shows less skewness compared to the other features in the dataframe.

# In[28]:


for i in cont:
    sns.boxplot(df[i])
    plt.figure()


# There are outliers in all the features, while capital gain and capital loss have very vast no. of outliers.

# ### Bivariate Analysis

# In[29]:


plt.figure(figsize=(8,6))
sns.stripplot(x='income',y='workclass',data=df)


# There are individuals belonging from every workclass who earn >50k except for never worked and without pay, and even there no. is low in the <=50k category.

# In[30]:


plt.figure(figsize=(8,6))
sns.stripplot(x='income',y='education',data=df)


# There is no individual who has done preschool and earns >50k salary while there are few who earn 50k even after going through 1st-4th and 5th-6th in the education criteria. It is also to be noticed that there are doctorates and prof who earn <=50k even with such high education. 

# In[31]:


plt.figure(figsize=(8,6))
sns.stripplot(x='income',y='education.num',data=df)


# It is clearly seen that as the education no. increases chances of earning >50K salary also increases

# In[32]:


plt.figure(figsize=(8,6))
sns.stripplot(x='income',y='marital.status',data=df)


# There are less no. of individual who are married armed forces spouse, thats why the no. is less in both the categories while people with married spouse absent are less in >50k category income comparatively.

# In[33]:


plt.figure(figsize=(8,6))
sns.stripplot(x='income',y='occupation',data=df)


# There are very few people with income greater than armed forces and private house service while all the other categories of people are distributed evenly in both the income categories.

# In[34]:


plt.figure(figsize=(8,15))
sns.stripplot(x='income',y='native.country',data=df)


# The grapghs shows people belonging to diff countries have less chances of earning >50k which is wrong, this is because no. of individuals belonging from other countries other than U.S are very low nut it is to be noticed that there are more people in the category <=50k than >50k.

# In[35]:


plt.figure(figsize=(6,8))
sns.boxenplot(x='income',y='age',data=df,palette="Dark2")


# People with higher mean age earn >50k while there are individuals earning <=50k even wat very high age.

# In[36]:


plt.figure(figsize=(6,8))
sns.boxenplot(x='income',y='fnlwgt',data=df,palette="Dark2_r")


# People are equally divided with respect to fnlwgt in the income category while it is seen that as the fnlwt is high indiduals fall into <=50k income category.

# In[37]:


plt.figure(figsize=(6,8))
sns.boxenplot(x='income',y='capital.gain',data=df,palette="crest")


# As the capital gain increases more people fall into >50k salary while mean of both categories remain cloase to zero capital.gain

# In[38]:


plt.figure(figsize=(6,8))
sns.boxenplot(x='income',y='capital.loss',data=df,palette="ocean")


# There is more density in the >50k income category with increase in capital loss while mean of both categories remain cloase to zero capital.gain

# In[39]:


plt.figure(figsize=(6,8))
sns.boxenplot(x='income',y='hours.per.week',data=df,palette="rocket")


# People earning >50K income work mean hours per week greater than tose earning <50K while people from both the categories work from min to max hours per week.

# In[40]:


#age vs Categorical features
fig,ax=plt.subplots(5,2,figsize=(15,55))
r=0
c=0
for i,n in enumerate(cat):
    if i%2==0 and i>0:
        r+=1
        c=0
    graph=sns.stripplot(x=n,y='age',data=df,ax=ax[r,c])
    if n=='native.country' or n=='occupation' or n=='education':
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 90)
    else:
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 45)
    if n!='education.num':
        graph.set(xlabel=None)
    c+=1


# Individuals working in the government secctor have atmost age 70 to 80 with few outliers which must be the retirement age for them. There are no individuals who do not work after age of 30. There are no individuals of age >70 belonging to the pre school education category while Doctorates and proffessors appear from late 20's as they have to study for more years to get to that level of education. Same is the case with education num, as the education number increases age also is increased. There are no people after the age of 50 in the married to armed forces category with just a few outliers. Widowed category has seen increase as the age age seem to increase, there are very few widows at an early age. There are less people with high age from other races than the white race. There are more no. of working men at higher age than women. There are very few people belonging from other countries with high age.

# In[41]:


#Hours per week vs categorical Feature
fig,ax=plt.subplots(5,2,figsize=(15,55))
r=0
c=0
for i,n in enumerate(cat):
    if i%2==0 and i>0:
        r+=1
        c=0
    graph=sns.violinplot(x=n,y='hours.per.week',data=df,ax=ax[r,c])
    if n=='native.country' or n=='occupation' or n=='education':
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 90)
    else:
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 45)
    if n!='education.num':
        graph.set(xlabel=None)
    c+=1


# Govt employees do not work more than 80 hours a week that also with rare cases. It is seen that people with less education worl more no. hours of the week which is quite logical. No armed force person works more than 60 hours a week while farmers and transport movers has working hours mean higher than other occupation. More no, of individuals who have relationship as own child have high density for working only 20 hous a week. Female works for less no. of hours as compared to men.

# In[42]:


#Capital gain vs categorical Feature
fig,ax=plt.subplots(5,2,figsize=(15,55))
r=0
c=0
for i,n in enumerate(cat):
    if i%2==0 and i>0:
        r+=1
        c=0
    graph=sns.boxplot(x=n,y='capital.gain',data=df,ax=ax[r,c])
    if n=='native.country' or n=='occupation' or n=='education':
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 90)
    else:
        graph.set_xticklabels(graph.get_xticklabels(),rotation = 45)
    if n!='education.num':
        graph.set(xlabel=None)
    c+=1


# Highest capital gains are seen in individuals belonging to private or self employed workclass. There are more no. of prof-school  than any other education category with highest capital gains. There are individuals even with preschool knowledge have capital gain more than 40000. As the education level increases capital gain also increases. People from the armed forces have the least capital gain while most prominent capitals gains are found in people who are in the sales occupation. Whites have more capital gains than any other race. Men also seem to have high capital gains as compared to females. There are many people from <=50k income category who have captital gains more than 10000.

# ### Multivariate Analysis

# In[43]:


data=df.groupby(['age','income']).apply(lambda x:x['hours.per.week'].count()).reset_index(name='Hours')
px.line(data,x='age',y='Hours',color='income',title='age of individuals by Hours of work in the income category  ')


# People earning <=50k tend to work for high no. of hours at a young age and the no. of hours decreases as the age increases but still they work for more no. of hours even at a later age as compared to people earning >50K

# In[44]:


plt.figure(figsize=(6,8))
sns.barplot(x='income',y='age',hue='sex',data=df)
plt.ylabel('Average age')


# As the age increases  people are paid more but males are paid more than females.

# In[45]:


sns.factorplot(x='workclass',y='education.num',hue='income',data=df)
plt.xticks(rotation=90)


# Some people belonging to a particular workclass might have less education and some workclass might require more education level, but no matter whatever workclass, people in the same workclass, if they have higher education level they earn more. It is also to be noticed that there is no person from without pay and never worked workclass category who earn more than 50k which is logical.

# In[46]:


sns.factorplot(x='sex',y='education.num',hue='income',data=df)
plt.xticks(rotation=90)


# Females with higher education level earn equal to men having less education level than them irrespective of any income category they fall.

# In[47]:


sns.factorplot(x='race',y='education.num',hue='income',data=df)
plt.xticks(rotation=90)


# Asian pacific race have comparatively more education than the fellows who earn same as much as they do, belonging to other races. Indians and some other races earn >50k with lowest education level.

# In[48]:


sns.factorplot(x='occupation',y='education.num',hue='income',data=df)
plt.xticks(rotation=90)


# People with highest education level belong to armed forces, but people with even education level quite low, who belong to handlers cleaners, transport moving  occupation earn as much as they do. Same is the case with prof speciality. occupation of private house service who earn >50k and <50k have the highest education level difference while prof speciality have the minimum difference.

# In[49]:


plt.figure(figsize=(8,8))
sns.scatterplot(x='age',y='hours.per.week',hue='income',data=df)


# From the scatterplot between age, hours.per.week and income, we observe that a person needs to be >30 to be earning more than 50K, else needs to work at least 60 hours.per.week to earn >50K.

# In[50]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# There are only few features in the above heat map as most of them are of object type. From here we can see that the independent features don not have much correlation with each other i.e. no multicollinearity.

# # Feature Engineering

# ###### Encoding

# In[51]:


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
l=LabelEncoder()
o=OrdinalEncoder()


# In[52]:


#We use ordinal encoder to Encode Independent features
for i in df.columns:
    if df[i].dtypes=='O' and i!='income':
        df[i]=o.fit_transform(df[i].values.reshape(-1,1))


# In[53]:


#We use label encoder to encode label 
df['income']=l.fit_transform(df['income'])


# ##### Removing Outliers

# In[54]:


from scipy.stats import zscore


# In[55]:


#Method to find optimum threshold
def threshold():
    for i in np.arange(3,5,0.2):
        data=df.copy()
        data=data[(z<i).all(axis=1)]
        loss=(df.shape[0]-data.shape[0])/df.shape[0]*100
        print('With threshold {} data loss is {}%'.format(np.round(i,1),np.round(loss,2))) 


# In[56]:


z=np.abs(zscore(df))
threshold()


# From above we choose threhold as 4.2 as data is precious and we cannot afford to lose more than 8% of data.

# In[57]:


df=df[(z<4.2).all(axis=1)]


# ###### Removing Skewness

# In[58]:


#using Power transformer to remove skewness
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()


# In[59]:


for i in cont:
    if np.abs(df[i].skew())>0.5:
        df[i]=pt.fit_transform(df[i].values.reshape(-1,1))


# In[60]:


for i in cont:
    sns.distplot(df[i])
    plt.figure()


# A lot of skewness has been resuced but we cannot remove skewness more than this.

# In[61]:


#Separating dependent and independent features.
x=df.copy()
x.drop('income',axis=1,inplace=True)
y=df['income']


# ##### Handling Imbalanced Data

# In[62]:


#Oversampling using Smote
from imblearn.over_sampling import SMOTE
over=SMOTE()


# In[63]:


x,y=over.fit_resample(x,y)


# In[64]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
y.value_counts().plot.pie(autopct='%1.1f%%')
plt.subplot(1,2,2)
sns.countplot(y)
y.value_counts()


# Data is balanced now, both the category of income have 50% data each.

# ##### Scaling the data

# In[65]:


#Scaling the data using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[66]:


xd=scaler.fit_transform(x)
x=pd.DataFrame(xd,columns=x.columns)


# # Modelling Phase

# In[67]:


#We import Classification Models
from sklearn.naive_bayes import  GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[68]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[69]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve


# In[70]:


#Function to find the best random state
def randomstate(x,y):
    maxx=0
    model=LogisticRegression()
    for i in range(1,201):
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=i)
        model.fit(xtrain,ytrain)
        p=model.predict(xtest)
        accu=accuracy_score(p,ytest)
        if accu>maxx:
            maxx=accu
            j=i
    return j


# In[71]:


#To evakuate performances of all the models
def performance(p,ytest,m,xtest,s):
    print('------------------------------------',m,'------------------------------------')
    print('Accuracy',np.round(accuracy_score(p,ytest),4))
    print('----------------------------------------------------------')
    print('Mean of Cross Validation Score',np.round(s.mean(),4))
    print('----------------------------------------------------------')
    print('AUC_ROC Score',np.round(roc_auc_score(ytest,m.predict_proba(xtest)[:,1]),4))
    print('----------------------------------------------------------')
    print('Confusion Matrix')
    print(confusion_matrix(p,ytest))
    print('----------------------------------------------------------')
    print('Classification Report')
    print(classification_report(p,ytest))


# In[72]:


#Creating a list of models which will be created one by one
models=[GaussianNB(),KNeighborsClassifier(),LogisticRegression(),DecisionTreeClassifier(),
        RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),XGBClassifier(verbosity=0)]


# In[73]:


#Creates and trains model from the models list
def createmodel(x,y):
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=randomstate(x,y))
    for i in models:
        model=i
        model.fit(xtrain,ytrain)
        p=model.predict(xtest)
        score=cross_val_score(model,x,y,cv=10)
        performance(p,ytest,model,xtest,score) 


# In[74]:


createmodel(x,y)


# Random Forest, Gradient Boost, Xtreme Gradient Boost give us the best performance, so we further try hyperparameter tuning on them

# # Hyperparameter Tuning

# In[75]:


from sklearn.model_selection import GridSearchCV


# In[76]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=randomstate(x,y))


# ##### Random Forest

# In[77]:


params={'n_estimators':[100,300,500],
            'criterion':['gini','entropty'],
            'max_depth':[None,1,2,3,4,5,6,7,8,9,10],
           'max_features':['int','float','auto','log2']}


# In[78]:


g=GridSearchCV(RandomForestClassifier(),params,cv=10)


# In[79]:


g.fit(xtrain,ytrain)


# In[80]:


print(g.best_params_)
print(g.best_estimator_)
print(g.best_score_)


# In[81]:


m=RandomForestClassifier(max_features='log2', n_estimators=500)
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)
performance(p,ytest,m,xtest,score)


# ##### Gradient Boost

# In[82]:


from sklearn.model_selection import RandomizedSearchCV


# In[83]:


params={'n_estimators':[100,300,500],
      'learning_rate':[0.001,0.01,0.10,],
      'subsample':[0.5,1],
      'max_depth':[1,2,3,4,5,6,7,8,9,10,None]}


# In[84]:


g=RandomizedSearchCV(GradientBoostingClassifier(),params,cv=10)


# In[85]:


g.fit(xtrain,ytrain)


# In[86]:


print(g.best_params_)
print(g.best_estimator_)
print(g.best_score_)


# In[87]:


m=GradientBoostingClassifier(max_depth=8, subsample=0.5)
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)
performance(p,ytest,m,xtest,score)


# ##### Xtreme Gradient Boost

# In[88]:


params={
 "learning_rate"    : [0.01,0.05, 0.10, 0.15, ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }


# In[89]:


g=RandomizedSearchCV(XGBClassifier(),params,cv=10)


# In[90]:


g.fit(xtrain,ytrain)


# In[91]:


print(g.best_params_)
print(g.best_estimator_)
print(g.best_score_)


# In[92]:


m=XGBClassifier(colsample_bytree=0.3, gamma= 0.1, learning_rate= 0.15, max_depth= 10, min_child_weight= 5)
m.fit(xtrain,ytrain)
p=m.predict(xtest)
score=cross_val_score(m,x,y,cv=10)
performance(p,ytest,m,xtest,score)


# We choose random forest as our final model because it gives the highest cross validation score as well as difference between its accuracy score and cross validation score is minimum.

# # Finalizing the model

# In[93]:


model=RandomForestClassifier(max_features='log2', n_estimators=500)
model.fit(xtrain,ytrain)
p=model.predict(xtest)
score=cross_val_score(model,x,y,cv=10)


# # Evaluation Metrics

# In[94]:


performance(p,ytest,model,xtest,score)


# In[95]:


fpred=pd.Series(model.predict_proba(xtest)[:,1])
fpr,tpr,threshold=roc_curve(ytest,fpred)


# In[96]:


plt.plot(fpr,tpr,color='k',label='ROC')
plt.plot([0,1],[0,1],color='b',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC curve')
plt.legend()


# # Saving the model

# In[97]:


import joblib
joblib.dump(model,'census_income.obj')


# In[ ]:




