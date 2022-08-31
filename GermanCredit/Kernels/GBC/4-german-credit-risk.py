#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir('../input'))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# index_col function help to remove the coloumn

df= pd.read_csv('../input/german credit1.csv',index_col=0)


# In[ ]:


# shape of the data
df.shape


# In[ ]:


df.info()


# In[ ]:


# removing the Unnamed column
#df.drop([' '],axis =1 )
sns.pairplot(df)


# In[ ]:


def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))


# In[ ]:





# In[ ]:


df['Credit amount'].hist()


# In[ ]:


df['Credit amount_log'] = np.log(df['Credit amount'])


# In[ ]:


df['Credit amount_log'].hist()


# In[ ]:


# summary statistics help to understand the distribution of data
# if the SD of any variable is 0 then we need to get rid of that 
# we will not get for categorical variable , only for numerical and continious numerical variable

df.describe()


# In[ ]:


cols=df.columns.tolist()


# In[ ]:


cols


# In[ ]:


cols = cols[-1:] + cols[:-1]


# In[ ]:


cols


# In[ ]:


df=df[cols]


# In[ ]:


df.head()


# In[ ]:


# Null data
df.isnull().sum()


# In[ ]:


# this will help us to know the fields under each header

print("Purpose : ",df.Purpose.unique())
print("Job : ",df.Job.unique())
print("Sex : ",df.Sex.unique())
print("Housing : ",df.Housing.unique())
print("Saving account : ",df['Saving account'].unique())
print("Checking account : ",df['Checking account'].unique())
print("Risk : ",df['Risk'].unique())


# In[ ]:


print("Saving accounts : ",df['Saving account'].value_counts())
print("Checking account : ",df['Checking account'].value_counts())


# # EDA

# # Exploring data

# In[ ]:


sns.countplot('Risk', data=df)


# In[ ]:


sns.countplot('Sex', data=df)


# In[ ]:


dimension = (15,5)
fig, ax = plt.subplots(figsize=dimension)
sns.countplot('Purpose', data=df)


# In[ ]:


sns.countplot('Saving account', data=df)


# In[ ]:


sns.countplot('Checking account', data=df)


# In[ ]:


dimension = (11, 6)
fig, ax = plt.subplots(figsize=dimension)
sns.countplot('Purpose', data=df)


# In[ ]:


sns.catplot(x='Purpose', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)
plt.title('Mean Credit Amount by purpose and Risk')
plt.show()


# In[ ]:


sns.catplot(x='Duration', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)
plt.title('Mean Duration by Credit Amount and Risk')
plt.show()


# In[ ]:


sns.catplot(x='Duration', y='Credit amount', hue='Sex', kind='bar', palette='Set1', data=df, height=4, aspect=4)
plt.title('Mean Duration by Credit Amount and Sex')
plt.show()


# In[ ]:


sns.catplot(x='Job', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)
plt.title('Mean job by Credit Amount and Risk')
plt.show()


# In[ ]:


sns.catplot(x='Job', y='Credit amount', hue='Sex', kind='bar', palette='Set1', data=df, height=4, aspect=4)
plt.title('Mean job by Credit Amount and sex')
plt.show()


# In[ ]:


sns.catplot(x='Checking account', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)
plt.title('Mean Checking account by Credit Amount and Risk')
plt.show()


# In[ ]:


sns.catplot(x='Saving account', y='Credit amount', hue='Risk', kind='bar', palette='Set1', data=df, height=4, aspect=4)
plt.title('Mean Saving accounts by Credit Amount and Risk')
plt.show()


# In[ ]:


dimension = (15, 6)
fig, ax = plt.subplots(figsize=dimension)
sns.countplot(x="Duration", data=df, 
              palette="hls",  hue = "Risk")


# In[ ]:


category = ["Checking account", 'Sex']
cm = sns.light_palette("pink", as_cmap=True)
pd.crosstab(df[category[0]],df[category[1]]).style.background_gradient(cmap = cm)


# In[ ]:


category = ["Saving account", 'Sex']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df[category[0]],df[category[1]]).style.background_gradient(cmap = cm)


# In[ ]:


category = ["Purpose", 'Sex']
cm = sns.light_palette("blue", as_cmap=True)
pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)


# In[ ]:


category = ["Sex", 'Risk']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)


# In[ ]:


category = ["Housing",'Sex']
cm = sns.light_palette("black", as_cmap=True)
pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)


# In[ ]:


category = ["Job",'Sex']
cm = sns.light_palette("violet", as_cmap=True)
pd.crosstab(df[category[0]], df[category[1]]).style.background_gradient(cmap = cm)


# In[ ]:


sns.catplot(x='Sex', y='Age', hue='Risk', kind='bar', palette='Set1', data=df, height=3, aspect=3)
plt.title('Mean Sex by Age and Risk')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.boxplot(x='Checking account', y='Credit amount', hue=None, data=df, palette='Set1')


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.boxplot(x='Saving account', y='Credit amount', hue=None, data=df, palette='Set1')


# In[ ]:


def scatters(credit, h=None, pal=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.scatterplot(x="Credit amount",y="Duration", hue=h, palette='Set1', data=df, ax=ax1)
    sns.scatterplot(x="Age",y="Credit amount", hue=h, palette='Set1', data=df, ax=ax2)
    sns.scatterplot(x="Age",y="Duration", hue=h, palette='Set1', data=df, ax=ax3)
    plt.tight_layout()


# In[ ]:


scatters(df, h="Saving account")


# In[ ]:


scatters(df, h="Checking account")


# In[ ]:


scatters(df, h="Risk")


# In[ ]:


scatters(df, h="Sex")


# # Cleaning 

# In[ ]:


# this will help to replace all the NAN values with little values in both saving and checking account

#df["Saving accounts"]=df["Saving accounts"].fillna(method="bfill")
#df["Checking account"]=df["Checking account"].fillna(method="bfill")

df.fillna('little',inplace=True)


# In[ ]:


print("Saving account : ",df['Saving account'].value_counts())
print("Checking account : ",df['Checking account'].value_counts())


# In[ ]:


df.info()


# # Model Building

# In[ ]:


features = df.iloc[:,:10]
label = df.iloc[:,[-1]]


# In[ ]:


features.head()


# In[ ]:


label.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


SexinNumeric=LabelEncoder()
HousinginNumeric=LabelEncoder()
SavingaccountinNumeric=LabelEncoder()
CheckingaccountinNumeric=LabelEncoder()
PurposeinNumeric=LabelEncoder()
RiskinNumeric=LabelEncoder()


# In[ ]:


features['SexinNumeric']=SexinNumeric.fit_transform(features['Sex'])
features['HousinginNumeric']=HousinginNumeric.fit_transform(features['Housing'])
features['SavingaccountinNumeric']=SavingaccountinNumeric.fit_transform(features['Saving account'])
features['CheckingaccountinNumeric']=CheckingaccountinNumeric.fit_transform(features['Checking account'])
features['PurposeinNumeric']=PurposeinNumeric.fit_transform(features['Purpose'])
label['RiskinNumeric']=RiskinNumeric.fit_transform(label['Risk'])


# In[ ]:


features.tail()


# In[ ]:


label.tail()


# In[ ]:


NewFeatures = features.drop(['Sex','Housing','Saving account','Checking account', 'Purpose'], axis='columns')
NewLabel = label.drop(['Risk'], axis='columns')


# In[ ]:


NewFeatures.head()


# In[ ]:


NewLabel.head()


# In[ ]:


NewFeatures = features.drop(['Sex','Housing','Saving account', 'Checking account', 'Purpose'], axis='columns').values
NewLabel = label.drop(['Risk'], axis='columns').values


# # Test & Train Splitting

# In[ ]:


#Create Train Test Split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(NewFeatures,
                                                NewLabel,
                                                test_size=0.20,
                                                random_state=44)


# In[ ]:


# Verifying

print(f'X_train dimension: {X_train.shape}')
print(f'X_test dimension: {X_test.shape}')
print(f'\ny_train dimension: {y_train.shape}')
print(f'y_test dimension: {y_test.shape}')


# In[ ]:


#print("Saving account : ",X_train['SavingaccountinNumeric'].value_counts())
#print("Checking account : ",X_train['CheckingaccountinNumeric'].value_counts())


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train.ravel())


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


ypred=model.predict(X_test)


# In[ ]:


print(ypred)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print(accuracy_score(ypred, y_test))


# In[ ]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(ypred,y_test)
CM


# In[ ]:


sns.heatmap(CM, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('actual')


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(ypred,y_test))


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[ ]:


y_pred_prob = model.predict_proba(X_test)[:,1]


# In[ ]:


Log_roc = roc_auc_score(y_test,y_pred_prob)
fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob)


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive rate(100-specificity)')
plt.ylabel('True Positive rate(sensitivity)')
plt.legend(loc='lower right')
plt.show()


# # KNN Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


model2 = KNeighborsClassifier(n_neighbors=25) #k = 5
model2.fit(X_train,y_train.ravel())


# In[ ]:


model2.score(X_train,y_train)


# In[ ]:


model2.score(X_test,y_test)


# In[ ]:


ypred2=model2.predict(X_test)


# In[ ]:


print(ypred2)


# In[ ]:


CM2 = confusion_matrix(ypred,y_test)
CM2


# In[ ]:


sns.heatmap(CM2, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('actual')


# In[ ]:


print(classification_report(ypred2,y_test))


# In[ ]:


y_pred_prob2 = model2.predict_proba(X_test)[:,1]


# In[ ]:


Log_roc2 = roc_auc_score(y_test,y_pred_prob2)
fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob2)


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc2)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive rate(100-specificity)')
plt.ylabel('True Positive rate(sensitivity)')
plt.legend(loc='lower right')
plt.show()


# # decission tree model

# In[ ]:


from sklearn import tree


# In[ ]:


model3 = tree.DecisionTreeClassifier()


# In[ ]:


model3=model3.fit(X_train,y_train)


# In[ ]:


model3.score(X_test,y_test)


# In[ ]:


ypred3=model3.predict(X_test)


# In[ ]:


print(ypred3)


# In[ ]:


from sklearn.metrics import confusion_matrix
CM3 = confusion_matrix(ypred3,y_test)
CM3


# In[ ]:


sns.heatmap(CM3, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('actual')


# In[ ]:


print(classification_report(ypred3,y_test))


# In[ ]:


y_pred_prob3 = model3.predict_proba(X_test)[:,1]


# In[ ]:


Log_roc3 = roc_auc_score(y_test,y_pred_prob3)
fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob3)


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc3)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive rate(100-specificity)')
plt.ylabel('True Positive rate(sensitivity)')
plt.legend(loc='lower right')
plt.show()


# # RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=29)
model4.fit(X_train, y_train.ravel())


# In[ ]:


model4.score(X_train,y_train)


# In[ ]:


model4.score(X_test,y_test)


# In[ ]:


ypred4 = model4.predict(X_test)


# In[ ]:


print(ypred4)


# In[ ]:


from sklearn.metrics import confusion_matrix
CM4 = confusion_matrix(ypred4,y_test)
CM4


# In[ ]:


sns.heatmap(CM4, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('actual')


# In[ ]:


print(classification_report(ypred4,y_test))


# In[ ]:


y_pred_prob4 = model4.predict_proba(X_test)[:,1]


# In[ ]:


Log_roc4 = roc_auc_score(y_test,y_pred_prob4)
fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob4)


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc4)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive rate(100-specificity)')
plt.ylabel('True Positive rate(sensitivity)')
plt.legend(loc='lower right')
plt.show()


# # Support Vector Machine

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model5 = SVC(kernel='linear', probability=False)


# In[ ]:


model5.fit(X_train,y_train.ravel())


# In[ ]:


model5.score(X_train,y_train)


# In[ ]:


model5.score(X_test,y_test)


# In[ ]:


ypred5 = model5.predict(X_test)


# In[ ]:


print(ypred5)


# In[ ]:


print(classification_report(ypred5,y_test))


# In[ ]:


CM5 = confusion_matrix(ypred5,y_test)
CM5


# In[ ]:


sns.heatmap(CM5, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('actual')


# In[ ]:





# In[ ]:


# y_pred_prob5 = model5.predict_proba(X_test)[:,1]


# In[ ]:


Log_roc5 = roc_auc_score(y_test,y_pred_prob5)
fpr, tpr, thresholds =  roc_curve(y_test,y_pred_prob5)


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, label="model area (area = %0.2f)" % Log_roc5)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive rate(100-specificity)')
plt.ylabel('True Positive rate(sensitivity)')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model


# In[ ]:


print(cross_val_score(model,X_test,y_test,cv=10,scoring='accuracy').mean())


# In[ ]:


print(cross_val_score(model2,X_test,y_test,cv=10,scoring='accuracy').mean())


# In[ ]:


print(cross_val_score(model3,X_test,y_test,cv=10,scoring='accuracy').mean())


# In[ ]:


print(cross_val_score(model4,X_test,y_test,cv=10,scoring='accuracy').mean())


# In[ ]:


print(cross_val_score(model5,X_test,y_test,cv=10,scoring='accuracy').mean())


# In[ ]:




