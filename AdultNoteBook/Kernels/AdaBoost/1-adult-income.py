#!/usr/bin/env python
# coding: utf-8

# <p id="part0"></p>
# 
# <p style="font-family: Arials; line-height: 2; font-size: 24px; font-weight: bold; letter-spacing: 2px; text-align: center; color: #FF8C00">Adult Income 💸🤑💰 </p>
# 
# <img src="https://miro.medium.com/max/1000/1*08ltbgXFxujakJZSJswp1Q.png" width="100%" align="center" hspace="5%" vspace="5%"/>
# 
# <p style = "font-family: Inter, sans-serif; font-size: 14px; color: rgba(0,0,0,.7)"> An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.</p>
# 
# 
# <p style="font-family: Arials; font-size: 20px; font-style: normal; font-weight: bold; letter-spacing: 3px; color: #808080; line-height:1.0">TABLE OF CONTENT</p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"><a href="#part1" style="color:#808080">0 PROLOGUE</a></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"><a href="#part2" style="color:#808080">1 IMPORTING LIBRARIES</a></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"><a href="#part3" style="color:#808080">2 DATA DESCRIPTION AND DATA CLEANING</a></p>
# 
# <p style="text-indent: 1vw; font-family: Arials; font-size: 14px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3">
# <a href="#part4" style="color:#808080">2.1 Import Data</a></p>
# 
# <p style="text-indent: 1vw; font-family: Arials; font-size: 14px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3">
# <a href="#part5" style="color:#808080">2.2 Data types</a></p>
# 
# <p style="text-indent: 1vw; font-family: Arials; font-size: 14px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3">
# <a href="#part6" style="color:#808080">2.3 Missing values</a></p>
# 
# <p style="text-indent: 1vw; font-family: Arials; font-size: 14px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"> 
# <a href="#part7" style="color:#808080">2.4 Duplicates</a></p>
# 
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"><a href="#part8" style="color:#808080">3 ANALYSIS</a></p>
# 
# <p style="text-indent: 1vw; font-family: Arials; font-size: 14px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"> 
# <a href="#part9" style="color:#808080">3.1 Uni-Vriate Analysis:</a></p>
# 
# <p style="text-indent: 1vw; font-family: Arials; font-size: 14px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"> 
# <a href="#part10" style="color:#808080">3.2 Bi-Vriate Analysis:</a></p>
# 
# <p style="text-indent: 1vw; font-family: Arials; font-size: 14px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"> 
# <a href="#part11" style="color:#808080">3.3 Multi-Vriate Analysis:</a></p>
# 
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"><a href="#part12" style="color:#808080">4 FINAL CONCLUSIONS</a></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 2px; color: #808080; line-height:1.3"><a href="#part13" style="color:#808080">5 MODELLING</a></p>

# <p id="part1"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: bold; font-weight: bold; letter-spacing: 3px; color: #FF8C00">0 PROLOGUE</p>
# <hr style="height: 0.5px; border: 0; background-color: #808080">
# 
# 
# 
# <p style="font-family: Arials, sans-serif; font-size: 14px; color: rgba(0,0,0,.7)"><strong>FEATURES:</strong></p>
# 
# <ol style="font-family: Arials, sans-serif; font-size: 14px; line-height:1.5; color: rgba(0,0,0,.7)">
# <li><strong>AGE</strong> -continuous. </li>
# <p></p>    
# <li><strong> workclass</strong> -Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.</li>
# <p></p>
# <li><strong>fnlwgt</strong> -continuous.</li>
# <p></p>    
# <li><strong>education</strong> - Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.</li>   
# <p></p>    
# <li><strong>education-num</strong> - continuous.</li>   
# <p></p>    
# <li><strong>marital-status</strong> -Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.</li>    
# <p></p>    
# <li><strong>occupation</strong> -Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.</li>      
# <p></p>     
# <li><strong>relationship</strong> -Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.</li>    
# <p></p> 
# <li><strong>race</strong> -White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.</li>    
# <p></p> 
# <li><strong>sex</strong> -Female, Male.</li>    
# <p></p>
# <li><strong>capital-gain</strong> -continuous.</li>    
# <p></p> 
# <li><strong>capital-loss</strong> -continuous.</li>    
# <p></p>
# <li><strong>hours-per-week</strong> -continuous.</li>    
# <p></p>
# <li><strong>native-country</strong> -United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.</li>    
# <p></p>
# <li><strong>class</strong> - >50K, <=50K</li>    
# <p></p>
#    
# </ol>
# 
# 
# 
# 

# <p style="font-family: Arials; line-height: 1.5; font-size: 16px; font-weight: bold; letter-spacing: 2px; text-align: center; color: #FF8C00">If you liked this notebook, please upvote.</p>
# <p style="text-align: center">😊😊😊</p>

# <p id="part2"></p>
# 
# # <span style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 3px; color: #FF8C00">1 IMPORTING LIBRARIES</span>
# <hr style="height: 0.5px; border: 0; background-color: #808080">
# 
# <p style="font-family: Arials, sans-serif; font-size: 14px; line-height:1.0; color: rgba(0,0,0,.7)"><strong>LIBRARIES:</strong></p>
# 
# <ol style="font-family: Arials, sans-serif; font-size: 14px; line-height:1.5; color: rgba(0,0,0,.7)">
# <li>Library <strong>pandas</strong> will be required to work with data in tabular representation.</li>
# <p></p>
# <li>Library <strong>numpy</strong> will be required to round the data in the correlation matrix.</li>
# <p></p>
# <li>Library <strong>missingno</strong> will be required to visualize missing values in the data.</li>
# <p></p>   
# <li>Library <strong>matplotlib, seaborn, plotly</strong> required for data visualization.</li>
# <p></p>
# </ol>

# In[1]:


## for eda and visuls:
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import missingno
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff



# <p id="part3"></p>
# 
# # <span style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: bold; letter-spacing: 3px; color: #FF8C00">2 DATA DESCRIPTION AND DATA CLEANING</span>
# <hr style="height: 0.5px; border: 0; background-color: #808080">
# 
# <p style="font-family: Arials, sans-serif; font-size: 14px; color: rgba(0,0,0,.7)">In this block, cleaning part will be carried out, data types, missing values, duplicates.</p>

# <p id="part4"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">2.1 Import Data</p>

# In[2]:


# Reading Data:
df=pd.read_csv("/kaggle/input/adult-census-income/adult.csv")
df.head()  #Loading the First Five Rows:


# In[3]:


# Let's Look The Dimensions Of The Data:
print(f'The Data-Set Contain {df.shape[0]} Rows and {df.shape[1]} Columns')


# <p id="part5"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">2.2 Data Types</p>

# In[4]:


#Check Data Types
df.dtypes


# <p id="part6"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">2.3 Missing values</p>

# <p style = "font-family: Inter, sans-serif; font-size: 14px; color: rgba(0,0,0,.7)"> Let's calculate the percentage of blanks and filled values for all columns.</p>

# In[5]:


# loop through the columns and check the missing values
for col in df.columns:
    pct_missing = df[col].isnull().sum()
    print(f'{col} - {pct_missing :.1%}')


# In[6]:


# Build a matrix of missing values
missingno.matrix(df, fontsize = 16)
plt.show()


# <div style="background: #DCDCDC"><p style="font-family: Arials, sans-serif; font-size: 16px; color: #000000"><strong>CONCLUSION:</strong> The data has no missing values, so no further transformations are required.</p></div>

# <p id="part7"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">2.4 Duplicates</p>

# In[7]:


#Check The Duplicates In the Data-Set:
df.duplicated().sum()


# <p style = "font-family: Inter, sans-serif; font-size: 14px; color: rgba(0,0,0,.7)">There are 24 Duplicate Value Present in the Data-set.</p>

# In[8]:


# We will drop the Duplicate value:
df=df.drop_duplicates(keep="first")


# ### In some columns there is ? present or null values let's handle that also:

# In[9]:


df["workclass"]=df["workclass"].replace("?",np.nan)
df["occupation"]=df["occupation"].replace("?",np.nan)
df["native.country"]=df["native.country"].replace("?",np.nan)


# In[10]:


df.isna().sum()


# In[11]:


df["workclass"]=df["workclass"].fillna(df["workclass"].mode()[0])
df["occupation"]=df["occupation"].fillna(df["occupation"].mode()[0])
df["native.country"]=df["native.country"].fillna(df["native.country"].mode()[0])


# <div style="background: #DCDCDC"><p style="font-family: Arials, sans-serif; font-size: 16px; color: #000000"><strong>CONCLUSION:</strong>Now our Data is Clean We can do Further Analysis.</p></div>

# <p id="part8"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">3. Analysis:</p>

# <p id="part9"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">3.1 Uni-variate Analysis:</p>

# In[12]:


fig =  plt.figure(figsize = (15,6))
fig.patch.set_facecolor('#f5f6f6')


                                                   
gs = fig.add_gridspec(2,3)
gs.update(wspace=0.2,hspace= 0.2)

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])

axes=[ax0,ax1,ax2,ax3,ax4,ax5]
for ax in axes:
    ax.set_facecolor('#f5f6f6')
    ax.tick_params(axis='x',
                   labelsize = 12, which = 'major',
                   direction = 'out',pad = 2,
                   length = 1.5)
    ax.tick_params(axis='y', colors= 'black')
    ax.axes.get_yaxis().set_visible(False)
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)


        
cols = df.select_dtypes(exclude = 'object').columns

sns.kdeplot(x = df[cols[0]],color="green",fill=True,ax = ax0)
sns.kdeplot(x = df[cols[1]],color="red",fill=True,ax = ax1)
sns.kdeplot(x = df[cols[2]],color="blue",fill=True,ax = ax2)
sns.kdeplot(x = df[cols[3]],color="black",fill=True,ax = ax3)
sns.kdeplot(x = df[cols[4]],color="pink",fill=True,ax = ax4)
sns.kdeplot(x = df[cols[5]],color="green",fill=True,ax = ax5)

fig.text(0.2,0.98,"Univariate Analysis on Numerical Columns:",**{'font':'serif', 'size':18,'weight':'bold'}, alpha = 1)
fig.text(0.1,0.90,"Most of the adults are range of 20-45 and on average an adult spend around 40hrs per week on work\n Also as we can see there is so much otliers present in the numerical columns:",**{'font':'serif', 'size':12,'weight':'bold'}, alpha = 1)


# In[13]:


df.select_dtypes(include="object").columns


# In[14]:


income=df["income"].reset_index()
px.pie(values=income["index"],names=income["income"], color_discrete_sequence=px.colors.sequential.RdBu,
      title='Income of the Adults')


# In[15]:


sex=df["sex"].reset_index()
px.pie(values=sex["index"],names=sex["sex"],title='%AGE OF MALE AND FEMALE', hole=.3)


# In[16]:


race=df["race"].reset_index()
px.pie(values=race["index"],names=race["race"])


# In[17]:


relationship=df["relationship"].reset_index()
px.pie(values=relationship["index"],names=relationship["relationship"])


# In[18]:


occupation=df["occupation"].reset_index()
px.pie(values=occupation["index"],names=occupation["occupation"])


# In[19]:


marital_status=df["marital.status"].reset_index()
px.pie(values=marital_status["index"],names=marital_status["marital.status"])


# In[20]:


education=df["education"].reset_index()
px.pie(values=education["index"],names=education["education"])


# In[21]:



fig=plt.figure(figsize=(10,6))
ax=sns.countplot(df["workclass"])
plt.title("COUNT OF WORK CLASS")

for loc in ['left', 'right', 'top', 'bottom']:
    ax.spines[loc].set_visible(False)
    

fig.show()

#workcls=df["workclass"].reset_index()
#px.pie(values=workcls["index"],names=workcls["workclass"])


# <p id="part10"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">3.2 Bi-Variate Analysis:</p>

# In[22]:


df.head(1)


# In[23]:


fig=plt.figure(figsize=(10,6))
ax=sns.kdeplot(x=df["age"],hue=df["income"],fill=True)
ax.set_facecolor('#f5f6f6')
for loc in ['left', 'right', 'top', 'bottom']:
    ax.spines[loc].set_visible(False)
 
fig.text(0.4,1,"Distribution of income with age:",**{'font':'serif', 'size':18,'weight':'bold'}, alpha = 1)
fig.text(0.1,0.90,"First of all most of the adults have income less than 50k \n But With increasing in age Income is also increasing :",**{'font':'serif', 'size':12,}, alpha = 1)


fig.show()


# In[24]:


fig=plt.figure(figsize=(10,6))
ax=sns.kdeplot(x=df["education.num"],hue=df["income"],fill=True,)
ax.set_facecolor('#f5f6f6')
for loc in ['left', 'right', 'top', 'bottom']:
    ax.spines[loc].set_visible(False)
 
fig.text(0.2,1,"Distribution of Number of years of Education with Income:",**{'font':'serif', 'size':18,'weight':'bold'}, alpha = 1)
fig.text(0.1,0.90,"With increasing in years of education Income is also increasing :",**{'font':'serif', 'size':12,'weight':'bold'}, alpha = 1)


fig.show()


# <p id="part11"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: normal; font-weight: normal; letter-spacing: 3px; color: #FF8C00; line-height:1.0">3.3 Multi-Variate Analysis:</p>

# <p id="part13"></p>
# 
# <p style="font-family: Arials; font-size: 16px; font-style: bold; font-weight: bold; letter-spacing: 3px; color: #FF8C00; line-height:1.0">5. Modelling:</p>
# 
# 

# <p style="font-family: Arials; font-size: 16px; font-style: bold; font-weight: bold; letter-spacing: 3px; color: #FF8C00; line-height:1.0">5.0 Make data ready for Modelling:</p>

# In[25]:


df.income.unique()


# In[26]:


df["income"]=df["income"].map({"<=50K":0,">50K":1})


# In[27]:


X = df.drop(['income'], axis=1)
y = df['income']


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[29]:


from sklearn import preprocessing

categorical = ['workclass','education', 'marital.status', 'occupation', 'relationship','race', 'sex','native.country',]
for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# In[30]:



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# ## Random forest

# In[31]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))


# ## knn

# from sklearn.neighbors import KNeighborsClassifier
# neig = np.arange(1, 25)
# train_accuracy = []
# test_accuracy = []
# # Loop over different values of k
# for i, k in enumerate(neig):
#     # k from 1 to 25(exclude)
#     knn = KNeighborsClassifier(n_neighbors=k)
#     # Fit with knn
#     knn.fit(X_train,y_train)
#     #train accuracy
#     train_accuracy.append(knn.score(X_train, y_train))
#     # test accuracy
#     test_accuracy.append(knn.score(X_test, y_test))
# 
# # Plot
# plt.figure(figsize=[13,8])
# plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
# plt.plot(neig, train_accuracy, label = 'Training Accuracy')
# plt.legend()
# plt.title('-value VS Accuracy')
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.xticks(neig)
# plt.savefig('graph.png')
# plt.show()
# print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

# In[32]:



from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

y_test_pred=model.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


test_score = accuracy_score(y_test, model.predict(X_test)) * 100
train_score = accuracy_score(y_train, model.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df


# In[35]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
dt=model.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))


# In[36]:


from sklearn.svm import SVC
model=SVC(kernel="rbf")
model.fit(X_train,y_train)
sv=model.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))


# In[37]:


from sklearn.ensemble import AdaBoostClassifier

model=AdaBoostClassifier(learning_rate= 0.15,n_estimators= 25)
model.fit(X_train,y_train)
ab=model.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))


# ## hyper para meter tuning:

# In[38]:


from sklearn.model_selection import GridSearchCV


# In[39]:


clf = DecisionTreeClassifier()
# Hyperparameter Optimization
parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 50], 
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Train the model using the training sets 
clf.fit(X_train, y_train)


# In[40]:


y_pred = clf.predict(X_test)


# In[41]:


# Calculating the accuracy
acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Accuracy of Decision Tree model : ', acc_dt )


# <p style="font-family: Arials; line-height: 1.5; font-size: 16px; font-weight: bold; letter-spacing: 2px; text-align: center; color: #FF8C00">Thank you for reading this work!
# Any feedback on this work would be very grateful.
# If you liked this notebook, Upvote.</p>
#     <p style="text-align: center">😊😊😊</p>

# In[ ]:




