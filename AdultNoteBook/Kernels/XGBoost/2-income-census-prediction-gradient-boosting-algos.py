#!/usr/bin/env python
# coding: utf-8

# # <span style="color:blue"> INCOME CENSUS DATA CLASSIFICATION WITH GRADIENT BOOSTING ALGORITHMS</span> 

# 
# # DATA PREPROCESSING
# 

# In[ ]:



import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import missingno as msno 


# In[ ]:


data=pd.read_csv("/kaggle/input/adult-census-income/adult.csv")
df=data.copy() 
df.head()


# In[ ]:


df.rename({"education.num":"educational-num","marital.status":"marital-status","sex":"gender","capital.gain":"capital-gain","capital.loss":"capital-loss",
         "hours.per.week":"hours-per-week","native.country":"native-country"},axis=1,inplace=True)


# In[ ]:


print("Rows : {} \nColumns : {}".format(df.shape[0],df.shape[1]))


# In[ ]:


sns.countplot(df["income"],palette="bright")


# In[ ]:


numeric_describe=df.describe().T 
numeric_describe  


# In[ ]:


object_describe=df.describe(include=["object"]).T 
object_describe


# In[ ]:


object_columns=df.select_dtypes(include=["object"]).columns 
for i in range(len(object_columns)):
    print("----- {}-----".format(object_columns[i]))
    print(df[object_columns[i]].value_counts()) 


# In[ ]:


df.isnull().sum() 


# In[ ]:


df=df.replace("?",np.nan) 


# In[ ]:


df.isnull().sum() 


# In[ ]:


nan_percentage = df.isna().sum() * 100 / len(df)
missing_percentage_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': nan_percentage}).reset_index(drop=True)
missing_percentage_df


# In[ ]:


msno.matrix(df) 


# In[ ]:


sns.pairplot(df) 


# In[ ]:


#sns.pairplot(df,hue="income",palette="bright")


# In[ ]:


df.dtypes 


# In[ ]:


df["education"].value_counts() 


# In[ ]:


df["educational-num"].value_counts()


# In[ ]:


df.drop("education",axis=1,inplace=True) 


# In[ ]:


df.head()


# In[ ]:


from pandas.api.types import CategoricalDtype  
df["educational-num"]=df["educational-num"].astype(CategoricalDtype(ordered=True)) 
df["educational-num"].head()


# In[ ]:


df["educational-num"].value_counts()


# In[ ]:


df.dtypes


# In[ ]:


df.head()


# In[ ]:


df.corr() 


# In[ ]:


plt.figure(figsize=(12,6))
plt.rcParams.update({'font.size': 15})
corr=df.corr()
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask) 


# In[ ]:


sns.set(font_scale=2)
plt.figure(figsize=(32,16)) 
#plt.rcParams.update({'font.size': 20}) 
plt.subplot(221)
sns.countplot(df["workclass"],hue=df["income"])
plt.xticks(rotation=90) 

plt.subplot(222)
sns.countplot(df["marital-status"],hue=df["income"])
plt.xticks(rotation=90)

plt.subplot(223)
sns.countplot(df["gender"],hue=df["income"])
plt.xticks(rotation=90)

plt.subplot(224)
sns.countplot(df["race"],hue=df["income"])
plt.xticks(rotation=90)

plt.subplots_adjust(hspace=1) 
plt.show()


# ## HANDLING WITH OUTLIERS

# In[ ]:


df.skew() 


# In[ ]:


df.kurtosis() 


# In[ ]:



plt.figure(figsize=(30,15))
plt.rcParams.update({'font.size': 20})
plt.subplot(321)
sns.boxplot(df["age"])
plt.xticks(rotation=90) 

plt.subplot(322)
sns.boxplot(df["capital-loss"])
plt.xticks(rotation=90)

plt.subplot(323)
sns.boxplot(df["capital-gain"])
plt.xticks(rotation=90)

plt.subplot(324)
sns.boxplot(df["fnlwgt"])
plt.xticks(rotation=90)

plt.subplot(325)
sns.boxplot(df["hours-per-week"])
plt.xticks(rotation=90)

plt.subplots_adjust(hspace=0.6) 
plt.show()


# In[ ]:


df_loss_withoutzero=df.loc[df["capital-loss"]!=0,:] 
df_loss_withoutzero.head()


# In[ ]:


df_gain_withoutzero=df.loc[df["capital-gain"]!=0,:] 
df_gain_withoutzero.head()


# In[ ]:



plt.figure(figsize=(22,7))

plt.subplot(121)
sns.boxplot(df_gain_withoutzero["capital-gain"])
plt.xticks(rotation=90) 

plt.subplot(122)
sns.boxplot(df_loss_withoutzero["capital-loss"])
plt.xticks(rotation=90)


# In[ ]:


numeric_columns=list(df.select_dtypes(include=["int64"]).columns) 
numeric_columns


# ![image.png](attachment:image.png)
# 

# In[ ]:


lower_limits=[]
upper_limits=[]
IQR_values=[]

for i in range(len(numeric_columns)):
    
    Q1=df[numeric_columns[i]].quantile(0.25) 
    Q3=df[numeric_columns[i]].quantile(0.75) 
    IQR=Q3-Q1 
    IQR_values.append(IQR)
    lower_limit=Q1-(1.5*IQR) 
    lower_limits.append(lower_limit)
    upper_limit=Q3+1.5*IQR 
    upper_limits.append(upper_limit)


# In[ ]:


IQR_table=pd.DataFrame({"numeric_columns":numeric_columns,"lower_limits":lower_limits,
                        "upper_limits":upper_limits,"IQR_values":IQR_values})
IQR_table 


# In[ ]:


Q1_loss=df[df["capital-loss"]!=0]["capital-loss"].quantile(0.25)
Q3_loss=df[df["capital-loss"]!=0]["capital-loss"].quantile(0.75)
IQR_loss=Q3_loss-Q1_loss
lower_limit_loss=Q1_loss-(1.5*IQR_loss)
upper_limit_loss=Q3_loss+(1.5*IQR_loss)

print("Capital-Loss Lower Limit :",lower_limit_loss)
print("Capital-Loss Upper Limit :",upper_limit_loss)


# In[ ]:


Q1_gain=df[df["capital-gain"]!=0]["capital-gain"].quantile(0.25)
Q3_gain=df[df["capital-gain"]!=0]["capital-gain"].quantile(0.75)
IQR_gain=Q3_gain-Q1_gain
lower_limit_gain=Q1_gain-(1.5*IQR_gain)
upper_limit_gain=Q3_gain+(1.5*IQR_gain)

print("Capital-Gain için Lower Limit :",lower_limit_gain)
print("Capital-Gain için Upper Limit:",upper_limit_gain)


# In[ ]:


df_loss_withoutzero[(df_loss_withoutzero["capital-loss"]<lower_limit_loss )|( df_loss_withoutzero["capital-loss"]>upper_limit_loss)]["capital-loss"].shape


# In[ ]:


df_gain_withoutzero[(df_gain_withoutzero["capital-gain"]<lower_limit_gain )|( df_gain_withoutzero["capital-gain"]>upper_limit_gain)]["capital-gain"].shape


# In[ ]:


df_gain_withoutzero["capital-gain"].mode()[0] 


# In[ ]:


df_loss_withoutzero["capital-loss"].mode()[0]


# In[ ]:


df[((df["capital-gain"]!=0 )& (df["capital-gain"]<lower_limit_gain )) | ((df["capital-gain"]!=0 )& (df["capital-gain"]>upper_limit_gain ))].head()


# In[ ]:


outlier_gain=((df["capital-gain"]!=0 )& (df["capital-gain"]<lower_limit_gain )) | ((df["capital-gain"]!=0 )& (df["capital-gain"]>upper_limit_gain ))


# In[ ]:


outlier_loss=((df["capital-loss"]!=0 )& (df["capital-loss"]<lower_limit_loss)) | ((df["capital-loss"]!=0 )& (df["capital-loss"]>upper_limit_loss ))


# In[ ]:


df.loc[outlier_gain,"capital-gain"]=df_gain_withoutzero["capital-gain"].mode()[0]


# In[ ]:


df.loc[outlier_loss,"capital-loss"]=df_loss_withoutzero["capital-loss"].mode()[0]


# In[ ]:


print("outlier number for age: {}".format(df[(df["age"]<(lower_limits[0]))|(df["age"]>(upper_limits[0]))].shape[0]))


# In[ ]:


print("outlier number for hours-per-week : {}".format(df[(df["hours-per-week"]<(lower_limits[4]))|(df["hours-per-week"]>(upper_limits[4]))].shape[0]))


# In[ ]:


df.drop(df[df["age"]>upper_limits[0]].index,inplace=True) 


# In[ ]:


df.head()


# In[ ]:


print("Final Weight Outlier Number :{}".format(df[(df["fnlwgt"]<(lower_limits[1]))|(df["fnlwgt"]>(upper_limits[1]))].shape[0]))


# In[ ]:


df.drop(df[df["fnlwgt"]>900000].index,inplace=True) 


# In[ ]:


numeric_describe


# In[ ]:


numeric_describe_2=df.describe().T


# In[ ]:


plt.figure(figsize=(12,6))
plt.rcParams.update({'font.size': 15})
corr=df.corr()
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask) 


# ## HANDLING WITH MISSING VALUES

# In[ ]:


df.isnull().sum() 


# In[ ]:


msno.matrix(df)


# In[ ]:


df["occupation"].value_counts().plot.barh(color="red")


# In[ ]:


df["workclass"].value_counts().plot.barh(color="orange")


# In[ ]:


df["native-country"].value_counts()


# In[ ]:



for i in ["occupation","workclass","native-country"]:
    df[i].fillna(df[i].mode()[0],inplace=True)


# In[ ]:


df.isnull().sum().sum() 


# In[ ]:


df["occupation"].value_counts()


# In[ ]:


df["workclass"].value_counts()


# In[ ]:


numeric_describe


# In[ ]:


numeric_describe_2


# ## HANDLING WITH CATEGORICAL DATA

# In[ ]:


df.head()
df_new=df.copy()


# In[ ]:


df_new=pd.get_dummies(df_new,columns=["gender","income"],drop_first=True) 


# In[ ]:


df_new.head()


# In[ ]:


df_new.rename({"gender_Male":"gender","income_>50K":"income"},axis=1,inplace=True)


# In[ ]:


df_new=pd.get_dummies(df_new,columns=["workclass","marital-status","occupation","relationship","race","native-country"])


# In[ ]:


df_new.head()


# In[ ]:


print("New Column Number :",df_new.shape[1])


# In[ ]:


df_new.columns 


# In[ ]:


df_new.shape


# ## SPLITTING DATA AS TRAIN AND TEST

# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


df_new.head()


# In[ ]:


df_new["income"].value_counts()


# In[ ]:


X=df_new.drop(columns=["income"],axis=1)
X["educational-num"]=X["educational-num"].astype("int") 


# In[ ]:


y=df_new["income"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42) 


# In[ ]:


print("X_train row number :",X_train.shape[0])


# In[ ]:


print("X_test row number :",X_test.shape[0])


# In[ ]:


print("y_train row number :",y_train.shape[0])


# In[ ]:


print("y_test row number :",y_test.shape[0])


# ## GRADIENT BOOSTING MODEL

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 
from sklearn.metrics import roc_auc_score,roc_curve,f1_score,recall_score,precision_score
from sklearn.metrics import accuracy_score
gbm=GradientBoostingClassifier()
gbm_model=gbm.fit(X_train,y_train)
y_pred=gbm_model.predict(X_test)

print("GBM  Accuracy Score :",accuracy_score(y_test,y_pred))
print("GBM  Train Score:",    gbm_model.score(X_train,y_train))
print("GBM  f1 score:",       f1_score(y_test,y_pred))

gbm_train_score=gbm_model.score(X_train,y_train)
gbm_accuracy_score=accuracy_score(y_test,y_pred)
gbm_f1_score=f1_score(y_test,y_pred)
gbm_recall_score=recall_score(y_test,y_pred)
gbm_precision_score=precision_score(y_test,y_pred)


# In[ ]:



from pdpbox import pdp, get_dataset, info_plots

features_to_plot = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
inter1  =  pdp.pdp_interact(model=gbm_model, dataset=X_test, model_features=X_test.columns, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()


# In[ ]:


import shap
shap.initjs()


# In[ ]:


gbm_explainer = shap.TreeExplainer(gbm_model)
gbm_shap_values = gbm_explainer.shap_values(X_test)


# In[ ]:


shap.force_plot(gbm_explainer.expected_value, gbm_shap_values[1,:], X_test.iloc[1,:])


# In[ ]:


shap.summary_plot(gbm_shap_values, X_test)


# In[ ]:


shap.summary_plot(gbm_shap_values,X_test,  plot_type="bar")


# In[ ]:


gbm_recall_score,gbm_precision_score


# In[ ]:


gbm_model


# ![image.png](attachment:image.png)

# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = "coolwarm");
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score:'+str(accuracy_score(y_test,y_pred))
plt.title(all_sample_title, size = 15);
plt.show()
print(classification_report(y_test,y_pred))



# In[ ]:


fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, gbm.predict_proba(X_test)[:,1])
plt.plot([0, 1], [0, 1], 'k--')                                                     
plt.plot(fpr_mlp, tpr_mlp)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.show()


# ## XGBOOST MODEL

# In[ ]:



from xgboost import XGBClassifier


# In[ ]:



xgb=XGBClassifier(seed=42)
xgb_model=xgb.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)
print("XGBoost  Accuracy Score :",accuracy_score(y_test,y_pred))
print("XGBoost  Train Score:",xgb_model.score(X_train,y_train))
print("XGBoost  f1 score:",f1_score(y_test,y_pred))

xgb_train_score=xgb_model.score(X_train,y_train)
xgb_accuracy_score=accuracy_score(y_test,y_pred)
xgb_f1_score=f1_score(y_test,y_pred)
xgb_recall_score=recall_score(y_test,y_pred)
xgb_precision_score=precision_score(y_test,y_pred)


# In[ ]:


xgb_model 


# In[ ]:


xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_test)
shap.summary_plot(xgb_shap_values, X_test)


# In[ ]:


shap.summary_plot(xgb_shap_values,X_test,  plot_type="bar")


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = "coolwarm");
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score:'+str(accuracy_score(y_test,y_pred))
plt.title(all_sample_title, size = 15);
plt.show()
print(classification_report(y_test,y_pred))


# In[ ]:


fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, xgb.predict_proba(X_test)[:,1])
plt.plot([0, 1], [0, 1], 'k--')                                                     
plt.plot(fpr_mlp, tpr_mlp)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.show()


# ## HISTOGRAM BASED BOOSTING MODEL

# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb=HistGradientBoostingClassifier(random_state=42,)
hgb_model=hgb.fit(X_train,y_train)
y_pred=hgb_model.predict(X_test)
print("Histogram Based Boosting  Accuracy Score:",accuracy_score(y_test,y_pred))
print("Histogram Based Boosting  Train Score:",hgb_model.score(X_train,y_train))
print("Histogram Based Boosting  f1 score:",f1_score(y_test,y_pred))

hgb_train_score=hgb_model.score(X_train,y_train)
hgb_accuracy_score=accuracy_score(y_test,y_pred)
hgb_f1_score=f1_score(y_test,y_pred)
hgb_recall_score=recall_score(y_test,y_pred)
hgb_precision_score=precision_score(y_test,y_pred)


# In[ ]:


hgb_model


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'YlGnBu');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score:'+str(accuracy_score(y_test,y_pred))
plt.title(all_sample_title, size = 15);
plt.show()
print(classification_report(y_test,y_pred))


# In[ ]:


fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, hgb.predict_proba(X_test)[:,1])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_mlp, tpr_mlp)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.show()


# ## LIGHTGBM MODEL

# In[ ]:


from lightgbm import LGBMClassifier,plot_importance
lgbm=LGBMClassifier(random_state=42)
lgbm_model=lgbm.fit(X_train,y_train)
y_pred=lgbm_model.predict(X_test)

print("LightGBM için Accuracy Score:",accuracy_score(y_test,y_pred))
print("LightGBM için Train Score:",lgbm_model.score(X_train,y_train))
print("LightGBM için f1 score:",f1_score(y_test,y_pred))

lgbm_train_score=lgbm_model.score(X_train,y_train)
lgbm_accuracy_score=accuracy_score(y_test,y_pred)
lgbm_f1_score=f1_score(y_test,y_pred)
lgbm_recall_score=recall_score(y_test,y_pred)
lgbm_precision_score=precision_score(y_test,y_pred)


# In[ ]:


lgbm_model


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'YlGnBu');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix - score:'+str(accuracy_score(y_test,y_pred))
plt.title(all_sample_title, size = 15);
plt.show()
print(classification_report(y_test,y_pred))


# In[ ]:


fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, lgbm.predict_proba(X_test)[:,1])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_mlp, tpr_mlp)
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC EĞRİSİ')
plt.show()


# ## COMPARE THE RESULTS

# In[ ]:


results=pd.DataFrame({
    "Algorithm":["GBM","XGBoost","HGB","LightGBM"],
    "Accuracy":[gbm_accuracy_score,xgb_accuracy_score,hgb_accuracy_score,lgbm_accuracy_score],
    "Train Score":[gbm_train_score,xgb_train_score,hgb_train_score,lgbm_train_score],
    "f1_Score":[gbm_f1_score,xgb_f1_score,hgb_f1_score,lgbm_f1_score],
     "Recall_Score":[gbm_recall_score,xgb_recall_score,hgb_recall_score,lgbm_recall_score],
      "Precision_Score":[gbm_precision_score,xgb_precision_score,hgb_precision_score,lgbm_precision_score]})

results.sort_values(ascending=False,by="Accuracy")


# ## UNDERSAMPLING BECAUSE OF UNBALANCED DATA

# In[ ]:


random_majority_indices=np.random.choice(df_new[df_new["income"]==0].index,
                                        len(df_new[df_new["income"]==1]),
                                        replace=False)


# In[ ]:


minority_class_indices=df_new[df_new["income"]==1].index
print(minority_class_indices)


# In[ ]:


under_sample_indices=np.concatenate([minority_class_indices,random_majority_indices])


# In[ ]:


under_sample=df_new.loc[under_sample_indices]


# In[ ]:


X=under_sample.drop(columns=["income"],axis=1)
X["educational-num"]=X["educational-num"].astype("int")
y=under_sample["income"]


# In[ ]:


sns.countplot(x="income",data=under_sample)


# ### GBM MODEL AFTER UNDERSAMPLING

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42) 
gbm=GradientBoostingClassifier()
gbm_model=gbm.fit(X_train,y_train)
y_pred=gbm_model.predict(X_test)

print("GBM  Accuracy Score :",accuracy_score(y_test,y_pred))
print("GBM  Train Score:",    gbm_model.score(X_train,y_train))
print("GBM  f1 score:",       f1_score(y_test,y_pred))

gbm_train_score=gbm_model.score(X_train,y_train)
gbm_accuracy_score=accuracy_score(y_test,y_pred)
gbm_f1_score=f1_score(y_test,y_pred)
gbm_recall_score=recall_score(y_test,y_pred)
gbm_precision_score=precision_score(y_test,y_pred)


# ### XGBOOST  AFTER UNDERSAMPLING

# In[ ]:


xgb=XGBClassifier(seed=42)
xgb_model=xgb.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)
print("XGBoost  Accuracy Score :",accuracy_score(y_test,y_pred))
print("XGBoost  Train Score:",xgb_model.score(X_train,y_train))
print("XGBoost  f1 score:",f1_score(y_test,y_pred))

xgb_train_score=xgb_model.score(X_train,y_train)
xgb_accuracy_score=accuracy_score(y_test,y_pred)
xgb_f1_score=f1_score(y_test,y_pred)
xgb_recall_score=recall_score(y_test,y_pred)
xgb_precision_score=precision_score(y_test,y_pred)


# In[ ]:


xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_test)
shap.summary_plot(xgb_shap_values, X_test)


# In[ ]:


shap.summary_plot(xgb_shap_values,X_test,  plot_type="bar")


# In[ ]:


features_to_plot = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
inter1  =  pdp.pdp_interact(model=xgb_model, dataset=X_test, model_features=X_test.columns, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()


# In[ ]:


feature_to_plot = 'capital-gain'
pdp_dist = pdp.pdp_isolate(model=xgb_model, dataset=X_test, model_features=X_test.columns, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# In[ ]:


feature_to_plot = 'capital-loss'
pdp_dist = pdp.pdp_isolate(model=xgb_model, dataset=X_test, model_features=X_test.columns, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()


# ### HGB AFTER UNDERSAMPLING

# In[ ]:



hgb=HistGradientBoostingClassifier(random_state=42,)
hgb_model=hgb.fit(X_train,y_train)
y_pred=hgb_model.predict(X_test)
print("Histogram Based Boosting  Accuracy Score:",accuracy_score(y_test,y_pred))
print("Histogram Based Boosting  Train Score:",hgb_model.score(X_train,y_train))
print("Histogram Based Boosting  f1 score:",f1_score(y_test,y_pred))

hgb_train_score=hgb_model.score(X_train,y_train)
hgb_accuracy_score=accuracy_score(y_test,y_pred)
hgb_f1_score=f1_score(y_test,y_pred)
hgb_recall_score=recall_score(y_test,y_pred)
hgb_precision_score=precision_score(y_test,y_pred)


# ### LGBM AFTER UNDERSAMPLING

# In[ ]:


lgbm=LGBMClassifier(random_state=42)
lgbm_model=lgbm.fit(X_train,y_train)
y_pred=lgbm_model.predict(X_test)

print("LightGBM için Accuracy Score:",accuracy_score(y_test,y_pred))
print("LightGBM için Train Score:",lgbm_model.score(X_train,y_train))
print("LightGBM için f1 score:",f1_score(y_test,y_pred))

lgbm_train_score=lgbm_model.score(X_train,y_train)
lgbm_accuracy_score=accuracy_score(y_test,y_pred)
lgbm_f1_score=f1_score(y_test,y_pred)
lgbm_recall_score=recall_score(y_test,y_pred)
lgbm_precision_score=precision_score(y_test,y_pred)


# In[ ]:


lgb_explainer = shap.TreeExplainer(lgbm_model)
lgb_shap_values = lgb_explainer.shap_values(X_test)
shap.summary_plot(lgb_shap_values, X_test)


# In[ ]:


features_to_plot = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
inter1  =  pdp.pdp_interact(model=lgbm_model, dataset=X_test, model_features=X_test.columns, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()


# ### RESULTS  AFTER  UNDERSAMPLING

# In[ ]:


results=pd.DataFrame({
    "Algorithm":["GBM","XGBoost","HGB","LightGBM"],
    "Accuracy":[gbm_accuracy_score,xgb_accuracy_score,hgb_accuracy_score,lgbm_accuracy_score],
    "Train Score":[gbm_train_score,xgb_train_score,hgb_train_score,lgbm_train_score],
    "f1_Score":[gbm_f1_score,xgb_f1_score,hgb_f1_score,lgbm_f1_score],
     "Recall_Score":[gbm_recall_score,xgb_recall_score,hgb_recall_score,lgbm_recall_score],
      "Precision_Score":[gbm_precision_score,xgb_precision_score,hgb_precision_score,lgbm_precision_score]})

results.sort_values(ascending=False,by="Accuracy")

