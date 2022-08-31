#!/usr/bin/env python
# coding: utf-8

# # Titanic EDA and Feature Engineering tutorial - Building intuition and best practices

# This kernel serves as a **beginner tutorial and basic guideline** for approaching the Exploratory data analysis and Feature enginnering. In the following Exploratory data analysis, i have tried to impart:
# * My Intuition about how to approach a dataset and ask questions, with documented reasons for creating any visualization and what value did it add to our understanding of the dataset.
# * Best practices.
# * Creating features that boost score and why they do so.
# 
# **First** we will go through how to use EDA to best fill the missing Fare and Cabin feature and check out other possibilities and why is the option we pick better than other missing value imputation methods, with support from the visualizations.
# 
# Then we will check out feature creation possibilities using Exploratory Data Analysis and i will guide you through the attributes in the visulizations that actually make the feature quite expressive and good for achieving better score.
# 
# After that we will look into creating a Feature engineering class to quickly narrow down to best set of features from the many features we engineered.

# In[1]:


get_ipython().system('pip install seaborn==0.11.0 # upgrading seaborn library to use newer plots and features!!')


# In[2]:


# libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Encoders
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

# Strategic imports
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Machine learning Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, VotingClassifier


# imports to mute warnings
pd.options.display.max_rows=200
pd.set_option('mode.chained_assignment', None)

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)


# In[3]:


# seaborn color palettes we will be using 

palette_1 = sns.color_palette('Accent', 6)
palette_2 = sns.color_palette('Set1', 6)
palette_3 = sns.color_palette('BrBG')
palette_4 = sns.color_palette('CMRmap')
palette_5 = sns.color_palette('Paired', 6)
palette_6 = sns.color_palette('RdYlBu')
palette_binary_1 = sns.color_palette('Accent_r', 2)
palette_binary_2 = sns.color_palette('Set1', 2)
palette_binary_3 = sns.color_palette('Set2', 2)

plt.style.use('fivethirtyeight')


# In[4]:


train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')
test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')


# **Small Note before we start:**
# 
# There are a lot of excellent EDA kernels, but few of them combine training and test set to perform EDA, for the given dataset i do not think it is as big of a deal(why i think so, i will explain at the end) as it should be, you should never combine train and test set!! titanic dataset is the starting point of many individuals in the field of data science and looking at this practice can make it very easy for beginners to develop the habit of peeking into the test set, i would go as far as saying that once you have separated test and training sets you should consider test set as something that you will get in future once you have trained the best model as per training data, consider it as something which is only given to you when you are to make final predictions.
# 
# Now why do i think it is not too bad to do so, reason is that... if we look at it from competition point of view, using the test data to extract information which can be helpful to make it easier for making predictions on test data and boost your raking is plausible, Given that titanic dataset is something which will always stay of the same size, that is no new out of sample data will be generated , such practice can help boost score. But it is deterimental for beginners to see such practice in action, developing the habit to combine test and train datasets is easy, the pros who do it in there kernels already now the implications, but a beginner does not. So i wanted to just point this out before you begin going through the kernel, We have avoided combining test and training dataset in EDA.

# # Exploratory Data Analysis
# 
# 
# 
# Lets start by exploring some features and there relations with other features.

# ### Age - Sex distributions

# In[5]:


fig, ax = plt.subplots(1, 2, figsize=(16, 8))
#sns.set_style('ticks')

sns.kdeplot(data=train, x='Age', hue='Sex', fill=True, palette=palette_binary_3, ax=ax[0])
sns.boxenplot(data=train, x='Sex', y='Age', ax=ax[1], palette=palette_2)

sns.despine()
plt.show()


# ### Sex - Fare distributions

# In[6]:


fig, ax = plt.subplots(1, 3, figsize=(16, 6))
#sns.set_style('ticks')

sns.kdeplot(data=train, x='Fare', hue='Sex', fill=True, palette=palette_binary_2, ax=ax[0])
sns.boxenplot(data=train, x='Sex', y='Fare', ax=ax[1], palette=palette_5)

sns.violinplot(data=train, x='Sex', y='Fare', ax=ax[2], palette=palette_3)

sns.despine()
plt.show()


# ### Pclass - Sex - Age distributions

# In[7]:


fig, ax = plt.subplots(1, 3, figsize=(16, 6))

sns.boxenplot(data=train, x='Pclass', y='Age', hue='Sex', dodge=True, ax=ax[0], palette=palette_1)

sns.violinplot(data=train, x='Pclass', y='Age', hue='Sex', split=True, ax=ax[1], palette=palette_6)

sns.stripplot(data=train, x='Pclass', y='Age', hue='Sex', dodge=True, ax=ax[2], palette=palette_binary_3)

sns.despine()
plt.show()


# ## Missing value imputation
# 
# For beginners EDA can become quite an overwhelming task, sometimes it may even be difficult to take the first step, you just don't know where to start and how to ask questions to your data!! My advice is to just start with plotting the relationship of continuous and categorical features with the target. That will tell you alot about the relationship of features with the target, it may give insights on whether some features can be simplified, combined and many more.
# 
# One basic employment of EDA is to find the best possible way to impute missing values.....
# While imputing missing values it is always beneficial to perform a correlation check and see which features correlate well with the feature we wish to impute, then you may make some visualizations to better understand the way you should impute the data.

# In[8]:


train.isna().sum()


# In[9]:


test.isna().sum()


# So we only have missing values in Age, Cabin for both train and test, Embarked and Fare have missing values for train and test respectively, we can impute them simply using median or mean of the whole feature.

# In[10]:


plt.figure(figsize=(12, 8))
sns.set(font_scale=1.5)
sns.set_style('white')

corr = train.corr()

# Generate a mask for the upper triangle
#mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, cmap='coolwarm', annot=True, annot_kws={'size':15})
plt.show()


# Pclass correlates best with the age feature, so we should use the pclass to impute the missing age values!! now lets check if there are any relationship of age with respect to pclass and sex combined

# In[11]:


plt.figure(figsize=(15, 8))
plt.style.use('fivethirtyeight')
#sns.set_style('ticks')
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black', 'linewidth':0.3},
    'medianprops':{'color':'black', 'linewidth':1.5},
    'whiskerprops':{'color':'black', 'linewidth':0.3},
    'capprops':{'color':'black', 'linewidth':0.3},
}

sns.boxplot(x='Pclass', data=train, y='Age', hue='Sex', showfliers=False,  **PROPS)
sns.stripplot(data=train, x='Pclass', hue='Sex', y='Age', palette=palette_binary_3, dodge=True)

sns.despine()
plt.legend(loc='upper right')
plt.show()


# We can see that the meidan age for each gender is slightly different and there is a general trend of decrease in median age for both the genders from Pclass 1 to 3.

# Now the next features with missing value is the Cabin feature, but the Cabin feature has a lot of missing value for us to arrive at a acceptable conclusion to impute missing values using correlation, and the feature has to many missing values for us to simply impute the missing values with the mean or median for the values of the feature.

# In[12]:


temp = train.copy()
temp['Cabin'] = temp.Cabin.str.extract(pat='([A-Z])')

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
plt.style.use('fivethirtyeight')

sns.countplot(data=temp, x='Cabin', hue='Pclass', ax=ax[0], palette=palette_1)
ax[0].set_title('Pclass-Cabin Proportions', x=0.28, y=1.04, size=25)

temp.Cabin.fillna('missing', inplace=True)
temp_missing = temp.loc[temp.Cabin == 'missing']

sns.countplot(data=temp_missing, x='Cabin', hue='Pclass', palette=palette_1)
ax[1].set_title('Missing Cabin proportions', x=0.27, y=1.04, size=25)

sns.despine()
plt.show()


# One way to impute values would be to just fill in the Cabins using the proportions as shown above. The missing values for Cabins in Pclass 1 as we see in the right plot, are less than all the other classes and these missing values will be imputed into the Cabins A,B,C,D,E only as they are the only Cabins where the Pclass 1 passengers are present, depending upon the already available Pclass 1 passenger propotions in these Cabins as shown left plot, But imputing values like this will create noise in already available information!! We might get a great result if the missing values got imputed correctly, But that will depend upon luck. So we should just impute the missing value as 'missing', atleast we retain the already available information free of noise.
# 
# We should also just simplify this feature and Combine Cabin A, B, C as deck ABC and D, E as DE and F, G as FG.

# In[13]:


temp = train.copy()
temp['Cabin'] = temp.Cabin.str.extract(pat='([A-Z])')
temp.Cabin.fillna('missing', inplace=True)

sns.set(font_scale=1.4)
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(2, 2, figsize=(16, 17))
sns.countplot(data=temp, x='Cabin', hue='Pclass', ax=ax[0, 0], palette=palette_5)
ax[0, 0].set_title('Cabin-Pclass proportions', x=0.28, y=1.04, size=25)



temp.Cabin.replace({'A':'ABC', 'B':'ABC', 'C':'ABC', 
                    'D':'DE', 'E':'DE', 'F':'FG', 
                    'G':'FG', 'T':'ABC', 'missing':'M'}, inplace=True)


sns.countplot(data=temp, x='Cabin', hue='Pclass', ax=ax[0, 1], palette=palette_5)
ax[0, 1].set_title('Deck-Pclass Proportions', x=0.27, y=1.04, size=25)
ax[0, 1].set_xlabel('Deck', size=18)



sns.barplot(data=temp, x='Cabin', y='Survived', hue='Pclass', ax=ax[1, 0], palette=palette_5)
ax[1, 0].set_title('Deck Survival rate', x=0.18, y=1.02, size=25)
ax[1, 0].set_xlabel('Deck', size=18)


sns.barplot(data=temp, x='Cabin', y='Fare', hue='Pclass', ax=ax[1, 1], palette=palette_5)
ax[1, 1].set_title('Deck Fare', x=0.1, y=1.02, size=25)
ax[1, 1].set_xlabel('Deck', size=18)

sns.despine()
plt.show()


# The feature looks more balanced now and we can see that now we can say that deck 'M' is the deck where most of the pclass 2 and 3 are present whereas deck ABC is where most pclass 1 are present. We can see from the plots that deck DE had the highest survival rate!!! and deck 'M' has the lowest survival rate. 
# 
# The great thing about this simplification from 'Cabin' to 'Deck' is that we have cornered the Pclass 2 and 3 passengers which had the lowest survival rates in new deck M we created!! in all the other decks it seems that they had better chance of survival!! Compare the **Deck-Pclass Proportions** and the **Deck Survival rate**, you can clearly see that those few Pclass 2 and 3 passenger in deck DE and FG have good survival chance, most of the Pclass 2 and 3 passengers are in the Deck 'M' and their survival rate is not good!!
# 
# This feature helps to harness this pattern!!!

# Lets impute the missing values as per the above understandings...

# In[14]:


def imputer(df):
    
    # imputing missing age values
    
    age_impute_series = df.groupby(['Pclass', 'Sex']).Age.transform('mean')
    df.Age.fillna(age_impute_series, inplace=True)
    
    # imputing Cabin missing value.
    df.Cabin = df.Cabin.str.extract(pat='([A-Z])')
    
    df.Cabin.fillna('M', inplace=True)
    
    df['Deck'] = df.Cabin.replace({'A':'ABC', 'B':'ABC', 'C':'ABC', 'D':'DE', 'E':'DE', 'F':'FG', 
                                   'G':'FG', 'T':'ABC'}) # we will drop the Cabin here and only take Deck from here on with us!!
    
    df.drop('Cabin', axis=1, inplace=True)
    
    # lets just finally fill all the left over missig value with the mode of the feature.
    for feature in df.columns:
        df[feature].fillna(df[feature].mode()[0], inplace=True)
        
    return df


# In[15]:


train_imputed = imputer(train.copy())
test_imputed = imputer(test.copy())


# ## Feature creation through EDA
# 
# 
# ### Target Distribution

# This is an important EDA step, to explore the distribution of values of feature with the Target, this EDA step provides us with good understanding of the features and helps us uncover the possibility of feature simplification, combinination etc.

# In[16]:


cat_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Deck', 'Embarked']

plt.figure(figsize=(16, 14))
sns.set(font_scale= 1.2)
plt.style.use('fivethirtyeight')

for i, feature in enumerate(cat_features):
    plt.subplot(2, 3, i+1)
    sns.countplot(data=train_imputed, x=feature, hue='Survived', palette=palette_6)  
    
sns.despine()


# Everything looks great, there are trends which all the categories in categorical feature follow with target. We can observe that it will be beneficial to combine the SibSp and Parch into a common feature named Family_size and further simplification into category such as 'Alone', 'Small_family', 'large_family' can help us capture the trend better, which is that for small family the survival rate is better, a person is more likely to survive if they had 2 to 4 Family size(Parch + SibSp)!!

# In[17]:


num_features = ['Fare', 'Age']
plt.style.use('fivethirtyeight')

plt.figure(figsize=(16, 14))
for i, feature in enumerate(num_features):
    plt.subplot(2, 2, i+1)
    plt.hist(x=[train_imputed[feature][train_imputed['Survived'] == 1], train_imputed[feature][train_imputed['Survived']==0]],
            stacked=True, label=['Survived', 'Not Survived'], bins=20, color=['orange', 'b'])
    plt.legend()
    plt.xlabel(f'{feature}', fontsize=15)
    plt.ylabel('Count', fontsize=15)

for i, feature in enumerate(num_features):
    plt.subplot(2, 2, i+3)
    sns.kdeplot(data=train_imputed, x=feature, hue='Survived', fill=True, palette=palette_binary_3)
    
sns.despine()


# From the Fare plot we can see that as the fare increases the survival likelihood also increases, we might be able to simplify the Fare feature into bins and capture this trend in a better way, it might or might not be helpful to do so, so we will have to cross check this decision by **grid searching this through a pipeline**!! 
# 
# Age feature also shows possibility of simplification as we can see that at different age brackets the ratio of survived likelihood to not survived likelihood is varying alot... Lets bin them and see all this in action.

# ### Binning the Fare feature

# In[18]:


temp = train_imputed.copy()
temp['Fare'] = pd.qcut(temp.Fare, 7)
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15, 8))
sns.countplot(data=temp, x='Fare', hue='Survived', palette=palette_binary_2)

plt.ylabel('Count', fontsize=18, labelpad=5)
plt.xlabel('Fare', fontsize=18, labelpad=10)
plt.tick_params(axis='x', labelsize=13)

plt.title('Fare Categories Survival Count plot', y=1.03, fontsize=25, loc='Left')
sns.despine()
plt.show()


# Yup!! looks great, this is the trend we observed in the plot in last section. Binning helped us express this relation better or we can say that we are able it visualize this feature better as humans, but maybe the continuous form is better as per the algorithm... But atleast we know what we wish to check out by asking this question to our model that what it likes and what it does not(*More on this in Data set creation section*)!! 
# 
# If we wish to use the feature as above, we will have to perform Ordinal encoding, where we encode the higher fare category with higher numeric value, for eg:- (-0.001, 7.55] -> 1, (7.55, 7.854] -> 2, (7.854, 8.05] -> 3 and so on. 
# 
# **Warning**:-
# 
# We can use LabelEncoder from scikit learn api to perform ordinal encoding only because there is order already present in our category which can be correctly understood by the LabelEncoder. But if the order present cannot be understood by the LabelEncoder for eg: Freezing, Cold, Warm, Hot, Lava Hot will be encoded as 2, 1, 5, 3, 4 on the basis of alphabetical order, which is not the correct order for this ordinal feature, it should be 1, 2, 3, 4, 5!! 
# 
# **Always beware of this trap!!!**

# ### Binning age feature

# In[19]:


temp = train_imputed.copy()
temp['Age_cat'] = pd.cut(temp.Age, bins=[0, 5, 24, 30, 36, np.inf])
temp['Age_qcat'] = pd.qcut(temp.Age, 15, precision=2)

fig, ax = plt.subplots(2, 1, figsize=(15, 14))
plt.style.use('fivethirtyeight')

sns.countplot(data=temp, x='Age_qcat', hue='Survived', ax=ax[0], palette=palette_binary_3)
ax[0].tick_params(axis='x', labelsize=9)

sns.countplot(data=temp, x='Age_cat', hue='Survived', ax=ax[1], palette=palette_binary_3)

sns.despine()
plt.show()


# from the second plot we can see that (0.0, 5.0] is a great age bracket for survival, whereas (24.0, 34.0] is the worst, for the (30.0, 36.0] category chances seems to 50-50, i have also plotted a qcut plot with 15 quantiles, just to show that how i visualized the age distributions in the second plot, for the second plot i wanted to have categories with some significant info gain, the above plot shows some good survival rate distribution for the categories we have selected after we went through the first plot.... Just a small trick to bin features to get better info gain for beginners to keep in mind.. 
# 
# **After testing out**:
# 
# The Age_cat feature improved the results for XGBClassifier and the GradientBoostingCLassifier significantly, observed 1% improvement from continuous age feature to Age_cat!!!

# For the titanc dataset many have observed that binning the continuous features helped improve the results slightly, as you can see that binning can let you express the survival chances a little better and in a simpler way, So always test this out with other datasets you work with and test out whether it helps or not!!

# ### Ticket based features

# Ticket feature contains tickets with few of them with alphabets and some of them with only numbers, we could extract this relation but it might not be useful as it may be the case that this relation is related to the Embarked feature and depending upon where you Embarked you might have ticket code in only numbers or with some alphabets... rather we can also extract the frequency of tickets, there are people with same ticket numbers which shows that group of friends and family travelled together and with same ticket number.
# 
# first lets check out if just dividing ticket into ticket with alphabets and only numeric serial code helps or not... Then we will perform even finer ticket categorization.

# In[20]:


temp = train_imputed.copy()
temp['Ticket_type'] = pd.Series(np.where(temp.Ticket.str.contains(pat='[A-Z]'), 'alphabet', 'numeric'), index=temp.index)
temp['Fare_cat'] = pd.qcut(temp.Fare, 7, precision=2)

y=1.01
size=17
fig, ax = plt.subplots(3, 2, figsize=(16, 19))
plt.subplots_adjust(hspace=0.3)

sns.set(font_scale=1)
plt.style.use('fivethirtyeight')

sns.countplot(data=temp, x='Embarked', hue='Ticket_type', ax=ax[0, 0], palette=palette_3)
ax[0, 0].set_title('Ticket Type V/s Embarked', size=size, loc='Left', y=y)


sns.countplot(data=temp, x='Pclass', hue='Ticket_type', ax=ax[0, 1], palette=palette_3)
ax[0, 1].set_title('Pclass V/s Ticket Type', size=size, loc='Left', y=y)

sns.countplot(data=temp, x='Embarked', hue='Pclass', ax=ax[1, 0], palette=palette_3)
ax[1, 0].set_title('Pclass V/s Embarked', size=size, loc='Left', y=y)

sns.countplot(data=temp, x='Fare_cat', hue='Ticket_type', ax=ax[1, 1], palette=palette_3)
ax[1, 1].set_title('Fare Category V/s Ticket type', size=size, loc='Left', y=y)
ax[1, 1].tick_params(axis='x', labelsize=9.5)

sns.countplot(data=temp, x='Ticket_type', hue='Survived', ax=ax[2, 0], palette=palette_3)
ax[2, 0].set_title('Ticket_type - Survived Plot', size=size, loc='Left', y=y)

sns.countplot(data=temp, x='Deck', hue='Ticket_type', ax=ax[2, 1], palette=palette_3)
ax[2, 1].set_title('Ticket_type - Deck Plot', size=size, loc='Left', y=y)

sns.despine()
plt.show()


# As seen from above, no general trend is visible by categorizing ticket with or without alphabets. There are few points to note though: 
# * Mostly numeric tickets were sold at the 'Q' which is Queenstown.
# * Proportion of ticket with alphabets bought is less for Pclass 3 as seen from 'Pclass V/s Ticket type plot'.
# 
# 
# Ticket type does not show any patterns in the Fare V/s Ticket type plot..... it was kind of disappointing :(
# 
# Overall there is no pattern in Ticket type V/s Fare, Deck, Pclass.... This feature does not look that good.
# 
# 
# Finally, it may become useful... again.. you already know by now :D...we will have to test it out in Dataset creation section using grid search.
# 
# 
# Now lets extract even finer ticket types, maybe model finds some use out of it... we should check it out!!

# In[21]:


def ticket_extractor(ticket):
    alpha = re.sub('\d', '', ticket)
    if alpha:
        return alpha
    else:
        num = re.search('\d{1,9}', ticket)
        return ticket
    
temp = train_imputed.copy()
temp['Ticket_extracted'] = temp.Ticket.apply(ticket_extractor)
for i in range(len(temp.Ticket)):
    try:
        int(temp.Ticket_extracted.iloc[i])
        temp.Ticket_extracted.iloc[i] = f'Num_{len(temp.Ticket_extracted.iloc[i])}'
    except:
        continue
        
for label, pattern in [('_ca', 'C[.]?A[.]?'), ('_PC', 'PC'), ('_SOTON', 'SOTON'), ('_STON', 'STON'), 
                       ('_WC', 'W[.]?[/]?C'), ('_SC', 'S[.]?C[.]?'), ('_A', 'A[.]?'), ('_SOC', 'S[.]?O[.]?[/]?C'), 
                       ('_PP', 'PP'), ('_FC', '(F.C.|F.C.C.)'), ('_LS_number', 'Num_(6|7)'), ('_SS_number', 'Num_(3|4|5)'), 
                       ('rare', '^[^_]')]:
    temp.Ticket_extracted[temp.Ticket_extracted.str.contains(pattern)] = label
    
temp['Ticket_extracted'].value_counts(dropna=False)

fig, ax = plt.subplots(2, 1, figsize=(15, 15))
plt.style.use('fivethirtyeight')

sns.countplot(data=temp, x='Ticket_extracted', hue='Survived', ax=ax[0], palette=palette_6)
ax[0].set_title('Extracted Ticket - Survival "count" plot', size=20, loc='Left', y=1.03)

sns.barplot(data=temp, x='Ticket_extracted', y='Survived', ax=ax[1], palette=palette_6)
ax[1].set_title('Extracted Ticket - Survival "chance" plot', size=20, loc='Left', y=1.03)

sns.despine()
plt.show()


# Looks decent, what attracts my attention is the ticket A, PC, CA, Large_Serial_number and Small_Serial_number as they are quite a few of these tickets to be confident in the survival chances shown by these tickets. This feature might help us improve the probability of survival of few passengers and may push there probability of survival over 50% if they actually survived, to get classified as survived, which is actually the problem with many instances as they get classified as not survived, i.e probability of survival is lower than 50%.

# In[22]:


temp['Ticket_frequency'] =temp.Ticket.map(temp.Ticket.value_counts(dropna=False))
temp['Fare_cat'] = pd.qcut(temp.Fare, 7, precision=2)

y=1.03
size=25
fig, ax = plt.subplots(2, 1, figsize=(15, 15))
sns.set(font_scale=1.4)
plt.style.use('fivethirtyeight')

sns.countplot(data=temp, x='Ticket_frequency', hue='Survived', ax=ax[0], palette=palette_2)
ax[0].set_title('Ticket frequency - Survived Plot', loc='Left', y=y, size=size)
ax[0].set_xlabel('')
ax[0].set_ylabel('')

sns.countplot(data=temp, x='Fare_cat', hue='Ticket_frequency', ax=ax[1], palette=palette_2)
ax[1].set_title('Ticket frequency - Fare category Plot', loc='Left', y=y, size=size)
ax[1].set_xlabel('')
ax[1].set_ylabel('')

sns.despine()
plt.show()


# **Findings:**
# * We can clearly see from the above plot that the survival chance for people with ticket frequency from 2-4 was more.
# * We were also able to uncover new information, it seems that higher frequency mean higher ticket fare which is actually the cummulative ticket price for all passenger, which means that Fare column does not contain the fare for an individual person but the total fare for a group with same ticket!! 
# * we might be able to derive a new feature which is fare_per_individual, lets check it out if it is good or not!!

# In[23]:


temp = train_imputed.copy()
temp['Ticket_frequency'] =temp.Ticket.map(temp.Ticket.value_counts(dropna=False))
temp['Fare_cat'] = pd.qcut(temp.Fare, 7, precision=2)
temp['Fare_per_individual'] = temp['Fare'] / temp['Ticket_frequency']
temp['Fare_per_individual_cat'] = pd.qcut(temp.Fare_per_individual, 7, precision=2)

y = 1.03
size = 18
sns.set(font_scale=1.2)
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

sns.kdeplot(data=temp, x='Fare_per_individual', hue='Survived', clip=[0, 200], fill=True, ax=ax[0], palette=palette_binary_3)

sns.kdeplot(data=temp, x='Fare', hue='Survived', clip=[0, 200],  fill=True, ax=ax[1], palette=palette_binary_3)
ax[1].set_ylabel('')
sns.despine()

fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

sns.countplot(data=temp, x='Fare_per_individual_cat', hue='Survived', ax=ax[0], palette=palette_binary_3)
ax[0].tick_params(axis='x', labelsize=9.5)
#ax[0].set_ylabel('')
#ax[0].set_xlabel('')

sns.countplot(data=temp, x='Fare_cat', hue='Survived', ax=ax[1], palette=palette_binary_3)
ax[1].tick_params(axis='x', labelsize=9.5)
ax[1].set_ylabel('')
#ax[1].set_xlabel('')

sns.despine()
plt.show()


# **Findings :**
# * From the above plot we can see that the probability distribution function did not show much difference, for the fare_per_individual it just got squeezed to the left and the corresponding values in the y axis were scaled equally...
# * But when we tried out making fare_per_individual category, we can see that fare_per_individual category gave us a better steady increase in survival rate from left to righ!!! 
# 
# Until now we have Pointed out a lot of possibilities for feature engineering and there so many tough decisions to make as per what to keep and what to drop, we only want the best, informative features to achieve best results, But there is no shortcut to this, we cannot just simply know what works best with the given data and our models.... We will have to test it all out!! Though the Feature Engineering class will help us to quickly narrow down our search for the best data sets by quickly dropping out dataset made by selecting features which are not good. 

# ### Family Size Feature

# In the data provided to us, each individual has a SibSp and Parch, so if we add these features together for an individual and add 1 to it, 1 being the individual themself, we can create a new feature Family_size.
# 
# It is an important feature as it reveals that passengers with family size 2 - 4 had a better survival rate than passengers travelling alone or who had larger families.

# In[24]:


temp = train_imputed.copy()
temp['Family_size'] = temp['SibSp'] + temp['Parch'] + 1
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(1, 1, figsize=(15, 8))
sns.countplot(data=temp, x='Family_size', hue='Survived', ax=ax, palette=palette_5)
ax.set_title('Family Size - Survived Plot', size=25, loc='Left', y=1.04)

sns.despine()
plt.show()


# Looks great!! the survival pattern is quite clear... but can we simplify and combine the family size 2-4 as 'small family' and family size of 1 as 'alone' to simplify the feature?

# In[25]:


temp['Family_size_cat'] = temp['Family_size'].replace({1:'alone', 2:'small_family', 3:'small_family', 4:'small_family'
                                                      ,5:'large_family', 6:'large_family', 7:'large_family'
                                                      ,8:'large_family', 9:'large_family', 10:'large_family', 
                                                       11:'large_family'})

fig, ax = plt.subplots(1, 1, figsize=(15, 8))
plt.style.use('fivethirtyeight')

sns.countplot(data=temp, x='Family_size_cat', hue='Survived', ax=ax, palette=palette_binary_2)
ax.set_title('Family Category - Survived Plot', size=25, loc='Left', y=1.04)

sns.despine()
plt.show()


# Yup, things look simpler and neat!! But again, it looks good to us but model might not find it as helpful as we think, maybe it could have found patterns with those family size numbers and some other feature, which it would not be able to harness as easily now, you never know... what we can be sure about is that we have made the feature simpler, if we one hot encode it we will have fewer dimensions, which is a good thing. Lets say we had many features which had to be one hot encoded but there cardinality(number of different categories in categorical feature) was high, label encoding was not an option and if we one hot encoded the feature we would be struck by curse of dimensionality, in such case if all the categorical features could be simplified in such a easy, beautiful way, nothing like that!!
# 
# But we do not have that problem at present, we have a small dataset and not aot of features with high cardinality, so we will have to figure out whether this feature helps or not. 

# ### Name Length Feature

# One of the features that can be derived from the Name feature is the Name length, the idea behind this is that people with higher status usually had longer names, we can use this fact to create a feature with name lengths.

# In[26]:


temp = train_imputed.copy()
temp['Name_length'] = temp.Name.str.replace(pat='[^a-zA-Z]', repl='').str.len()
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
sns.histplot(data=temp, x='Name_length', hue='Survived', kde=True, fill=True, ax=ax, palette=palette_binary_3)
ax.set_title('Name Length - Survived Plot', size=20, loc='Left', y=1.03)

sns.despine()
plt.show()


# Great!! we can see that with increase in the name length the survival rate increases!! from about name_length 26 more pople with name length more than 26 survived than not survived.

# ### Name Title

# In[27]:


temp = train_imputed.copy()

temp['Title'] = temp.Name.str.extract(pat='([a-zA-Z]+\.)')

temp.Title[~temp.Title.isin(['Mr.', 'Miss.', 'Mrs.', 'Master.'])] = 'rare'


# In[28]:


fig, ax = plt.subplots(1, 1, figsize=(15, 8))
plt.style.use('fivethirtyeight')

sns.countplot(data=temp, x='Title', hue='Survived', ax=ax, palette=palette_6)
ax.set_title('Title - Survived Plot', loc='Left', size=25, y=1.03)

sns.despine()
plt.show()


# The above plot shows that Mrs., Miss. and Master. titles had better chances if survival, chances of survival is less than that for survival in case of Mr. and rare title, but it is worse for the Mr. title.

# ### Survival Rate

# ***This is the most important feature that almost all the top leaderboard submissions will have.***
# 
# 
# Intuition behind the Survival rate feature:
# 
# It was observed that usually people belonging to a family and Ticket group survived together or died together, this can help us assign survival rate for instances in test set which belonged to a family or ticket group in training set, this can help us make it easier for the model predict few more instances and boost the score... All this made more clear by the visualiations below!!
# 

# In[29]:


temp = train_imputed.copy()
temp['Family_name'] = temp.Name.str.split(',', n=1, expand=True).iloc[:, 0]
temp['Family_size'] = temp['SibSp'] + temp['Parch'] + 1
temp['Ticket_frequency'] = temp.Ticket.map(temp.Ticket.value_counts()) 

temp['Family_survival_rate'] = temp.Family_name.map(temp.groupby(['Family_name']).Survived.median())
temp['Ticket_group_survival_rate'] = temp.Ticket.map(temp.groupby(['Ticket']).Survived.median())


fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
plt.style.use('fivethirtyeight')

sns.countplot(data=temp[temp['Family_size'] > 1], x='Family_survival_rate', hue='Survived', ax=ax[0], palette=palette_5)
# we will only take into account those instance with Family size > 1 for visualization to explore the effectiveness of the 
# Survival rate feature

sns.countplot(data=temp[temp['Ticket_frequency'] > 1], x='Ticket_group_survival_rate', hue='Survived', ax=ax[1], palette=palette_5)
# we will only take into account those instance with Ticket_frequency > 1 for visualization to explore the effectiveness of the 
# Survival rate feature

sns.despine()
plt.show()


# Well, things are crystal clear now!! those passengers which were assigned survival rate 0 because the median survival of the group was 0 actually only very few of them survived!! same the case when we see the passengers with median family survival rate 1, majority survived!!
# 
# 
# ***Now how does this feature help us get better results??***
# 
# First we have to understand that getting an accuracy score of 80% and getting to top 4% of the leaderboard is possible just by tunning model hyper parameters well and performing good feature encoding practices, but moving up from there is very difficult unless you have feature engineered very good features such as the one we are creating right now... The difference of 80% to 84% accuracy is just of about 2-4 more correct survival predictions. i got up to 79 % with just stacking and good encoding practice, [check it out](https://www.kaggle.com/awwalmalhi/comprehensive-beginners-guide-to-top-6)
# 
# Lets say we have few family members or ticket group members which were not in training set but in test set, we can easily, with confidence assign a survival rate to them in the test set depending upon the above results and turns out that ususally people are able to push from 75% to 81-83% with this feature!! just by predicting few more stubborn instances as 1!!

# ### EDA conclusion

# ***Missing value imputation***
# * imputing Age feature
# * imputing Embarked feature
# 
# We were able to find better ways to impute the Age and Embarked feature with reasonable support from the present data explored via Visualizations, you should too explore the data through EDA and find out best ways to impute data rather straight away using sklearn.impute or .fillna()
# 
# 
# ***Feature creation through EDA***
# 
# we were able to explore feature simplification, combination possibilities and possibilities to derive new features with proper support for these decisions from the present data...
#  * Bin Fare feature to simplify and express patterns
#  * Bin Age feature to simplify and express patterns
#  * Better way to extract information from Ticket feature and creation of Ticket_frequency feature and Ticket_extracted feature
#  * through the Ticket frequency feature we were able to gain information regarding Fare_per_individual feature and possibility of binning it.
#  * Family size feature and possibility of simplification
#  * Name length feature
#  * Survival rate feature
#  
# until now we have created few features and explored possibilities for simplification, we have many possible ways to create the final dataset. There are a lot of decision we will have to make now, some of them are as follows:
# * Is binned Fare feature better or the continuous one?
# * is Binned Age feature better of the continuous one?
# * Should we keep the SibSp and Parch rather than the Combined Family size feature?
# * If we should keep the Family_size feature, should we simplify it?
# * if we simplify the Family_size feature, should we one hot encode it or Target encode it?
# * What features should we drop?
# * Should we use StandardScaler, MinMaxScaler or RobustScaler?
# 
# 
# These are some of good questions that come to my mind... There can be many more, but lets keep things simple or the class will become difficult to debug XD
#  
#  
#  
# 

# # Feature Engineering

# In[30]:


def feature_creator(df_train, df_test):
    
    # creating binned fare feature
    df_train['Fare_cat'] = pd.qcut(df_train['Fare'], 7)
    df_test['Fare_cat'] = pd.qcut(df_test['Fare'], 7)
    
    df_train['Fare_cat'] = LabelEncoder().fit_transform(df_train['Fare_cat'])
    df_test['Fare_cat'] = LabelEncoder().fit_transform(df_test['Fare_cat'])
    
    
    
    # creating binned age feature
    df_train['Age_cat'] = pd.cut(df_train.Age, bins=[0, 5, 24, 30, 36, np.inf])
    df_test['Age_cat'] = pd.cut(df_test.Age, bins=[0, 5, 24, 30, 36, np.inf])
    
    df_train['Age_cat'] = LabelEncoder().fit_transform(df_train['Age_cat'])
    df_test['Age_cat'] = LabelEncoder().fit_transform(df_test['Age_cat'])
    
    
    
    # Title feature
    for df in [df_train, df_test]:
        df['Title'] = df.Name.str.extract(pat='([a-zA-Z]+\.)')
        df.Title[~df.Title.isin(['Mr.', 'Miss.', 'Mrs.', 'Master.'])] = 'rare'
        
        

    # creating Ticket frequency feature
    df_full = pd.concat([df_train, df_test], axis=0)
    df_full['Ticket_frequency'] = df_full.Ticket.map(df_full.Ticket.value_counts())
    df_train = df_full.loc[df_train.index]
    df_test = df_full.loc[df_test.index].loc[df_test.index].drop('Survived', axis=1)
    
    
    
    # creating extracted ticket feature
    for df in [df_train, df_test]:
        
        df['Ticket_extracted'] = df.Ticket.apply(ticket_extractor)
        for i in range(len(df.Ticket)):
            try:
                int(df.Ticket_extracted.iloc[i])
                df.Ticket_extracted.iloc[i] = f'Num_{len(df.Ticket_extracted.iloc[i])}'
            except:
                continue

        for label, pattern in [('_ca', 'C[.]?A[.]?'), ('_PC', 'PC'), ('_SOTON', 'SOTON'), ('_STON', 'STON'), 
                               ('_WC', 'W[.]?[/]?C'), ('_SC', 'S[.]?C[.]?'), ('_A', 'A[.]?'), ('_SOC', 'S[.]?O[.]?[/]?C'), 
                               ('_PP', 'PP'), ('_FC', '(F.C.|F.C.C.)'), ('_LS_number', 'Num_(6|7)'), 
                               ('_SS_number', 'Num_(3|4|5)'), ('rare', '^[^_]')]:
            df.Ticket_extracted[df.Ticket_extracted.str.contains(pattern)] = label
            
            
    
    # Fare per individual feature
    #df_train['Fare_per_individual'] = df_train['Fare'] / df_train['Ticket_frequency']
    #df_test['Fare_per_individual'] = df_test['Fare'] / df_test['Ticket_frequency']
    
    #df_train['Fare_per_individual_cat'] = pd.qcut(df_train['Fare_per_individual'], 7)
    #df_test['Fare_per_individual_cat'] = pd.qcut(df_test['Fare_per_individual'], 7)
    
    #df_train['Fare_per_individual_cat'] = LabelEncoder().fit_transform(df_train['Fare_per_individual_cat'])
    #df_test['Fare_per_individual_cat'] = LabelEncoder().fit_transform(df_test['Fare_per_individual_cat'])
    
    
    
    # Family size feature
    df_train['Family_size'] = df_train['SibSp'] + df_train['Parch'] + 1
    df_test['Family_size'] = df_test['SibSp'] + df_test['Parch'] + 1
    
    
    
    # creating binned family sized feature
    df_train['Family_size_cat'] = df_train.Family_size.replace({1:'alone', 2:'small_family', 3:'small_family', 4:'small_family'
                                                               ,5:'large_family', 6:'large_family', 7:'large_family'
                                                               ,8:'large_family', 9:'large_family', 10:'large_family'
                                                               ,11:'large_family'})
    
    df_test['Family_size_cat'] = df_test.Family_size.replace({1:'alone', 2:'small_family', 3:'small_family', 4:'small_family'
                                                               ,5:'large_family', 6:'large_family', 7:'large_family'
                                                               ,8:'large_family', 9:'large_family', 10:'large_family' 
                                                               ,11:'large_family'})
    
    
    
    # Name length feature
    df_train['Name_length'] = df_train.Name.str.replace(pat='[^a-zA-Z]', repl='').str.len()
    df_test['Name_length'] = df_test.Name.str.replace(pat='[^a-zA-Z]', repl='').str.len()
    
    
    
    # survival rate feature
    df_full = pd.concat([df_train, df_test], axis=0).reset_index()

    df_full['Family_name'] = df_full.Name.str.split(',', n=1, expand=True).iloc[:, 0]
    
    df_full['Family_Survival'] = 0.5
    
    # the following Survival rate code has been copied from https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83
    for grp, grp_df in df_full.groupby(['Family_name', 'Fare']):
        if (len(grp_df) != 1):
            # Fare and Family_name are used as category to group families together, as these features should be same for
            # families and once we get the groups we filter for grp_df which have ore than 1 length, hence a family.
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 0

    for _, grp_df in df_full.groupby('Ticket'):
        if (len(grp_df) != 1):
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if (smax == 1.0):
                        df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif (smin==0.0):
                        df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 0
     
    df_full.set_index('PassengerId', inplace=True)
    df_train = df_full[:891]
    df_test = df_full[891:]
    
    
    # finally we will drop the family name feature we created to help us create the Survival rate feature.
    df_train.drop(['Family_name'], axis=1, inplace=True)
    df_test.drop(['Family_name', 'Survived'], axis=1, inplace=True)
    
    # lets finally drop the survived from our training set and create the target variable
    y_train =df_train.Survived
    df_train.drop('Survived', axis=1, inplace=True)
    
    return df_train, y_train, df_test


# In[31]:


X_train, y_train, X_test = feature_creator(train_imputed.copy(), test_imputed.copy())
y_train = y_train.astype(int)


# In[32]:


X_train.head(5)


# In[33]:


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
plt.style.use('fivethirtyeight')

size = 20
y = 1.03
X_train['Family_Survival'] = np.round(X_train.Family_Survival, 2)

sns.countplot(data=pd.concat([X_train, y_train], axis=1), x='Family_Survival', hue='Survived', ax=ax[0], palette=palette_2)
ax[0].set_title('Survival rate in training set', loc='Left', size=size, y=y)


X_test.Family_Survival = np.round(X_test.Family_Survival, 2)
sns.countplot(data=X_test, x='Family_Survival', ax=ax[1], palette=palette_2)
ax[1].set_title('Survival rate - Count plot test set', loc='Left', size=size, y=y)

sns.despine()
plt.show()


# What is of importance to us is the left plot where with certain confidence we have assigned the survival rates to the instances and we have about 175 instances for which we have some certainity of final outcome.

# ### Dataset creation

# **Now we are ready to ask questions!!!** Lets create FeatureEngineering class and use it in a pipeline with Logistic Regression(we can use other estimators too, but i prefer to use a simple one to keep computational expense low and search fast), we will pass the questions we wish to ask as the parameter to this class in the pipeline and grid search the whole feature space to narrow down our search for the best dataset!!
# 
# * Is binned Fare feature better or the continuous one?
# * is Binned Age feature better of the continuous one?
# * Should we keep the SibSp and Parch rather than the Combined Family size feature?
# * If we should keep the Family_size feature, should we simplify it?
# * if we simplify the Family_size feature, should we one hot encode it or Target encode it?
# * What features should we drop?
# * Should we use StandardScaler, MinMaxScaler or RobustScaler?
# 
# Before we get to the above Q/A setion, lets make few transformations i have noted below.
# * Pclass and Sex to one hot encodings
# * Drop Name and Ticket

# In[34]:


# if we wish to create a transformer which we can grid search in a pipeline, that transformer has to inherit from the 
# BaseEstimator class, we will also make our FeatureEngineering transformer inherit from the TransformerMixin which just 
# provides the functionality of fit_transform() method to our transformer.

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, bin_fare=False, bin_age=True, family_size=True, bin_family_size=True,
                 drop_Name_length=False, drop_Ticket_frequency=False, drop_all=True, drop_Family_Survival=True, 
                 drop_Ticket_extracted=False, scaling='StandardScaler', target_encode_title=True, test=False):
        self.bin_fare = bin_fare
        self.bin_age = bin_age
        #self.bin_fare_individual_cat = bin_fare_individual_cat
        self.family_size = family_size
        self.bin_family_size = bin_family_size
        self.drop_Name_length = drop_Name_length
        self.drop_Ticket_frequency = drop_Ticket_frequency
        self.drop_all = drop_all
        self.drop_Ticket_extracted = drop_Ticket_extracted
        self.scaling = scaling
        self.drop_Family_Survival = drop_Family_Survival
        self.target_encode_title = target_encode_title
        self.transformer = None
        self.drop_list = []
        self.test = test
        
    def fit(self, X, y=None):
        return self # the fit method usually return self only, if not sure, try fitting a scikit learn estimator/transformer.. 
                    # only the fitted object will be returned, fit() will include code in case of these transformers and
                    # estimators but we do not need to have some fitted variables on our dataset before we go for transform,
                    # or atleast the way i am doing it i don't need to. :) 
    
    def transform(self, X, y=None):
        
        X['Pclass'] = X['Pclass'].astype(str) # import convert numeric values to str if we wish to use get_dummies on them.
        dummies = pd.get_dummies(X.loc[:, ['Pclass', 'Sex']], drop_first=True)
        X = pd.concat([X, dummies], axis=1)
        X.drop(['Pclass', 'Sex'], axis=1, inplace=True)
        
        # lets also drop Name and Ticket feature as we have extracted what we needed!!
        X.drop(['Name', 'Ticket'], axis=1, inplace=True)
        
        
        
        
        if self.bin_fare:
            # so if we self.bin_fare = True we want the binned fare, so drop the continuous fare feature.
            self.drop_list.append('Fare')
            
        else:
            self.drop_list.append('Fare_cat')
            
        
        if self.bin_age:
            self.drop_list.append('Age')
        
        else:
            self.drop_list.append('Age_cat')
            
            
        #if self.bin_fare_individual_cat:
            #self.drop_list.append('Fare_per_individual')
            
        #else:
            #self.drop_list.append('Fare_per_individual_cat')
                
                
                
                
        if self.family_size:
            self.drop_list.extend(['SibSp', 'Parch'])
            
            if self.bin_family_size:
                self.drop_list.append('Family_size')
                
            else:
                self.drop_list.append('Family_size_cat')
                
        else:
            self.drop_list.extend(['Family_size', 'Family_size_cat'])
            
        if self.drop_Name_length:
            X.drop('Name_length', axis=1, inplace=True)
            
        if self.drop_Ticket_frequency:
            X.drop('Ticket_frequency', axis=1, inplace=True)
            
        if self.drop_Family_Survival:
            X.drop('Family_Survival', axis=1, inplace=True)
            
        if self.drop_all:
            X.drop(self.drop_list, axis=1, inplace=True)
            
       
        self.cols = ['Title', 'Ticket_extracted'] if self.target_encode_title else ['Ticket_extracted']
        
        if not self.test and len(X) > 400:
            # we want the encoder object to be fitted with traing set only so we will check 'not self.test' and we also do not
            # want to program to pass through the if statement in the case of cross validation test is being considered, 
            # hence we also check len(X) > 400 assuming the cross val test size will always stay less than 400.
            self.target_encoder = TargetEncoder(cols=self.cols, smoothing=5)
            X.loc[:, self.cols] = self.target_encoder.fit_transform(X.loc[:, self.cols], y_train.loc[X.index])

        else:
            X.loc[:, self.cols] = self.target_encoder.transform(X.loc[:, self.cols])
                
                
        if self.drop_Ticket_extracted:
            X.drop('Ticket_extracted', axis=1, inplace=True)
        
        # finally lets convert all remaining categorical features into on hot features
        X = pd.get_dummies(X, drop_first=True)  
     
        # only features with no. of unique values as 2, will one hot encoded features and we do not want to apply Normalization
        # on them(reason explained in a bit)
        features_to_scale = [feature for feature in X.columns if X[feature].nunique() != 2]  
        if not self.test and len(X) > 400:    
            if self.scaling == 'StandardScaler':
                self.transformer = StandardScaler()
                X.loc[:, features_to_scale] = self.transformer.fit_transform(X.loc[:, features_to_scale])
                
            elif self.scaling == 'RobustScaler':
                self.transformer = RobustScaler()
                X.loc[:, features_to_scale] = self.transformer.fit_transform(X.loc[:, features_to_scale])
                
            elif self.scaling == 'MinMaxScaler':
                self.transformer = MinMaxScaler()
                X.loc[:, features_to_scale] = self.transformer.fit_transform(X.loc[:, features_to_scale])
        
        else:
            X.loc[:, features_to_scale] = self.transformer.transform(X.loc[:, features_to_scale])
            
        return X


# You can see that we are only scaling the features which are not like the one hot encoded columns or in general which do not have zeros as most of the column entries, this is because if we were to pass the one hot encoded columns through the a StandardScaler or RobustScaler, all the zeros will get converted to some other value when scaled and the one hot encoding will be wrecked.. MinMaxScaler will retain it only if the min value is 0, We can understand this property of scaling transformers when we look at there functions, StandardScaler subtracts mean of all values by the value being scaled in numerator, but MinMaxScaler subtracts the min value by the value being scaled, hence if we are scaling 0 and 0 is the min value we retain 0, But that wont be the case for the StandardScaler.... So look out for this detail.

# In[35]:


def results(grid, n=10):
    for index, row in pd.DataFrame(grid.cv_results_)[['params', 
                                                      'mean_test_score']].nlargest(n=n, columns='mean_test_score').iterrows():
        print(f'{row[0]} : {row[1]}')


# In[36]:


param_grid_pipeline = {'feature_engineering__bin_fare':[True, False],
                       'feature_engineering__bin_age':[True, False],
                       #'feature_engineering__bin_fare_individual_cat':[True, False],
                       'feature_engineering__family_size':[True, False],
                       'feature_engineering__bin_family_size':[True, False],
                       'feature_engineering__drop_Name_length':[False],
                       'feature_engineering__drop_Ticket_frequency':[False],
                       'feature_engineering__drop_all':[True],
                       'feature_engineering__drop_Ticket_extracted':[True, False],
                       'feature_engineering__target_encode_title':[True, False],
                       'feature_engineering__scaling':['StandardScaler']} 



pipeline_ = Pipeline([('feature_engineering', FeatureEngineering()),
                 ('model', LogisticRegression())])

cv = StratifiedShuffleSplit(n_splits=20, test_size=0.15, random_state=101)

grid = GridSearchCV(pipeline_, param_grid_pipeline, 
                    cv=cv.split(X_train, X_train.Deck),
                    scoring='accuracy', verbose=2, n_jobs=-1)

grid.fit(X_train.copy(), y_train)


# In[37]:


pd.DataFrame(grid.cv_results_)['mean_test_score'].isna().sum() # check if error occured while using a feature combination or not.


# In[38]:


results(grid, n=60)


# **Findings:**
# * Looking at the above results, model loved the family size feature and specially when binned!!
# * bin fare feature was also liked, though it gets a little confused between binning and not.
# * it likes the Name length too.
# * Ticket_frequency was liked too.
# * I also let the model decide whether it wishes to work with all the features or we should drop the feature we appended in the self.drop_list, it did not like not dropping them.. which was what we expected.
# * it is a little confused between RobustScaler and StandardScaler, lets test out both..
# 
# so finally, we will create dataset using following features...
# 
# * bin_fare = [True, False] -- finally we pick **False**
# * family_size = True 
# * bin_family_size = True
# * drop_Name_length = False 
# * drop_Ticket_frequency = True
# * drop_all = True  
# * scaling = ['StandardScaler', 'RobustScaler'] -- finally we pick **StandardScaler**

# In[39]:


fe = FeatureEngineering(bin_fare=False, bin_age=True, family_size=True, bin_family_size=True,
                 drop_Name_length=False, drop_Ticket_frequency=False, drop_all=True, drop_Family_Survival=False, 
                 drop_Ticket_extracted=False, scaling='StandardScaler', target_encode_title=False, test=False)


# In[40]:


X_train_fe = fe.fit_transform(X_train.copy())
X_train_fe


# In[41]:


fe.test = True


# In[42]:


X_test_fe = fe.transform(X_test.copy())
X_test_fe


# # Hyper parameter Tuning

# In this section we will be finding best hyper parameters for a number of models with different strengths and weaknesses so that we can finally blend the results and achieve higher accuracy. The idea is that every model has different strengths and weaknesses, there will be instances that,  for eg. SVC classifies correctly but LogisticRegression cannot, but LogisticRegression might classify some other instances correctly which SVC cannot, these difference will be learnt by the final **meta model** in stacking and we will be able to achieve higher accuracy.
# 
# We will train the following models as base models for stacking:
# * LogisticRegression
# * SVC
# * KNeighborsClassifier
# * RandomForestClassifier
# * GradientBoostingClassifier
# * XGBCLassifier
# 
# 
# #### Tips:
# 
# * When grid searching hyperparameters you should look at top few best results to get a better idea, if you find a parameter mix which gives you a decent result(not the best) but it is more regulrized than the top result, choose the regularized option and accept the loss in accuracy(if its not a lot), This will ensure that the variance is low and you will actually have more chance of achieving similar result with first model than the best model as per grid search. 
# 
# * The above practice becomes a cumbersome for tree ensemble methods if you just have to go through the grid search results, simple reason being that there are too many parameters which affect the bias and variance, unlike LogisticRegression or SVC. In this case we will plot the results!!! check out the code below.

# In[43]:


def parameter_plot(model, X, y, n_estimators=[100, 200, 300, 400, 600, 900, 1300, 1700, 2000, 2500], 
                   hyper_param=None, **kwargs):
    param_name, param_vals = hyper_param
    param_grid = {'n_estimators':n_estimators,
                  f'{param_name}':param_vals}
    
    grid = GridSearchCV(model(**kwargs), param_grid, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42), 
                        scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    results = pd.DataFrame(grid.cv_results_)['mean_test_score'].values
    results = results.reshape(len(param_vals), len(n_estimators))
    
    plt.figure(figsize=(15, 9))
    for i in range(1, len(param_vals) + 1):
        plt.plot(n_estimators, results[i-1], label=f'{param_name} - {param_vals[i-1]}')
      
    plt.legend()
    plt.show()


# In[44]:


def learning_curve_plotter(Model, X, y, params_1, params_2, step=50):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    plt.figure(figsize=(16, 7))
    for i, (name, params) in enumerate([params_1, params_2]):
        train_score = []
        val_score = []
        plt.subplot(1, 2, i+1)
        for j in range(100, len(X_train), step):
            model = Model(**params).fit(X_train[:j], y_train[:j])
            train_score.append(model.score(X_train[:j], y_train[:j]))
            val_score.append(model.score(X_test, y_test))
            
        plt.plot(train_score, 'r-', label='Training accuracy')
        plt.plot(val_score, 'b-', label='Validation accuracy')
        plt.title(f'{name}')
        plt.xlabel('Training set size')
        plt.ylabel('Accuracy')
        plt.legend()
            
    plt.show()


# ### Logistic Regression

# In[45]:


param_grid_logreg = {'penalty':['elasticnet'],
                     'C':0.01 * np.arange(100),
                     'l1_ratio':0.1 * np.arange(10),
                     'solver':['saga']} 


# In[46]:


grid_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[47]:


grid_logreg.fit(X_train_fe, y_train)


# In[48]:


results(grid_logreg, n=60)


# In[49]:


params_logreg = {'C': 0.28, 'l1_ratio': 0.9, 'penalty': 'elasticnet', 'solver': 'saga'}


# ### KNearestNeighbor

# In[50]:


param_grid_knn = {'n_neighbors':np.arange(50),
                  'weights':['uniform'],
                  'algorithm':['ball_tree'],
                  'leaf_size':np.arange(1, 40, 2)}


# In[51]:


grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[52]:


grid_knn.fit(X_train_fe, y_train)


# In[53]:


results(grid_knn, n=60)


# In[54]:


params_knn = {'algorithm': 'ball_tree', 'leaf_size': 1, 'n_neighbors': 7, 'weights': 'uniform'}


# ### SVC

# In[55]:


param_grid_svc = {'C':[0.001, 0.01, 0.1, 1, 5],
                  'kernel':['rbf'],
                  'gamma':0.01 * np.arange(100),
                  'probability':[True]}


# In[56]:


grid_svc = GridSearchCV(SVC(), param_grid_svc, cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[57]:


grid_svc.fit(X_train_fe, y_train)


# In[58]:


results(grid_svc, n=60)


# In[59]:


params_svc = {'C': 1, 'gamma': 0.09, 'kernel': 'rbf', 'probability': True}


# ### RandomForestClassifier

# In[60]:


parameter_plot(RandomForestClassifier, X_train_fe, y_train, hyper_param=('max_depth', [3, 4, 5, 8, 9, None]))


# We can see that max depth of 3 is not good in terms of accuracy, but it will be great to prevent overitting... as we increase the depth accuracy increases, Lets pick max depth of 9 and 5 to test out, later we will see that though max depth 9 gives better accuracy in grid search, it tends to create a overfitted model whereas max depth 5 will create a model with slightly lower accuracy but significantly balanced

# In[61]:


param_grid_random = {'n_estimators':[300, 500, 1000],
                     'max_depth':[5, 9],
                     'max_samples':[0.5, 0.7, 0.9],
                     'max_features':[0.5, 0.7, 0.9], 
                     'min_samples_split':[2, 5, 8]
                    } 


# In[62]:


grid_random = GridSearchCV(RandomForestClassifier(), param_grid_random, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[63]:


grid_random.fit(X_train_fe, y_train)


# In[64]:


results(grid_random, n=162)


# In[65]:


params_1 = ('Regularized model', {'max_depth': 5, 'max_features': 0.5, 'max_samples': 0.9,
                                  'min_samples_split': 8, 'n_estimators': 300} )

params_2 = ('Best model', {'max_depth': 9, 'max_features': 0.7, 'max_samples': 0.9, 
                           'min_samples_split': 8, 'n_estimators': 300})

learning_curve_plotter(RandomForestClassifier, X_train_fe, y_train, params_1, params_2, step=30)


# Its quite clear from the above plot that which set of params we are going to use, So, you should not just pick the best grid search result.... check for a regularized options that give you decent accuracy result too.

# In[66]:


params_random = {'max_depth': 5, 'max_features': 0.5, 'max_samples': 0.9,
                                  'min_samples_split': 8, 'n_estimators': 300}


# ### GradientBoostingClassifier

# In[67]:


parameter_plot(GradientBoostingClassifier, X_train_fe, y_train, hyper_param=('max_depth', [3, 4, 5, 6, 9]))


# max depth of 3 looks good

# In[68]:


parameter_plot(GradientBoostingClassifier, X_train_fe, y_train, hyper_param=('learning_rate', [0.01, 0.03, 0.05, 0.07, 0.1]),
              max_depth=3)


# lower learning rates are good!! lets test out small learning rates with a slightly higher one, we can see that as n_estiators increase the model starts to overfit and the accuracy goes down.

# In[69]:


param_grid_gradient = {'max_depth':[3, 4],
                       'n_estimators':[300, 400, 500],
                       'learning_rate':[0.01, 0.03, 0.05],
                       'subsample':[0.5, 0.7],
                       'max_features':[0.5, 0.7],
                       #'min_samples_split':[2, 5, 8, 12]
                      }


# In[70]:


grid_gradient = GridSearchCV(GradientBoostingClassifier(), param_grid_gradient, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[71]:


grid_gradient.fit(X_train_fe, y_train)


# In[72]:


results(grid_gradient, n=60)


# In[73]:


params_1 = ('Regularized model', {'learning_rate': 0.01, 'max_depth': 3, 'max_features': 0.5, 
                                  'n_estimators': 500, 'subsample': 0.5})
params_2 = ('Best model', {'learning_rate': 0.01, 'max_depth': 3, 'max_features': 0.7, 
                           'n_estimators': 400, 'subsample': 0.7})

learning_curve_plotter(GradientBoostingClassifier, X_train_fe, y_train, params_1, params_2, step=30)


# In[74]:


params_gradient = {'learning_rate': 0.01, 'max_depth': 3, 'max_features': 0.5, 'n_estimators': 500, 'subsample': 0.5}


# ### XGBClassifier

# In[75]:


parameter_plot(XGBClassifier, X_train_fe, y_train, hyper_param=('max_depth', [3, 4, 5, 6]))


# From the above plot, max_depth of 3 - 4 looks good!!

# In[76]:


parameter_plot(XGBClassifier, X_train_fe, y_train, 
               hyper_param=('learning_rate', [0.01, 0.03, 0.05, 0.07, 0.1]), max_depth=4) 


# we passed in the max depth of 4 as we found in the last plot to get a better estimate of learning rate as per max depth 4, from the above plots lower learning rates look good!! you can also see that as number of trees increase the accuracy goes down.

# In[77]:


param_grid_xgb = {'n_estimators':[400, 600],
                  'learning_rate':[0.01, 0.03, 0.05],
                  'max_depth':[3, 4],
                  'subsample':[0.5, 0.7],
                  'colsample_bylevel':[0.5, 0.7],
                  'reg_lambda':[15, None],
                 }


# In[78]:


grid_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[79]:


grid_xgb.fit(X_train_fe, y_train)


# In[80]:


results(grid_xgb, n=60)


# From the these set of params lets select the best params and the params we think are less prone to overfitting..
# 
# In Gradient boosting more learning rate, more number of trees, more subsample and colsample_bylevel or node or tree and no reg_lambda or alpha... causes overfitting.

# In[81]:


params_1 = ('Regularized model',{'colsample_bylevel': 0.7, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 400,
                                 'reg_lambda': 15, 'subsample': 0.5})
params_2 = ('Best model', {'colsample_bylevel': 0.7, 'learning_rate': 0.03, 'max_depth': 4, 'n_estimators': 600, 
                           'reg_lambda': None, 'subsample': 0.7} )

learning_curve_plotter(XGBClassifier, X_train_fe, y_train, params_1, params_2, step=50)


# The reg_lambda : None, max_depth:4 and n_estimators:600 which are more than the regularized model are the features which are making the best model overfit the data.

# In[82]:


params_xgb = {'colsample_bylevel': 0.7, 'learning_rate': 0.03, 'max_depth': 3, 
              'n_estimators': 400, 'reg_lambda': 15, 'subsample': 0.5}


# #### Lets summarize the way we just grid searched good parameters
# 
# * we first checked out our tree ensemble model's performance on the given dataset and then shortlisted range of max depths and n_estimators which are the mone of the most influential parameters.
# 
# * then we grid searched the shorlisted parameter values and with some other parameters such as max_samples and max_features, these parameters are must to search as they diversify our trees and make them less similar to each other.
# 
# * after we get the results, we then start searching the small parameter space near the optimal parameters obtained from gridsearch, we will usually see increase in accuracy for next few small gridsearchs this way.
# 
# * Check for overfitting and pick that model that has good accuracy with least overfitting...

# ### Predictions

# In[83]:


logreg = LogisticRegression(**params_logreg)
svc = SVC(**params_svc)
knn = KNeighborsClassifier(**params_knn)
rfc = RandomForestClassifier(**params_random)
gradient = GradientBoostingClassifier(**params_gradient)
xgb = XGBClassifier(**params_xgb)

estimators = [('logreg', logreg), ('knn', knn), ('svc', svc), ('rfc', rfc), ('gradient', gradient), 
              ('xgb', xgb)]


stack = StackingClassifier(estimators=estimators,
                           cv=10, n_jobs=-1)


# Stacking as above yielded 0.799 with survival rate and 0.784 without survival rate, an improvement of 0.145 from without survival rate feature... But i expected more and when i explored the individual model accuracy, i knew why i did not find many notebooks with survival rate feature and stacking...
# 
# * With survival rate : 0.799
# * Without survival rate : 0.784

# In[84]:


model = XGBClassifier(**{'colsample_bylevel': 0.7, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 400,
                                 'reg_lambda': 15, 'subsample': 0.5}).fit(X_train_fe, y_train)


# In[85]:


y_preds = model.predict(X_test_fe)


# In[86]:


submission = pd.DataFrame({'PassengerId':test.index, 
              'Survived':y_preds})


# In[87]:


submission.to_csv('submission.csv', index=False)


# In[88]:


pd.read_csv('submission.csv')


#  ***If you found it helpful, upvote!!***
#  
#  ***If you have forked the kernel and not upvoted yet, then show the support by upvoting :)***
#  
#  ***Please leave you constructive criticism and suggestion in comments below!!***

# In[ ]:




