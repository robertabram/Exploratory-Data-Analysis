#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import usefull libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read data

df = pd.read_csv('marketing_analysis.csv', skiprows = 2)

df.head()


# In[3]:


# split and drop come columns

df['job'] = df['jobedu'].apply(lambda x: x.split(',')[0])
df['education'] = df['jobedu'].apply(lambda x: x.split(',')[1])

df.drop(['customerid','jobedu'], axis = 1, inplace = True)

df.head()


# In[4]:


# check missing values

df.isnull().sum()

df.dropna(subset = 'age',inplace = True)

df.isna().sum()

month_mode = df['month'].mode()[0]

df['month'].fillna(month_mode, inplace = True)

df.dropna(subset = 'response',inplace = True)

df.isna().sum()


# In[5]:


# plot bar chart with categorical column job

df.job.value_counts(normalize = True)

df.job.value_counts(normalize = True).plot.barh()
plt.show()


# In[6]:


# create pie chart of education column

df.education.value_counts(normalize = True)

df.education.value_counts(normalize = True).plot.pie()
plt.show()


# In[7]:


# scatterplot of two numeric data

df.plot.scatter(x = 'salary',y = 'balance')
plt.show()

df.plot.scatter(x = 'age',y = 'balance')
plt.show()


# In[8]:


# create a pairplot

sns.pairplot(data = df, vars = ['balance','age','salary'])
plt.show()


# In[9]:


# correlation matrix

df[['balance','salary','age']].corr()

sns.heatmap(df[['balance','salary','age']].corr(), annot = True, cmap = 'Blues')


# In[10]:


# categorical-numeric analysis (mean and median)

df.groupby('response')['salary'].mean()[0]

#df.groupby('response')['salary'].median()


# In[11]:


# plot boxplot for salary and response

sns.boxplot(x = df.response,y = df.salary)
plt.show()


# In[12]:


# categorical-categorical analysis
# create a new column woth substitute for yes and no
df['response_rate'] = np.where(df['response'] == 'yes',1, 0)

df.response_rate.value_counts()


# In[13]:


# count values by marital status

df.groupby('marital')['response_rate'].mean().plot.bar()
plt.show()


# In[14]:


# multivariate analysis
# create pivot table and heatmap

pivot = pd.pivot_table(data = df, index = 'education',columns = 'marital',values = 'response_rate')

sns.heatmap(pivot, annot = True, cmap = 'Blues', center = 0.117)
plt.show()

pivot


# In[19]:





# In[ ]:




