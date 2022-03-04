# checkpoint5
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[2]:


get_ipython().system('pip install numpy')


# In[3]:


get_ipython().system('pip install matplotlib')


# In[4]:


get_ipython().system('pip install seaborn')


# In[5]:


get_ipython().system('pip install scikit-learn')


# In[6]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict 


# In[7]:


#load the data set
df=pd.read_excel('kc_house_data.xlsx')
df.head(5)


# In[ ]:


# data preprocessing 


# In[8]:


df.info()


# In[9]:


df.shape


# In[12]:


df.dropna()


# In[13]:


##Describe gives statistical information about numerical columns in the dataset
df.describe()


# In[14]:


df['price'].describe()


# In[22]:


#make a list of important features which is need to be included in training data
f =['price','bedrooms','bathrooms','sqft_living','floors','condition','sqft_above','sqft_basement','yr_renovated']
df=df[f]
df.shape


# In[ ]:


#data visualization 


# In[16]:


sns.pairplot(df)


# In[21]:


sns.heatmap(df)


# In[20]:


df.hist(figsize=(20,20))


# In[23]:


x=df[f[1:]]
y=df['price']


# In[24]:


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[25]:


#fit he regression model
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.coef_)


# In[26]:


#create the predictions
y_test_predict=lr.predict(x_test)
print(y_test_predict)


# In[27]:


#plot the error 
g=plt.plot((y_test - y_test_predict),marker='o',linestyle="")


# In[39]:


#fit he regression model without b
lr=LinearRegression(fit_intercept=False)
lr.fit(x_train,y_train)
y_test_predict=lr.predict(x_test)

g=plt.plot((y_test - y_test_predict),marker='o',linestyle="")


# In[32]:


lr.score(x_test,y_test)  


# In[60]:


x_train_ploy, x_test_ploy, y_train_ploy, y_test_ploy = train_test_split( x, y, test_size=0.2, random_state=42)

print(x_train_ploy.shape)
print(x_test_ploy.shape)
print(y_train_ploy.shape)
print(y_test_ploy.shape)


# In[56]:


from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(2)
x_train_poly=pr.fit_transform(x_train)
x_test_poly=pr.fit_transform(x_test)
print(pr.get_feature_names())
print(x_train_poly.shape)


# In[70]:


lr = LinearRegression()
lr.fit(x_train_ploy, y_train)


# In[72]:


y_test_predict =lr.predict(x_test_ploy)
y_test_predict.shape


# In[73]:


g=plt.plot((y_test - y_test_predict),marker='o',linestyle="")


# In[ ]:




