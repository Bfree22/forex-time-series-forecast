#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')


# # Regression Analysis: Seasonal Effects with Sklearn Linear Regression
# __________________________
# 
# * In this notebook, I will build a SKLearn linear regression model to predict Yen futures ("settle") returns with lagged Yen futures returns. 

# In[4]:


# Futures contract on the Yen-dollar exchange rate:

# This is the continuous chain of the futures contracts that are 1 month to expiration

yen_futures = pd.read_csv("yen.csv", parse_dates=True,index_col="Date")
yen_futures.head()


# In[5]:


# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures.head()


# ## Data Preparation
# _______________________
# 
# ##### Returns

# In[11]:


# Create a series using "Settle" price percentage returns, drop any nan"s, and check the results:
# In this case, you may have to replace inf, -inf values with np.nan"s

yen_futures["Returns"] = yen_futures["Settle"].pct_change() * 100
returns = yen_futures["Returns"].replace(-np.inf, np.nan).dropna()
returns = pd.DataFrame(returns).head()


# In[19]:


yen_futures = yen_futures.dropna()


# In[21]:


yen_futures.tail()


# ## Lagged Returns
# ________________________

# In[25]:


# Create a lagged return using the shift function

yen_futures["lagged_return"] = yen_futures["Returns"].shift()
yen_futures=yen_futures.dropna()
yen_futures.tail()


# ## Train Test Split
# _________________________________

# In[26]:


# Create a train/test split for the data using 2018-2019 for testing and the rest for training
train = yen_futures[:'2017']
test = yen_futures['2018':]


# In[27]:


X_train = train["lagged_return"].to_frame()

X_test = test["lagged_return"].to_frame()

y_train = train["Returns"]

y_test = test["Returns"]


# In[28]:


X_train


# ## Linear Regression Model
# ______________________

# In[30]:


# Create a Linear Regression model and fit it to the training data
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)


# In[31]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)


# ## Make predictions using the Testing Data
# _________________________________________
# 
# * Note: We want to evaluate the model using data that it has never seen before, in this case: X_test.

# In[32]:


# Make a prediction of "y" values using just the test dataset

predicted_y_values = model.predict(X_test)


# In[33]:


# Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:

Results = y_test.to_frame()

Results["Predicted Return"] = predicted_y_values


# In[35]:


Results[:20].plot(subplots=True)


# ## Out-Of-Sample Performance
# ________________________
# 
# * Evaluate the model using "out-of-sample" data (X_test and y_test)

# In[36]:


from sklearn.metrics import mean_squared_error
# Calculate the mean_squared_error (MSE) on actual versus predicted test "y" 

mse=mean_squared_error(Results["Returns"], Results["Predicted Return"])


# In[38]:


rmse= np.sqrt(mse)
rmse


# ## In-Sample Performance
# ________________________
# 
# * Evaluate the model using in-sample data (X_train and y_train)

# In[39]:


in_sample_results=y_train.to_frame()
    
in_sample_results["in sample prediction"] = model.predict(X_train)

mse_in_sample=mean_squared_error(in_sample_results["Returns"], in_sample_results["in sample prediction"])
rmse_in_sample= np.sqrt(mse_in_sample)
rmse_in_sample


# In[40]:


rmse_in_sample > rmse


# ## Conclusion
