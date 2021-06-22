#!/usr/bin/env python
# coding: utf-8

# # Yen-Dollar Forex Trading

# In[ ]:





# In[3]:


import numpy as np
import pandas as pd
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm


# ## Return Forecasting: Read Historical Daily Yen Futures Data
# _____________________________________________
# 
# #### In this notebook, you will load historical Dollar-Yen exchange rate futures data and apply time series analysis and modeling to determine whether there is any predictable behavior.

# In[4]:


# Futures contract on the Yen-dollar exchange rate:
# This is the continuous chain of the futures contracts that are 1 month to expiration
yen_futures = pd.read_csv("yen.csv", index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()


# In[5]:


# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures.head() 


# ## Return Forecasting: Initial Time-Series Plotting
# ____________________________________
# 
# ##### Start by plotting the "Settle" price. Do you see any patterns, long-term and/or short?

# In[6]:


settle=yen_futures["Settle"].plot()
settle


# In[ ]:





# ## Decomposition Using a Hodrick-Prescott Filter
# _______________________________
# 
# ##### Using a Hodrick-Prescott Filter, decompose the Settle price into a trend and noise.

# In[7]:


s_noise, s_trend = sm.tsa.filters.hpfilter(yen_futures["Settle"])


# In[8]:


yen_futures["noise"] = s_noise
yen_futures["trend"] = s_trend


# In[9]:


new_settle_df=yen_futures[["Settle","noise","trend"]]


# In[10]:


new_settle_df.head()


# In[11]:


settle_vs_trend = yen_futures[["Settle","trend"]]
settle_vs_trend.plot(figsize=(20,10))


# In[12]:


# Settle Noise

new_settle_df["noise"].plot(figsize=(20,10, ))


# In[ ]:





# ## Forecasting Returns using an ARMA Model
# ___________________________________
# 
# ##### Using futures Settle Returns, estimate an ARMA model

# * ARMA: Create an ARMA model and fit it to the returns data. Note: Set the AR and MA ("p" and "q") parameters to p=2 and q=1: order=(2, 1). 
# 
# * Output the ARMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
# 
# * Plot the 5-day forecast of the forecasted returns (the results forecast from ARMA model)

# In[13]:


# Create a series using "Settle" price percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
returns = (new_settle_df[["Settle"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()


# In[15]:


from statsmodels.tsa.arima_model import ARMA

# Estimate and ARMA model using statsmodels (use order=(2, 1))

model = ARMA(returns["Settle"].values,order=(2,1))

results = model.fit()

results.summary()

# Fit the model and assign it to a variable called results


# In[ ]:


# Plot the 5 Day Returns Forecast


# In[17]:


pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5-Day Returns Forecast")


# ## Forecasting the Settle Price using an ARIMA Model
# ____________________________________________
# 
# * Using the raw Yen Settle Price, estimate an ARIMA model.
#  * Set P=5, D=1, and Q=1 in the model (e.g., ARIMA(df, order=(5,1,1))
#   * P= # of Auto-Regressive Lags, D= # of Differences (this is usually =1), Q= # of Moving Average Lags
#                                           
# * Output the ARIMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
# * Construct a 5 day forecast for the Settle Price. What does the model forecast will happen to the Japanese Yen in the near term?

# In[24]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(yen_futures["Settle"], order=(5,1,1))

results = model.fit()

results.summary()


# In[26]:


# Plot the 5 Day Price Forecast

pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5-Day Futures Price Forecast")


# ## Volatility Forecasting with GARCH
# __________________________________
# 
# ##### Rather than predicting returns, let's forecast near-term volatility of Japanese Yen futures returns. Being able to accurately predict volatility will be extremely useful if we want to trade in derivatives or quantify our maximum loss.
# 
# * GARCH: Create an GARCH model and fit it to the returns data. Note: Set the parameters to p=2 and q=1: order=(2, 1).
# * Output the GARCH summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
# * Plot the 5-day forecast of the volatility.

# In[30]:


import arch  
from arch import arch_model


# In[31]:


# Estimate a GARCH model:

model=arch_model(returns,mean="Zero", vol="GARCH")
results = model.fit()
results.summary()


# In[32]:


# Find the last day of the dataset
last_day = returns.index.max().strftime('%Y-%m-%d')
last_day


# In[37]:


# Create a 5 day forecast of volatility
forecast=results.forecast(horizon=5)


# In[39]:


annual = np.sqrt(forecast.variance.dropna() * 252)
annual


# In[41]:


# Transpose the forecast so that it is easier to plot

final = annual.dropna().T
final.head()


# In[43]:


# Plot the final forecast

annual.T.plot()


# ## Conclusions
# __________________________
# 
# * Based on your time series analysis, would you buy the yen now?
# * Is the risk of the yen expected to increase or decrease?
# * Based on the model evaluation, would you feel confident in using these models for trading?
# 

# In[ ]:




