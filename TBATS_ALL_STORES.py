# -*- coding: utf-8 -*-
"""
@author: SRamasamy
"""

import numpy as np #math operations
import matplotlib.pyplot as plt #plots
import pandas as pd #importing datasets
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pyodbc #set up a DW connection
from sklearn.metrics import mean_squared_error
import datetime
from pylab import rcParams
rcParams['figure.figsize'] = 12,6

# Hide warning messages in notebook
import warnings
warnings.filterwarnings('ignore')

server = 'from config' 
database = 'from config'
username = 'from config' 
password = 'from config'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
db_cmd = "SELECT date, Unitno, Sales FROM tablename WHERE  accountdetails ='Net Sales' AND date >='your date' order by date ASC"
df = pd.read_sql(db_cmd,cnxn)

from tbats import TBATS, BATS   
from sklearn.metrics import mean_squared_error
df_forecast1 = pd.DataFrame()

# group by 'IDQ' column
groups = df.groupby('Unitno')
  
for store, group in groups:
    try:
        #if(store == '46128' or store == '40347' or store == '41423'):
        print('Store Name:',store)
        dff = group[['Date','Sales']].set_index('Date')
        print("\n")
        dff.index = pd.to_datetime(dff.index)
        df=dff.asfreq('D').fillna(dff.mean())
        #print(df)
        nobs=14
        train = df.iloc[:(len(df)-nobs)]
        test = df.iloc[(len(df)-nobs):]
        # Fit the model
        estimator = TBATS(seasonal_periods=(7,365.25) )
        model = estimator.fit(train)
        # model.summary()
        # Forecast steps= days ahead
        forecast = pd.DataFrame(model.forecast(steps=len(test)+30))
        idx=pd.date_range(test.index[0], periods=len(test)+30, freq='D')
        df_forecast = pd.DataFrame(data=forecast, index=idx, columns = ['Sales Act', 'Sales Fcst'])
        df_forecast.reset_index(inplace=True)
        df_forecast['Sales Fcst'] = forecast
        df_forecast.set_index('index', inplace=True)
        df_forecast.index.freq='D'
        df_forecast['Sales Act'] = df['Sales']
        df_forecast['Unitno'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','Sales Fcst','Unitno']])
        #df_forecast.to_csv('TBATS_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('TBATS_MODEL_OUTPUT/output_TBATS'+'.csv')
    except:
        print('Skipped store_ name:  ',store)
        pass