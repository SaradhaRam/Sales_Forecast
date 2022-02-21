# -*- coding: utf-8 -*-
"""
@author: SRamasamy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from pylab import rcParams
rcParams['figure.figsize'] = 12,6

server = 'from config' 
database = 'from config'
username = 'from config' 
password = 'from config'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
db_cmd = "SELECT date, Unitno, Sales FROM tablename WHERE  accountdetails ='Net Sales' AND date >='your date' order by date ASC"
df = pd.read_sql(db_cmd,cnxn)

df["Weekday"]=df["Date"].dt.day_name()

Weekend = ['Friday','Saturday','Sunday']
df["Weekday"] = np.where(df["Weekday"].isin(Weekend)==False, 0,1)
df['Date'] = pd.to_datetime(df['Date'])

from statsmodels.tsa.ar_model import AutoReg    
from sklearn.metrics import mean_squared_error
df_forecast1 = pd.DataFrame()

# group by 'IDQ' column
groups = df.groupby('Unitno')
  
for store, group in groups:
    try:
        #if(store == '46128' or store == '40347' or store == '41423'):
        print('Store Name:',store)
        dff = group[['Date','Sales','Weekday']].set_index('Date')
        print("\n")
        dff.index = pd.to_datetime(dff.index)
        df=dff.asfreq('D').fillna(dff.mean())
        df=df.dropna()
        auto_arima(df['Sales'],maxiter=1000)
        df_transformed = df.diff().dropna()
        nobs=14
        train,test = df_transformed[:-nobs],df_transformed[-nobs:]
        model=VARMAX(train, order=(5,1), trend = 'c')
        results = model.fit(maxiter=1000, disp=False)
        results.summary()
        fcst = nobs+14
        df_forecast=results.forecast(fcst)
        df_forecast
        df_forecast['VARMA predictions']= df['Sales'].iloc[-nobs-1]+df_forecast['Sales'].cumsum()
        df_forecast['VARMA predictions'].plot(legend=True)
        df['Sales'][-nobs:].plot(legend=True)

    #             from sklearn.metrics import mean_squared_error
    #             rmse = np.sqrt(mean_squared_error(df['Sales'][-nobs:], df_forecast['VARMA predictions'])) #compare error to other models
    #             print('Test RMSE: %.3f' % rmse)

        df_forecast['Sales Act'] = df['Sales'].iloc[-30:]
        df_forecast['Unitno'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','VARMA predictions','Unitno']])
        #df_forecast.to_csv('VARMA_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('VARMA_MODEL_OUTPUT/output_VARMA'+'.csv')
    except:
        print('Skipped store_ name:  ',store)
        pass
        
        
        
        
        