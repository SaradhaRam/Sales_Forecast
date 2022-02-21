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
from statsmodels.tsa.api import VAR
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
Weekend = ['Thursday','Friday','Saturday','Sunday']
df["Weekday"] = np.where(df["Weekday"].isin(Weekend)==False, 0,1)

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
        df_transformed = df.diff().dropna()
        nobs=14
        train=df_transformed[:-nobs]
        test = df_transformed[-nobs:]
        model=VAR(train)
        for p in range(20):
            results=model.fit(p)
        results = model.fit(7)
        lagged_values = train.values[-7:]
        z = results.forecast(y=lagged_values,steps=len(test))
        idx=pd.date_range(df.index[-1], periods=len(test), freq='D')
        df_forecast = pd.DataFrame(data=z, index=idx, columns = ['Sales_diff', 'Weekday_diff'])
        #print(df_forecast)
        df_forecast['VAR predictions']= df['Sales'].iloc[-nobs-1]+df_forecast['Sales_diff'].cumsum()
        df_forecast['Weekday']= df['Weekday'].iloc[-nobs-1]+df_forecast['Weekday_diff'].cumsum() 
        df_forecast['VAR predictions'].plot(legend=True)
        df['Sales'][-nobs:].plot(legend=True)

        rmse = np.sqrt(mean_squared_error(test['Sales'], df_forecast['VAR predictions'])) #compare error to other models
        #print('Test RMSE: %.3f' % rmse)

        df_forecast['Sales Act'] = df['Sales'].iloc[-30:]
        df_forecast['Unitno'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','VAR predictions','Unitno']])
        #df_forecast.to_csv('VAR_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('VAR_MODEL_OUTPUT/output_VAR'+'.csv')
    except:
        print('Skipped store_ name:  ',store)
        pass
