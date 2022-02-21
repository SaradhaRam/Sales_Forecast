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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pylab import rcParams
rcParams['figure.figsize'] = 12,6
import warnings
warnings.filterwarnings("ignore")
import datetime
#time
import time


server = 'from config' 
database = 'from config'
username = 'from config' 
password = 'from config'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
db_cmd = "SELECT date, Unitno, Sales FROM tablename WHERE  accountdetails ='Net Sales' AND date >='your date' order by date ASC"
df = pd.read_sql(db_cmd,cnxn)

from statsmodels.tsa.ar_model import AutoReg    
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

        a = auto_arima(df['Sales'], Seasonal=True, m=7)
        nobs=14
        train=df.iloc[:-nobs]
        test = df.iloc[-nobs:]
        model=SARIMAX(train,order=(a.order[0],a.order[1],a.order[2]), seasonal_order=(a.seasonal_order[0],a.seasonal_order[1],a.seasonal_order[2],a.seasonal_order[3]))
        results = model.fit()
        results.summary()
        start = len(train)
        end=len(train)+len(test)-1
        predictions = results.predict(start,end, typ='levels').rename('SARIMA predictions')
        test['Sales'].plot(legend=True)
        predictions.plot(legend=True,xlim=['2021-11-01', '2022-01-01'])
        rmse = np.sqrt(mean_squared_error(test, predictions)) #compare error to other models
        #print('Test RMSE: %.3f' % rmse)

        #FORECAST
        model=SARIMAX(df,order=(a.order[0],a.order[1],a.order[2]), seasonal_order=(a.seasonal_order[0],a.seasonal_order[1],a.seasonal_order[2],a.seasonal_order[3]))
        results = model.fit()
        fcast = results.predict(len(df), len(df)+14, typ="level").rename('SARIMA Forecast')
        df.tail(nobs+10).plot(legend=True)
        #fcast.plot(legend=True,xlim=['2021-11-01', '2022-01-01'])
        fcast.tail(nobs+10).plot(legend=True)
        predictions.tail(nobs+10).plot(legend=True)
        idx=pd.date_range(df.index[-1]+ datetime.timedelta(days=-30), periods=len(test)+30, freq='D')
        df_forecast = pd.DataFrame(data=fcast, index=idx, columns = ['Sales Act', 'Sales Pred', 'Sales Fcst'])
        df_forecast['Sales Act'] = df['Sales']
        #df_forecast['LY Sales']=df['Sales'].shift(364)
        df_forecast['Sales Pred'] = predictions
        df_forecast['Sales Fcst'] = fcast
        #print(df_forecast)
        #df_forecast.plot()
        df_forecast['Unitno'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','Sales Pred','Sales Fcst','IDQ']])
        #df_forecast.to_csv('SARIMA_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('SARIMA_MODEL_OUTPUT/output_SARIMA'+'.csv', index=True, mode='a')
    except:
        print('Skipped store_ name:  ',store)
        pass