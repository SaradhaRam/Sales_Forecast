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
import datetime

server = 'from config' 
database = 'from config'
username = 'from config' 
password = 'from config'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
db_cmd = "SELECT date, Unitno, Sales FROM tablename WHERE  accountdetails ='Net Sales' AND date >='your date' order by date ASC"
df = pd.read_sql(db_cmd,cnxn)

df.head()
# df.reset_index('Date', inplace=True)
df["Weekday"]=df["Date"].dt.day_name()

from statsmodels.tsa.ar_model import AutoReg    
from sklearn.metrics import mean_squared_error
df_forecast1 = pd.DataFrame()

# group by 'IDQ' column
groups =df.groupby('Unitno')
  
for store, group in groups:
    try:
        #if(store == '46128' or store == '40347' or store == '41423'):
        print('Store Name:',store)
        dff = group[['Date','Sales','Weekday']].set_index('Date')
        print("\n")
        dff.index = pd.to_datetime(dff.index)
        df=dff.asfreq('D').fillna(dff.mean())
        df.reset_index(inplace=True)
        ffholidays = ['2021-11-25','2021-12-25']
        Weekend = ['Friday','Saturday','Sunday']
        df["Holiday"] = np.where(df["Date"].isin(ffholidays)==False, 0,1)
        df["Weekday"] = np.where(df["Weekday"].isin(Weekend)==False, 0,1)
        df.set_index('Date', inplace=True)
        df.index.freq='D'
        df=df.fillna(0)
        df=df.dropna()
        #ax=df['Sales'].plot()
        #for day in df.query('Holiday==1').index:
            #ax.axvline(x=day, color='black', alpha = 0.5);
        nobs=14
        train=df.iloc[:-nobs]
        test = df.iloc[-nobs:]
        df=df.dropna()
        a = auto_arima(df['Sales'], exogenous = df[['Weekday','Holiday']], seasonal=True, m=7,trace=False)
        a.summary()
        #Find the SARIMAX order above
        model=SARIMAX(train['Sales'],exog=train[['Weekday','Holiday']],order=(a.order[0],a.order[1],a.order[2]), seasonal_order=(a.seasonal_order[0],a.seasonal_order[1],a.seasonal_order[2],a.seasonal_order[3]), enforce_invertibility=False)
        results = model.fit()
        start = len(train)
        end=len(train)+len(test)-1
        pred = results.predict(start,end,exog=test[['Weekday','Holiday']], typ='levels').rename('SARIMAX predictions')
        #ax=test['Sales'].plot(legend=True)
        #pred.plot() #legend=True,xlim=['2021-11-01', '2022-01-01']
        #for day in test.query('Holiday==1').index:
            #ax.axvline(x=day, color='black', alpha = 0.5);
        rmse = np.sqrt(mean_squared_error(test['Sales'], pred)) #compare error to other models
        #print('Test RMSE: %.3f' % rmse)
        #FORECAST
        model=SARIMAX(df['Sales'],order=(a.order[0],a.order[1],a.order[2]), seasonal_order=(a.seasonal_order[0],a.seasonal_order[1],a.seasonal_order[2],a.seasonal_order[3]))
        results = model.fit()
        fcast = results.predict(len(df), len(df)+30, typ="levels").rename('SARIMAX Forecast')
        #ax=df['Sales'].plot(legend=True)
        #fcast.plot()#(legend=True,xlim=['2021-11-01', '2022-01-10'])
        #for day in df.query('Holiday==1').index:
            #ax.axvline(x=day, color='black', alpha = 0.5);
        idx=pd.date_range(df.index[-1]+ datetime.timedelta(days=-30), periods=len(test)+30, freq='D')
        df_forecast = pd.DataFrame(data=fcast, index=idx, columns = ['Sales Act', 'Sales Pred', 'Sales Fcst'])
        df_forecast['Sales Act'] = df['Sales']
        #df_forecast['LY Sales']=df['Sales'].shift(364)
        df_forecast['Sales Pred'] = pred
        df_forecast['Sales Fcst'] = fcast
        #df_forecast.plot()
        df_forecast['Unitno'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','Sales Pred','Sales Fcst','Unitno']])
        #df_forecast.to_csv('SARIMAX_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('SARIMAX_MODEL_OUTPUT/output_SARIMAX'+'.csv')
    except:
        print('Skipped store_ name:  ',store)
        pass
