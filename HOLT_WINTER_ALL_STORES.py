# -*- coding: utf-8 -*-
"""
@author: SRamasamy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
from pylab import rcParams
import math
rcParams['figure.figsize'] = 12,6
import datetime

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

from statsmodels.tsa.holtwinters import ExponentialSmoothing    
from sklearn.metrics import mean_squared_error

df_forecast1 = pd.DataFrame()
# group by 'IDQ' column
groups = df.groupby('unitno')
for store, group in groups:
    try:
        #if(store == '46128' or store == '40347' or store == '41423'):
        print('Store Name:',store)
        dff = group[['Date','Sales']].set_index('Date')
        print("\n")
        dff.index = pd.to_datetime(dff.index)
        df=dff.asfreq('D').fillna(dff.mean())
        nobs=14
        train=df.iloc[:len(df)-nobs]
        test=df.iloc[len(df)-nobs:]
        fitted_model=ExponentialSmoothing(train['Sales'],trend='add', seasonal='add',seasonal_periods=7).fit()
        test_predictions = fitted_model.forecast(len(test))
        train['Sales'].plot(legend=True, label='Train')
        test['Sales'].plot(legend=True, label='Test')
        test_predictions.plot(legend=True, label='Predictions',xlim=['2021-11-01', '2022-01-01'])
        rmse = np.sqrt(mean_squared_error(test, test_predictions.iloc[:len(test)])) #compare error to other models
        #print('Test RMSE: %.3f' % rmse)
        final_model=ExponentialSmoothing(df['Sales'],trend='add', seasonal='add',seasonal_periods=7).fit()
        forecast_predictions =final_model.forecast(15)
        train['Sales'].plot(legend=True, label='Train')
        test['Sales'].plot(legend=True, label='Test')
        test_predictions.plot(legend=True, label='Predictions')
        forecast_predictions.plot(legend=True, label="Forecast",xlim=['2021-11-01', '2022-01-01'])
        idx=pd.date_range(df.index[-1]+ datetime.timedelta(days=-30), periods=len(test)+30, freq='D')
        df_forecast = pd.DataFrame(data=df['Sales'], index=idx, columns = ['Sales Act', 'Sales Pred', 'Sales Fcst'])
        df_forecast['Sales Act'] = df['Sales']
        #df_forecast['LY Sales']=df['Sales'].shift(364)
        df_forecast['Sales Pred'] = test_predictions
        df_forecast['Sales Fcst'] = forecast_predictions
        #print(df_forecast)
        #df_forecast.plot()
        df_forecast['IDQ'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','Sales Pred','Sales Fcst','Unitno']])
        #df_forecast.to_csv('HOLTWINTER_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('HOLTWINTER_MODEL_OUTPUT/output_HW'+'.csv')
    except:
        print('Skipped store_ name:  ',store)
        pass
