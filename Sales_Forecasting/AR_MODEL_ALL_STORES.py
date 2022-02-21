# -*- coding: utf-8 -*-
"""
@author: SRamasamy
"""
import pandas as pd
import numpy as np
# import AutoReg
# from statsmodels.tsa.ar_model import AutoReg
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
        nobs=14
        train=df.iloc[:-nobs]
        test = df.iloc[-nobs:]
        model=AutoReg(train['Sales'],7)
        ARfit = model.fit()
        ARfit.params
        start = len(train)
        end = len(train)+len(test)-1
        predictions = ARfit.predict(start=start, end=end)
        predictions = predictions.rename('Auto Regression Predictions')
        model=AutoReg(df['Sales'],7)
        ARfit = model.fit()
        forecasted_values = ARfit.predict(start = len(df)-1, end=len(df)+30).rename("Forecast")    
        error = mean_squared_error(test, predictions)
        #print(f'RMSE Error: {math.sqrt(error)}')
        idx=pd.date_range(df.index[-1]+ datetime.timedelta(days=-30), periods=len(test)+30, freq='D')
        df_forecast = pd.DataFrame(data=forecasted_values, index=idx, columns = ['Sales Act', 'Sales Pred', 'Sales Fcst'])
        df_forecast['Sales Act'] = df['Sales']
        df_forecast['Sales Pred'] = predictions
        df_forecast['Sales Fcst'] = forecasted_values
        #df_forecast.plot()
        df_forecast['Unitno'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','Sales Pred','Sales Fcst','IDQ']])
        #df_forecast.to_csv('AR_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('AR_MODEL_OUTPUT/output_AR'+'.csv')
    except:
        print('Skipped store_ name:  ',store)
        pass

