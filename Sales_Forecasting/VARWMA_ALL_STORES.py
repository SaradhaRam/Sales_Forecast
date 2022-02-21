# -*- coding: utf-8 -*-
"""
@author: SRamasamy
"""

# Dependencies
import pandas as pd
import numpy as np
import requests
import random
import io

# datetime
import datetime as dt 
from datetime import date ,timedelta


import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from pylab import rcParams
rcParams['figure.figsize'] = 12,6

# Hide warning messages in notebook
import warnings
warnings.filterwarnings('ignore')

# Set up a DW connection
import pyodbc 

#Sales data
#Setting up the connection to the datawarehouse
server = 'from config' 
database = 'from config'
username = 'from config' 
password = 'from config'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
db_cmd = "SELECT date, Unitno, Sales FROM tablename WHERE  accountdetails ='Net Sales' AND date >='your date' order by date ASC"
df = pd.read_sql(db_cmd,cnxn)

sales_df = sales_df.astype({"UnitNumber": int})
sales_df.set_index('Date', inplace=True)
# Rename the column
sales_dff = sales_df.rename(columns={'UnitNumber':'Unitno'})
df_sales_groupby = sales_dff.groupby(['Unitno','Date']).mean()

# Store data
#Setting up the connection to the datawarehouse
server = 'from config' 
database = 'from config'
username = 'from config' 
password = 'from config' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
db_cmd = "SELECT Store, Unitno, Latitude, Longitude FROM Stores where ( Latitude IS NOT NULL) and Status = 'Open'"
store_df = pd.read_sql(db_cmd,cnxn)

store_df[['Latitude','Longitude']] = store_df[['Latitude','Longitude']].astype(str)
# List od DateGroup

# Date Group for 2 years - begin_date should be '2020-01-01'. If you need to change the beginDate, need to adjust the DateGroup array accordingly
# each element in the array represents 3 months,since we can only call weather API for 90 days.
begin_date = date(2020,1,1)
DateGroup = [[begin_date, begin_date + dt.timedelta(90) ],[begin_date + dt.timedelta(91), begin_date + dt.timedelta(180)],
             [begin_date + dt.timedelta(181), begin_date + dt.timedelta(270)],[begin_date + dt.timedelta(271), begin_date + dt.timedelta(360)],
             [begin_date + dt.timedelta(361), begin_date + dt.timedelta(450)],[begin_date + dt.timedelta(451), begin_date + dt.timedelta(540)],
             [begin_date + dt.timedelta(541), begin_date + dt.timedelta(630)],[begin_date + dt.timedelta(631), begin_date + dt.timedelta(720)],
             [begin_date + dt.timedelta(721),dt.datetime.today().date()]]   
#API call:
# Initialize the weatherData
WeatherData = []
from itertools import islice

# for row in islice(store_df.itertuples(), 2):
#     print(row.Latitude,row.Longitude)
for row in store_df.itertuples():
        print(row.Latitude,row.Longitude)
        # for all the DateGroup in the list
        for i in DateGroup:
            BeginDate = i[0].strftime('%Y%m%d')+"0000"
            EndDate = i[1].strftime('%Y%m%d')+"0000"
            query_url = baseURL+row.Latitude+","+row.Longitude+"&startDateTime="+BeginDate+"&endDateTime="+EndDate+"&units=e&format=csv&apiKey="YOUR API KEY"
            weather_response = requests.get(url=query_url,headers={'Content-Type': 'application/octet-stream'})
            data = weather_response.content
            rawData = pd.read_csv(io.StringIO(data.decode('utf-8')))
            WeatherData1 = rawData[['observationTimeUtcIso','precip6Hour','snow6Hour','temperatureMax24Hour','uvIndex']]
            WeatherData1['Unitno'] = int(row.Unitno)
            WeatherData.append(WeatherData1)
weather_df = pd.concat(WeatherData)# Array list to dataframe 
    
weather_df_selected = weather_df[['observationTimeUtcIso','temperatureMax24Hour','Unitno']]   
# Convert datetime to date
weather_df_selected['observationTimeUtcIso'] = pd.to_datetime(weather_df_selected['observationTimeUtcIso']).dt.date
# Rename the column
weather_df_selected= weather_df_selected.rename(columns={'observationTimeUtcIso':'Date','temperatureMax24Hour':'TempMax'})   
df_weather_groupby = weather_df_selected.groupby(['Unitno','Date']).mean()   
df =pd.merge(df_sales_groupby, df_weather_groupby, left_index=True, right_index=True)
df_final1 = df.dropna()
df_final = df_final1.reset_index(level= ['Date','Unitno'])

# Over all loop   
from sklearn.metrics import mean_squared_error
df_forecast1 = pd.DataFrame()

# group by 'Unitno' column
groups = df_final.groupby('Unitno')

  
for storeNo, group in groups:
    store = str(storeNo)
    try:
    #if(store == '40347'):
        print('Store Name:',store)
        dff = group[['Date','Sales','TempMax']].set_index('Date')
        print("\n")
        dff.index = pd.to_datetime(dff.index)
        df=dff.asfreq('D').fillna(dff.mean())
        df=df.dropna()
        df_transformed = df.diff().dropna()
        nobs=14 # 14
        train=df_transformed[:-nobs]
        test = df_transformed[-nobs:]
        model=VAR(train)
        for p in range(20):
            results=model.fit(p)
        #print(f'Order {p}')
        #print(f'AIC {results.aic}')
        print('\n')
        results = model.fit(7)
        results.summary()
        lagged_values = train.values[-7:]
        z = results.forecast(y=lagged_values,steps=len(test))
        #df['Date'] = pd.to_datetime(df['Date']).dt.date
        idx=pd.date_range(df.index[-1], periods=len(test), freq='D') # change the date
        df_forecast = pd.DataFrame(data=z, index=idx, columns = ['Sales_diff', 'Weather_diff'])
        df_forecast['VAR weather predictions']= df['Sales'].iloc[-nobs-1]+df_forecast['Sales_diff'].cumsum()
        df_forecast['TempMax']= df['TempMax'].iloc[-nobs-1]+df_forecast['Weather_diff'].cumsum() 
        #df_forecast['VAR weather predictions'].plot(legend=True)
        #df['Sales'][-nobs:].plot(legend=True)
        #rmse = np.sqrt(mean_squared_error(test['Sales'], df_forecast['VAR weather predictions'])) #compare error to other models
        #print('Test RMSE: %.3f' % rmse)
        df_forecast['Sales Act'] = df['Sales'].iloc[-30:]
        df_forecast['Unitno'] = store
        df_forecast = np.round(df_forecast, decimals=2)
        df_forecast.index.names = ['Date']
        df_forecast1=df_forecast1.append(df_forecast[['Sales Act','VAR weather predictions','Unitno']])
        #df_forecast.to_csv('VAR_MODEL_OUTPUT/output_'+store+'.csv', sep=' ')
        df_forecast1.to_csv('VARWMA_MODEL_OUTPUT/output_VARWMA'+'.csv')
    except:
        print('Skipped store_ name:  ',store)
        pass
    