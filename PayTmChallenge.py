# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:36:01 2019

@author: PCUser
"""

import re
import datetime
import pandas as pd
import numpy as np
#from pandas import datetime


log_file_path = r"2015_07_22_mktplace_shop_web_log_sample.log"
#regex = '(<property name="(.*)"<\/property>)'
 
# I am goign to ttake a simpler route, just extract the following fields from the log file and 
# put them in a dictionary 
dic = {'datetime':[],
       'date': [],
       'time':[],
       'year': [],
       'month': [],
       'day':[],
       'hour': [],
       'minute':[],
       'second':[],
       'ipAdr': [],
       'userAgent': [],
       'url': []
      }

# every line is a session that starts with a date and time field, I use strptime to extract the details from 
# that fields and store them in their respective columns
#basically I am sessionizing by time.
import matplotlib.pyplot as plt

with open(log_file_path, "r") as file:
     for line in file:
            fields = line.split()
            a = datetime.datetime.strptime(fields[0], "%Y-%m-%dT%H:%M:%S.%fZ")
            dic['datetime'].append(a)
            dic['date'].append(a.date())
            dic['time'].append(a.time())
            dic['year'].append(a.year)
            dic['month'].append(a.month)
            dic['day'].append(a.day)
            dic['hour'].append(a.hour)
            dic['minute'].append(a.minute)
            dic['second'].append(a.second)
            dic['ipAdr'].append(fields[2])
            dic['userAgent'].append(fields[3])
            dic['url'].append(fields[12])

datadict = pd.DataFrame(dic)

#Determine the average session time
avgsession = datadict.groupby(['date', 'userAgent','url','ipAdr'])['datetime'].agg({
        'duration': lambda x: (max(x)-min(x)).total_seconds()
        })   
avgsession.duration.unique()
avgsession.duration.mean()
avgsession.duration.max()


#Determine unique URL visits per session. To clarify, count a hit to a unique URL only once per session.
counturl = datadict.groupby(['date','userAgent','url'])['url'].agg({'count'})       
            
#Find the most engaged users, ie the IPs with the longest session times
avgsession.sort_values("duration", ascending = 0 )
avgsession.duration.max()


#from collections import Counter
#Counter(datadict['ipAdr']).values()

#Additional questions for Machine Learning Engineer (MLE) candidates:
#Predict the expected load (requests/second) in the next minute
# Time series foreacting using ARMIA for this . 
from statsmodels.tsa.arima_model import ARIMA
cpm = datadict.groupby(['time'])['url'].agg({'count'})       
cpm.plot()

model_arima = ARIMA(cpm, order=(3,1,1,))
model_arima_fit = model_arima.fit()
prediction = model_arima_fit.forecast(steps=9)[0]
#Predict the session length for a given IP
iplen = avgsession.groupby(['ipAdr'])['duration'].agg({'sum'})       

model_arima = ARIMA(cpm, order=(3,1,1,))
model_arima_fit = model_arima.fit()
prediction = model_arima_fit.forecast(steps=9)[0]

#Predict the number of unique URL visits by a given IP
urlip = datadict.groupby(['ipAdr','url'])['url'].agg({'count'})       

model_arima = ARIMA(cpm, order=(3,1,1,))
model_arima_fit = model_arima.fit()
prediction = model_arima_fit.forecast(steps=9)[0]