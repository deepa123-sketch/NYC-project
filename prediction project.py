#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[3]:


data_1=pd.read_csv(r'C:\Users\DELL\Documents\nyc_taxi_trip_duration.csv')
data_1.head()


# In[4]:


# Trip Duration
print('The value of largest 5 trip duration values are as follows : \n {} '.format(data_1['trip_duration'].nlargest(5)))
print('The the number of rows with 1 as their trip duration values is {}'.format(len(data_1[data_1['trip_duration']==1 ])))


# In[5]:


print('The value of largest 5 trip duration values are as follows : \n {} '.format(data_1['trip_duration'].nlargest(5)))
print('The the number of rows with 1 as their trip duration values is {}'.format(len(data_1[data_1['trip_duration']==1 ])))


# In[6]:


data_1['trip_duration_hour']=data_1['trip_duration']/3600 


# In[7]:


data_1.passenger_count.value_counts()


# In[8]:


data_1=data_1[data_1.passenger_count<=6]
data_1=data_1[data_1.passenger_count!=0]


# In[9]:


data_1['dropoff_datetime']=pd.to_datetime(data_1['dropoff_datetime'])
data_1['pickup_datetime']=pd.to_datetime(data_1['pickup_datetime'])


# In[10]:


data_1['pickup_day']=data_1['pickup_datetime'].dt.day_name()
data_1['dropoff_day']=data_1['dropoff_datetime'].dt.day_name()
data_1['pickup_month']=data_1['pickup_datetime'].dt.month
data_1['dropoff_month']=data_1['dropoff_datetime'].dt.month


# In[11]:


data_1['pickup_month'].value_counts()


# In[12]:


data_1['dropoff_month'].value_counts()


# In[13]:


print(data_1[data_1.dropoff_month==7].pickup_datetime.dt.month.value_counts())
print(data_1[data_1.dropoff_month==7].pickup_datetime.dt.day.value_counts())


# In[14]:


import csv
import math


# In[15]:


def haversine(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2=map(math.radians,[lon1,lat1,lon2,lat2])
    dlon=lon2-lon1
    dlat=lat2-lat1
    a=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c=2*math.asin(math.sqrt(a))
    km=6367*c
    return km


# In[16]:


data_1['distance'] = data_1.apply(lambda x: haversine(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude'] ), axis=1)


# In[17]:


sns.scatterplot(x='distance',y='trip_duration',data=data_1)


# In[18]:


print('The no of rows with distance =0 are {}'.format(len(data_1[data_1.distance==0])))


# In[19]:


mean_dist=data_1['distance'].mean()
data_1.loc[data_1['distance']==0,'distance']=mean_dist


# In[20]:


data_1['speed']=data_1['distance']/data_1['trip_duration_hour']
sns.boxplot(data_1['speed'])


# In[21]:


sns.scatterplot(x='distance',y='trip_duration_hour',data=data_1)


# In[22]:


data_1['log_distance']=np.log(data_1.distance)
data_1['log_trip_duration']=np.log(data_1.trip_duration_hour)
sns.scatterplot(x='log_distance',y='log_trip_duration',data=data_1)


# In[23]:


data_1=data_1[data_1.log_trip_duration<2]


# In[24]:


data_1.columns


# In[25]:


#distribution of timezones in morning,afternoon,evening and night
def time_of_day(x):
    if x in range(6,12):
        return 'Morning'
    elif x in range(12,16):
        return 'Afternoon'
    elif x in range(16,22):
        return 'Evening'
    else:
        return 'night'


# In[26]:


data_1['pick_up_hour']=data_1['pickup_datetime'].dt.hour
data_1['drop_off_hour']=data_1['dropoff_datetime'].dt.hour


# In[27]:


data_1['pick_up_timezone']=data_1['pick_up_hour'].apply(time_of_day)
data_1['drop_off_timezone']=data_1['drop_off_hour'].apply(time_of_day)


# In[28]:


data_1.columns


# In[29]:


data_1.reset_index()


# In[30]:


data_2=data_1.drop(['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime','pickup_longitude', 'pickup_latitude',
             'trip_duration_hour','log_trip_duration','dropoff_longitude','drop_off_hour', 'dropoff_latitude' ,'pick_up_hour',],axis=1)


# In[31]:


data_2.head()


# In[32]:


data2=pd.get_dummies(data_2,columns=['store_and_fwd_flag','pickup_day','dropoff_day','pickup_month','dropoff_month','pick_up_timezone', 'drop_off_timezone'])


# In[33]:


base_line_col=['distance']
predictor_cols=['passenger_count','distance','store_and_fwd_flag_N','store_and_fwd_flag_Y',
               'pickup_day_Friday','pickup_day_Monday','pickup_day_Saturday','pickup_day_Sunday',
               'pickup_day_Thursday','pickup_day_Tuesday','pickup_day_Wednesday','dropoff_day_Friday',
               'dropoff_day_Monday','dropoff_day_Saturday','dropoff_day_Sunday','dropoff_day_Thursday',
               'dropoff_day_Tuesday','dropoff_day_Wednesday','pickup_month_1','pickup_month_5','pickup_month_6',
               'dropoff_month_1','dropoff_month_5','dropoff_month_6','pickup_timezone_late night',
               'pick_up_timezone_midday','pick_up_timezone_morning','drop_off_timezone_evening',
               'drop_off_timezone_late night','drop_off_timezone_midday','drop_off_timezone_morning']
target_col=['trip_duration']


# In[34]:


from sklearn import  metrics
from sklearn.model_selection import cross_val_score
def modelfit(estimator,data_train,data_test,predictors,target):
    #print(data_train.head())
    #fitting model
    estimator.fit(data_train[predictors],data_train.loc[:,target])
    #train data prediction
    train_pred=estimator.predict(data_train[predictors])
    #cross_validation score
    cv_score=cross_val_score(estimator,data_train[predictors],data_train.loc[:,target],cv=20,scoring='neg_mean_squared_error')
    
    cv_score=np.sqrt(np.abs(cv_score))
    #Print model report:
    print ("\nModel Report")
    print ("RMSE on Train Data: %.4g" % np.sqrt(metrics.mean_squared_error(data_train.loc[:,target].values, train_pred)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    test_pred=estimator.predict(data_test[predictors])
    print ("RMSE on Test Data: %.4g" % np.sqrt(metrics.mean_squared_error(data_test.loc[:,target].values, test_pred)))


# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
validation_size = 0.20
seed = 7
X_train, X_test = train_test_split(data2,test_size=validation_size, random_state=seed)


# In[38]:


import math


# In[39]:


mean_pred=np.repeat(X_train[target_col].mean(),len(X_test[target_col]))
from sklearn.metrics import mean_squared_error as mae
math.sqrt(mae(X_test[target_col],mean_pred))


# In[40]:


alg1 = LinearRegression(normalize=True)
print('The baseline model')
y_pred=modelfit(alg1, X_train, X_test,base_line_col,target_col)
coef1 = alg1.coef_
print('The coeffient is {}'.format(coef1))


# In[ ]:




