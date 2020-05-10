# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:09:56 2020

@author: User
"""
#https://www.kaggle.com/furiousx7/xgboost-time-series

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import imageio
from statsmodels.graphics.tsaplots import plot_acf
import datetime


import os
import pandas as pd
path = os.getcwd()


print(path)
#%%
#base = datetime.datetime.today()
#date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]

data = pd.read_csv('sales_train_validation.csv')
calendar = pd.read_csv('calendar.csv')
#testdata = pd.read_csv('sales_test_validation.csv')
rows,columns = data.shape
productnr = 4
data=data.iloc[productnr,5:columns]
data = data.to_frame()
data = data.set_index(pd.to_datetime(calendar.iloc[0:1913,0]))
columns = data.size

testsize=28 #days
train = data.iloc[0:columns-testsize]
test = data.iloc[columns-testsize:columns]

print('done')



#%%
# =============================================================================
# plt.figure(figsize=(20,5))
# plt.plot(train.iloc[i, 0:365])
# plt.show
# 
# =============================================================================

#%%



def create_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X

#%%
X_train, y_train = create_features(train), train[productnr]
X_test, y_test   = create_features(test), test[productnr]

X_train.shape, y_train.shape

#%%

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
        verbose=True) # Change verbose to True if you want to see it train


xgb.plot_importance(reg, height=0.9)

#%%
def plot_performance(base_data, date_from, date_to, title=None):
    plt.figure(figsize=(15,3))
    if title == None:
        plt.title('From {0} To {1}'.format(date_from, date_to))
    else:
        plt.title(title)
    plt.xlabel('time')
    plt.ylabel('sales')
    plt.plot(data.index,data, label='data')
    plt.plot(X_test.index,X_test_pred, label='prediction')
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    
X_test_pred = reg.predict(X_test)
    
plot_performance(data, data.index[0].date(), data.index[-1].date(),
                 'Original and Predicted Data')

plot_performance(y_test, y_test.index[0].date(), y_test.index[-1].date(),
                 'Test and Predicted Data')

plot_performance(y_test, '01-01-2016', '24-04-2016', 'Prediction')

plt.legend()

plt.show()








