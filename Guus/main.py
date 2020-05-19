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


import sys, os
import pandas as pd

from contextlib import contextmanager


#suppress output of certain code by using 'with suppress_stdout(): and an inline' 
#example:
#print "You can see this"
#with suppress_stdout():
#    print "You cannot see this"
#print "And you can see this again"
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


path = os.getcwd()




print(path)



#%%
# =============================================================================
# plt.figure(figsize=(20,5))
# plt.plot(train.iloc[i, 0:365])
# plt.show
# 
# =============================================================================

#%%
#This is where the features are created, I think we can leave out hour


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
#This is the plotting functions that creates the nice plots of your 
#predictions and how much they rely on which feature. don't do this for all 
#products! Your pc will freeze
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
    return

#%%
#This is where the model is created and predictions are made
def do_predictions(productnr):
    
    X_train, y_train = create_features(train), train[productnr]
    X_test, y_test   = create_features(test), test[productnr]
    
    X_train.shape, y_train.shape
    



    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
            verbose=False) # Change verbose to True if you want to see it train


    #xgb.plot_importance(reg, height=0.9)
    
    
    
        
    X_test_pred = reg.predict(X_test)
    print(str(productnr)+" : "+str(reg.best_score))   
    #plot_performance(data, data.index[0].date(), data.index[-1].date(),
    #                 'Original and Predicted Data')
    
   # plot_performance(y_test, y_test.index[0].date(), y_test.index[-1].date(),
    #                 'Test and Predicted Data')
    
    #plot_performance(y_test, '01-01-2016', '24-04-2016', 'Prediction')
    
    #plt.legend()
    
   # plt.show()

    return X_test_pred, reg.best_score

#%%
#This is kind of the main code I used to read in data and do predictions. sorry for the mess :)
#base = datetime.datetime.today()
#date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]

data = pd.read_csv('sales_train_validation.csv')
calendar = pd.read_csv('calendar.csv')
#testdata = pd.read_csv('sales_test_validation.csv')
testsize=28 #days
rows,datacolumns = data.shape
#create empty dataframe for predictions
predictions = pd.DataFrame(np.zeros((rows-1, testsize)))
#create dataframe for storing the RMSE score for each product
scores = pd.DataFrame(np.zeros((rows-1, 1)))


#%%
for productnr in range(2, 1000):
    #productnr = 4
    product=data.iloc[productnr,5:datacolumns]
    columns = product.size
    product = product.to_frame()
    
    product = product.set_index(pd.to_datetime(calendar.iloc[0:columns,0]))
    
    
    print('hoi'+str(productnr))

    train = product.iloc[0:columns-testsize]
    test = product.iloc[columns-testsize:columns]
    
    predictions.iloc[productnr], scores.iloc[productnr]=do_predictions(productnr)
    #print(scores.iloc[productnr])
    
print('Mean RSME: '+str(scores.mean(axis=0)))


#%%
rows,columns = data.shape
product=data.iloc[380,5:columns]
columns = product.size
product = product.to_frame()

product = product.set_index(pd.to_datetime(calendar.iloc[0:columns,0]))

plt.plot(product)