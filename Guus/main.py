# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:09:56 2020

@author: User
"""
#https://www.kaggle.com/furiousx7/xgboost-time-series

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import xgboost as xgb
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
    predictionfeatures1 = create_features(p1)
    predictionfeatures2 = create_features(p2)
    X_train.shape, y_train.shape
    



    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
            verbose=False) # Change verbose to True if you want to see it train


    #xgb.plot_importance(reg, height=0.9)
    
    
    
        
    X_test_pred = reg.predict(X_test)
    print(str(productnr)+" : "+str(reg.best_score))   
    pred1 = reg.predict(predictionfeatures1)
    pred2 = reg.predict(predictionfeatures2)
    #print(pred1)
    
    #plot_performance(data, data.index[0].date(), data.index[-1].date(),
    #                 'Original and Predicted Data')
    
   # plot_performance(y_test, y_test.index[0].date(), y_test.index[-1].date(),
    #                 'Test and Predicted Data')
    
    #plot_performance(y_test, '01-01-2016', '24-04-2016', 'Prediction')
    
    #plt.legend()
    
   # plt.show()

    return pred1,pred2, X_test_pred, reg.best_score

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
testpredictions = pd.DataFrame(np.zeros((rows, testsize)))
prediction1 = pd.DataFrame(np.zeros((rows, 28)))
prediction2 = pd.DataFrame(np.zeros((rows, 28)))
finalprediction = pd.DataFrame(np.zeros(((rows)*2, 28)))

#create dataframe for storing the RMSE score for each product
scores = pd.DataFrame(np.zeros((rows-1, 1)))


#%%
#loop for training and prediction for each product (0,rows) for all products
#changed to row from row-1, python range has [,) interval
for productnr in range(0, rows):
    #productnr = 4
    product=data.iloc[productnr,6:datacolumns]
    p1=pd.DataFrame(np.zeros((28,1)))
    p2=pd.DataFrame(np.zeros((28,1)))
    columns = product.size
    product = product.to_frame()
    
    product = product.set_index(pd.to_datetime(calendar.iloc[0:columns,0]))
    p1 = p1.set_index(pd.to_datetime(calendar.iloc[columns:columns+testsize,0]))
    p2 = p2.set_index(pd.to_datetime(calendar.iloc[columns+testsize:calendar.shape[0]+1,0]))

    train = product.iloc[0:columns-testsize]
    test = product.iloc[columns-testsize:columns]
    
    
    prediction1.iloc[productnr], prediction2.iloc[productnr], testpredictions.iloc[productnr], scores.iloc[productnr]=do_predictions(productnr)
    #print(scores.iloc[productnr])
    
print('Mean RSME: '+str(scores.mean(axis=0)))
finalprediction = pd.concat([prediction1,prediction2], ignore_index=True)


#%%
submission = pd.read_csv('sample_submission.csv')
columnnames = submission.columns[1:29]
finalprediction.set_axis(columnnames,axis=1, inplace=True)
submission.update(finalprediction)
submission.to_csv('submission.csv')
#%%
 #finalprediction = finalprediction.rename(columns={"0": "F1", "1": "F2","2": "F3","3": "F4","4": "F5","5": "F6","6": "F7","7": "F8","8": "F9","9": "F10","10": "F11","11": "F12","12": "F13","13": "F14","14": "F15","15": "F16","16": "F17","17": "F18","18": "F19","19": "F20","20": "F21","21": "F22","22": "F23","23": "F24","24": "F25","25": "F26","26": "F27","27": "F28"})