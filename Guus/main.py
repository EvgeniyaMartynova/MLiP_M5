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
#This is where the features are created
def create_features(df,productnr,rolledmean7,rolledmean28, laggedsales7,laggedsales28):
    # add lag vectors
    #lags = [7, 28]
    #lag_cols = [f"lag_{lag}" for lag in lags]
   # for lag, lag_col in zip(lags, lag_cols):
    #    df[lag_col] = df.loc[:,productnr].shift(lag).astype("float32")
    
        
    #get the relevant shifted sales
    startdate=df.index[0]
    startindex=laggedsales7.index.get_loc(startdate)
    enddate=df.index[df.shape[0]-1]
    endindex=laggedsales7.index.get_loc(enddate)+1
    #make new feature
    colname='lag_7'
    df[colname]=laggedsales7.iloc[startindex:endindex,0]
    
    
    #get the relevant shifted sales
    startdate=df.index[0]
    startindex=laggedsales28.index.get_loc(startdate)
    enddate=df.index[df.shape[0]-1]
    endindex=laggedsales28.index.get_loc(enddate)+1
    #make new feature
    colname='lag_28'
    df[colname]=laggedsales28.iloc[startindex:endindex,0]
    

        
    #get the relevant shifted sales
    startdate=df.index[0]
    startindex=rolledmean7.index.get_loc(startdate)
    enddate=df.index[df.shape[0]-1]
    endindex=rolledmean7.index.get_loc(enddate)+1
    #make new feature
    colname='rollingmean_7'
    df[colname]=rolledmean7.iloc[startindex:endindex,0]
    #get the relevant shifted sales
    startdate=df.index[0]
    startindex=rolledmean28.index.get_loc(startdate)
    enddate=df.index[df.shape[0]-1]
    endindex=rolledmean28.index.get_loc(enddate)+1
    #make new feature
    colname='rollingmean_28'
    df[colname]=rolledmean28.iloc[startindex:endindex,0]
        
    
    """
    # rolling features on lag_cols
    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            df[f"rmean_{lag}_{win}"] = df[["productnr", lag_col]].groupby("productnr")[lag_col].transform(lambda x : x.rolling(win).mean())
    """
    # time features
    df['date'] = pd.to_datetime(df.index)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    df=df.drop(['date'], axis=1)
    X = df[['dayofweek','month', 'year','dayofyear','dayofmonth','weekofyear','lag_7','lag_28', 'rollingmean_7','rollingmean_28']]#,
    #X = df[['dayofweek','quarter','month', 'year','dayofyear','dayofmonth','weekofyear','lag_7','lag_28']]#,
    
    X=X.fillna(0)
            #'lag_7', 'lag_28']]#, 'rmean_7_7', 'rmean_7_28', 'rmean_7_28', 'rmean_28_28']]
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
    #plt.plot(base_data.index,product, label='data')
    plt.plot(base_data.index,X_test_pred, label='prediction')
    plt.legend()
    plt.xlim(left=date_from, right=date_to)
    return

#%%
#This is where the model is created and predictions are made
def do_predictions(productnr):
    
    # make a start on rolling mean/lag feature creation, as this only has to be done once and in create_features it'll be redundant
    rolledmean=product
    #copy date-indices
    index=pd.date_range(start=calendar.iloc[product.shape[0],0], end= calendar.iloc[calendar.shape[0]-1,0])
    #get correct columname because pandas sucks and doesnt correctly append even if you give the exact location because "columnames" are more important than that
    columns=[productnr]
    #extend the df so we don't lose data when we shift
    empty = pd.DataFrame(index=index, columns=columns)
    rolledmean = rolledmean.append(empty, ignore_index=False)
    #compute the mean with a window
    rolledmean7 = rolledmean.rolling(7).sum()
    rolledmean28 = rolledmean.rolling(28).sum()
    #shift it with a year
    #rolledmean7 = rolledmean7.shift(365)
    #rolledmean28 = rolledmean28.shift(365)
    #placeholder storing sales to be shifted
    shift=product
    #copy date-indices
    index=pd.date_range(start=calendar.iloc[product.shape[0],0], end= calendar.iloc[calendar.shape[0]-1,0])
    #get correct columname because pandas sucks and doesnt correctly append even if you give the exact location because "columnames" are more important than that
    columns=[productnr]
    #extend the df so we don't lose date when we shift
    empty = pd.DataFrame(index=index, columns=columns)
    shift = shift.append(empty, ignore_index=False)
    #shift with the specified lag
    laggedsales7=shift.shift(7)
    laggedsales28=shift.shift(28)
    
    
    #create features 
    X_train, y_train = create_features(train,productnr,rolledmean7,rolledmean28, laggedsales7,laggedsales28), train[productnr]
    X_test, y_test   = create_features(test,productnr,rolledmean7,rolledmean28, laggedsales7,laggedsales28), test[productnr]
    #with GPU
    #params = {"objective": "count:poisson","learning_rate" : 0.075,"max_depth": 6,'n_estimators': 200,'min_child_weight': 50,"tree_method": 'gpu_hist', "gpu_id": 0}
    #without GPU
    params = {'n_estimators': 200,"learning_rate" : 0.075,"max_depth": 6, 'min_child_weight': 50}
    reg = xgb.XGBRegressor(**params)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=10, #stop if 50 consequent rounds without decrease of error
            verbose=False) # Change verbose to True if you want to see it train

    #xgb.plot_importance(reg, height=0.9)
    
    X_test_pred = reg.predict(X_test)
    print(str(productnr)+" : "+str(reg.best_score))
    
    predictionfeatures1 = create_features(p1,productnr,rolledmean7,rolledmean28, laggedsales7,laggedsales28)
    predictionfeatures2 = create_features(p2,productnr,rolledmean7,rolledmean28, laggedsales7,laggedsales28)
    pred1 = reg.predict(predictionfeatures1)
    pred2 = reg.predict(predictionfeatures2)
    plt.figure(figsize=(15,3))
    plt.plot(y_test.index, test2, label='data')
    plt.plot(y_test.index, pred1, label='test')
    plt.show()
    
   # plot_performance(product, product.index[0].date(), product.index[-1].date(),
   #                  'Original and Predicted Data')

# =============================================================================
#     plot_performance(X_test_pred, y_test.index[0].date(), y_test.index[-1].date(),
#                      'Test and Predicted Data')
#     
#     plot_performance(y_test, '01-01-2016', '24-04-2016', 'Prediction')
# =============================================================================
    
   # plt.legend()
    
    

    return pred1,pred2, X_test_pred, reg.best_score

#%%
#This is kind of the main code I used to read in data and do predictions. sorry for the mess :)
#base = datetime.datetime.today()
#date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]

#numcols can be modified to use less days
numcols = [f"d_{day}" for day in range(1,1914)]
catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
dtype = {numcol:"float32" for numcol in numcols} 
#data = pd.read_csv('sales_train_validation.csv',usecols = catcols + numcols, dtype = dtype)
data= pd.read_csv('sales_train_evaluation.csv')
calendar = pd.read_csv('calendar.csv')
testsize=28 #days
rows,datacolumns = data.shape
#create empty dataframe for predictions
testpredictions = pd.DataFrame(np.zeros((rows, testsize)))
prediction1 = pd.DataFrame(np.zeros((rows, 28)))
prediction2 = pd.DataFrame(np.zeros((rows, 28)))
finalprediction = pd.DataFrame(np.zeros(((rows)*2, 28)))

#create dataframe for storing the RMSE score for each product
scores = pd.DataFrame(np.zeros((rows, 1)))

#j=0
#%%
#loop for training and prediction for each product (0,rows) for all products
#changed to row from row-1, python range has [,) interval
for productnr in range(rows-1001,rows):
    #productnr = 4
    
    #if(data.iloc[productnr,3]=="FOODS" and j<1000):
    #test2=data2.iloc[productnr,datacolumns:data2.shape[1]]    
    product=data.iloc[productnr,6:datacolumns]
    p1=pd.DataFrame(np.zeros((28,1)))
    #p2=pd.DataFrame(np.zeros((28,1)))
    columns = product.size
    product = product.to_frame()
    
    product = product.set_index(pd.to_datetime(calendar.iloc[0:columns,0]))
    p1 = p1.set_index(pd.to_datetime(calendar.iloc[columns:columns+testsize,0]))
    #p2 = p2.set_index(pd.to_datetime(calendar.iloc[columns+testsize:calendar.shape[0]+1,0]))

    train = product.iloc[0+28:columns-testsize]
    test = product.iloc[columns-testsize:columns]
    
    #why do you add testpredictions? it seems like it's not used
    prediction1.iloc[productnr], prediction2.iloc[productnr], testpredictions.iloc[productnr], scores.iloc[productnr]=do_predictions(productnr)
#    j=j+1
    #else:
    #    print(data.iloc[productnr,3])
print('Mean RSME: '+str(scores.mean(axis=0)))
finalprediction = pd.concat([prediction1,prediction2], ignore_index=True)
# =============================================================================
# last=prediction2.iloc[30489,:]
# for i in range(0,27):
#     finalprediction.iloc[60979,i]=last.iloc[i]
# =============================================================================
    

#%%
submission = pd.read_csv('sample_submission.csv')
columnnames = submission.columns[1:29]
finalprediction.set_axis(columnnames,axis=1, inplace=True)
submission.update(finalprediction)
submission.to_csv('submission.csv', index=False)
#%%
 #finalprediction = finalprediction.rename(columns={"0": "F1", "1": "F2","2": "F3","3": "F4","4": "F5","5": "F6","6": "F7","7": "F8","8": "F9","9": "F10","10": "F11","11": "F12","12": "F13","13": "F14","14": "F15","15": "F16","16": "F17","17": "F18","18": "F19","19": "F20","20": "F21","21": "F22","22": "F23","23": "F24","24": "F25","25": "F26","26": "F27","27": "F28"})