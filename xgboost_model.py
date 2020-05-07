import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

input_dir = os.path.join(os.path.abspath(os.getcwd()),'input')
sales_train_validation_csv = os.path.join(input_dir,'sales_train_validation.csv')
sales_train_val = pd.read_csv(sales_train_validation_csv)

data = pd.DataFrame(sales_train_val)
#drop all the columns that aren't the ids of the different objects and aren't the days
data = data.drop(columns=['item_id','dept_id','cat_id','store_id','state_id'])
# i thought transposing it will be the solution, but it doesn't work like this either
data = data.transpose()
print(data.head())

train, val = train_test_split(data, test_size=0.2, random_state=1)

X_train, y_train = train.drop(columns=['id']), train.id
X_val, y_val = val.drop(columns=['id']), val.id
print(X_train.head())

model = xgb.XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=20)

model.fit(X_train, y_train, eval_metric="rmse")
preds = model.predict(X_val)
print(preds)
