# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:47:04 2019

@author: 16703
"""
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
import glob
import numpy as np
import pandas as pd
from catboost import  CatBoostClassifier
from sklearn.metrics import f1_score
import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import time
from sklearn.model_selection import train_test_split
import os
train = pd.read_csv('./train.csv', header=None ,sep =';')
y = train.iloc[629]
train.columns = y.values.tolist()
Y = train["quality"]
X = train.drop(labels = ["quality"],axis = 1)
X = X.drop(['density'],axis=1)
# free some space
del train
X = X.drop(629).values
#from sklearn.preprocessing import scale
#X = scale(X)

Y = Y.drop(629).astype(np.int64).values
Y = Y-3
result_test = pd.read_csv('./test.csv', header=None,sep =';')

def unique_count(index_col, feature, df_data):
    if isinstance(index_col, list):
        name = "{0}_{1}_nq".format('_'.join(index_col), feature)
    else:
        name = "{0}_{1}_nq".format(index_col, feature)
    print(name)
    gp1 = df_data.groupby(index_col)[feature].nunique().reset_index().rename(
        columns={feature: name})
    df_data = pd.merge(df_data, gp1, how='left', on=[index_col])
    return df_data.fillna(0)
    #return df_data


print(X.shape,Y.shape)
#X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1, random_state=42)
#oof = np.zeros(X_train.shape[0])
#prediction = np.zeros(X_test.shape[0])
seeds = [19941227, 2019 * 2 + 1024, 4096, 2048, 1024]
num_model_seed = 1
gc.collect()
prediction_cat=np.zeros((result_test.shape[0],7))
skf = StratifiedKFold(n_splits=5, random_state=seeds[4], shuffle=True)
for train,test in skf.split(X, Y):
    #print(index)
    train_x, test_x, train_y, test_y = X[train], X[test], Y[train], Y[test]
    cbt_model = CatBoostClassifier(iterations=3000,learning_rate=0.01,max_depth=7,verbose=100,loss_function='MultiClass',
                                   early_stopping_rounds=500,task_type='CPU',eval_metric='Accuracy',max_ctr_complexity=4)    
    cbt_model.fit(train_x, train_y,eval_set=(test_x,test_y))
    del train_x, test_x, train_y, test_y
    prediction_cat += cbt_model.predict_proba(result_test)/5
    
#def print_best_score(gsearch,param_test):
     # 输出best score
#    print("Best score: %0.3f" % gsearch.best_score_)
#    print("Best parameters set:")
#    # 输出最佳的分类器到底使用了怎样的参数
#    best_parameters = gsearch.best_estimator_.get_params()
#    for param_name in sorted(param_test.keys()):
#        print("\t%s: %r" % (param_name, best_parameters[param_name]))    
#params = {'depth': [4, 7, 10],
#          'learning_rate' : [0.03, 0.1, 0.15,0.5],
#         'l2_leaf_reg': [1,4,9],
#         'iterations': [3000]}

#estimator =CatBoostClassifier(iterations=2000,verbose=400,early_stopping_rounds=400,
#                                        loss_function='MultiClass',task_type='CPU',eval_metric='Accuracy')
#cbt_model = GridSearchCV(estimator, param_grid = params, scoring="accuracy", cv = 3)
    
#cbt_model.fit(X_train, y_train,eval_set=(X_test,y_test))
#print_best_score(cbt_model,params)
#prediction_cat = cbt_model.predict(result_test)[:,0]

# write to csv
prediction = np.argmax(prediction_cat, axis=1)
print(prediction)
results = pd.Series(prediction)
submission = pd.concat([pd.Series(range(1,1501)),results+3],axis=1)
submission.to_csv('./submission.csv',index=False,header=None)
