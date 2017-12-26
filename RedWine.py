# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:59:01 2017

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt

df= pd.read_csv('http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
print(df.head())
print(df.tail())
print(df.describe())
y= df.quality
x=df.drop('quality', axis=1)
print(y)
print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=123,stratify=y)
print(df.shape)
scaler= preprocessing.StandardScaler().fit(x_train)
x_train_scaled= scaler.transform(x_train)
print(x_train_scaled.mean(axis=0))
print(x_train_scaled.std(axis=0))

x_test_scaled= scaler.transform(x_test)
print(x_test_scaled.mean(axis=0))
print(x_test_scaled.std(axis=0))
#print(x_train)
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf=GridSearchCV(pipeline,hyperparameters, cv=10)
clf.fit(x_train, y_train)
print(clf.best_params_)
y_predict= clf.predict(x_test)
print(r2_score(y_test,y_predict))
print(mean_squared_error(y_test, y_predict))
joblib.dump(clf, 'rf_regressor.pkl')

plt.scatter(y_test, y_predict)
plt.xlabel(True)
plt.ylabel(y_predict)

