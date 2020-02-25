#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:53:38 2019

@author: em812
"""
import numpy as np

def cross_validation(X,c,estimator,splitter,scale=True):
    from sklearn.metrics import accuracy_score
    
    c_pred = np.empty_like(c)
    
    for train_index, test_index in splitter.split(X, c):
        X_train,X_test = X[train_index],X[test_index]
        c_train = c[train_index] #,c_test,c[test_index]
        
        # Normalize
        if scale:
            X_train = (X_train-np.mean(X_train,axis=0))/np.nanstd(X_train,axis=0)
            X_test = (X_test-np.mean(X_train,axis=0))/np.nanstd(X_train,axis=0)
        
        # Train classifier
        estimator.fit(X_train,c_train)
        
        # Predict 
        c_pred[test_index] = estimator.predict(X_test)
        
    cv_accuracy = accuracy_score(c_pred,c)
    estimator.fit(X,c)
    
    return estimator,cv_accuracy


def grid_search_hyperparameters(X,c,estimator,param_grid,splitter,scale=True,refit=True):
    from sklearn.model_selection import GridSearchCV
    
    if scale:
        X = (X-np.mean(X,axis=0))/np.nanstd(X,axis=0)
    
    grid_search = GridSearchCV(estimator,param_grid,cv=splitter,n_jobs=-1,refit=refit)
    
    grid_search.fit(X,c)
    
    return grid_search.best_estimator_,grid_search.best_score_
    