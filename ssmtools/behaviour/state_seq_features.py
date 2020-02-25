#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:39:53 2019

@author: em812
"""
import numpy as np
import pandas as pd
import pdb

def state_duration(inferred_z,states,return_dataframe=True):
    
    #pdb.set_trace()
    state_dur = np.zeros(states.shape)
    
    unq_states,counts = np.unique(inferred_z,return_counts=True)
    
    for i in range(len(unq_states)):
        state_dur[states==unq_states[i]]=counts[i]
    
    state_dur = state_dur/len(inferred_z)
    
    if np.sum(np.isnan(state_dur)):
        pdb.set_trace()
            
    if return_dataframe:
        state_dur = pd.DataFrame(state_dur.reshape(1,-1),columns = ['duration_in_state_{}'.format(i) for i in states])
        
    if np.any(state_dur>1e10):
        pdb.set_trace()
        
    return state_dur


def transition_probas(inferred_z,n_states,return_dataframe=True):
    
    #create matrix of zeros
    trans = np.zeros((n_states,n_states))
    
    for (i,j) in zip(inferred_z,inferred_z[1:]):
        trans[i,j] += 1
        
    #now convert to probabilities:
    trans_probas = np.divide(trans.T,np.sum(trans,axis=1)).T
    trans_probas[np.sum(trans,axis=1)==0,:] = 0.0
    
    if np.sum(np.isnan(trans_probas)):
        pdb.set_trace()
    
    if return_dataframe:
        trans_probas = pd.DataFrame(np.array([trans_probas[i,j] for i in range(n_states) for j in range(n_states)]).reshape(1,-1), 
                                     columns=['transition_proba_{}_{}'.format(i,j) for i in range(n_states) for j in range(n_states)])
            
    return trans_probas

def extract_state_seq_features(model,y,K,stateDuration=True,transitionProbas=True):
    
    z = []
    features = pd.DataFrame()
    
    for iworm,yworm in enumerate(y):
        smoothed_z = model.most_likely_states(yworm)
        feat_point = pd.DataFrame()
        if stateDuration:
            duration = state_duration(smoothed_z,np.arange(K))
            feat_point = pd.concat((feat_point,duration),axis=1)
        if transitionProbas:
            transition = transition_probas(smoothed_z,K)
            feat_point = pd.concat((feat_point,transition),axis=1)
        
        z.append(smoothed_z)
        features = pd.concat((features,feat_point),axis=0)
        
    features.reset_index(drop=True,inplace=True)
    
    return features,z

if __name__ == "__main__":
    
    z = [1,1,2,6,8,5,5,7,8,8,1,1,4,5,5,0,0,0,1,1,4,4,5,1,3,3,4,5,4,1,1]
    n_states = len(np.unique(z))
    trans_probas = transition_probas(z,n_states,return_dataframe=True)
    state_dur = state_duration(z,np.arange(n_states),return_dataframe=True)
