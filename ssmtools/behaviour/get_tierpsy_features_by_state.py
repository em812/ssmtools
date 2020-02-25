#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to get tierpsy features by state for a specific worm trajectory

Created on Thu Jun  6 14:09:55 2019

@author: em812
"""
import pandas as pd
import numpy as np

def get_tierpsy_feat_by_state(
        features_file, worm_id, time_start, time_end, smoothed_z, K,
        tierpsy_timeseries=None, stats=['mean','percentiles']):
    
    if tierpsy_timeseries is None:
        tierpsy_timeseries = [
                'speed', 'angular_velocity', 'path_curvature_midbody',
                'relative_to_body_angular_velocity_midbody',
                'curvature_midbody'
                ] #,'d_speed','d_angular_velocity']
    
    mask_in_state = {}
    for k in range(K):
        mask_in_state[k] = np.isin(smoothed_z,k)
        
    state_features = {}
    for k in range(K):
        state_features[k] = []
    
    timeseries = pd.read_hdf(features_file, 'timeseries_data', mode='r')
    
    if isinstance(tierpsy_timeseries,str):
        if tierpsy_timeseries=='all':
            tierpsy_timeseries = timeseries.columns.difference(['worm_index','timestamp'])
            tierpsy_timeseries = [col for col in tierpsy_timeseries if 'coord_' not in col]
        else:
            tierpsy_timeseries = [tierpsy_timeseries
                                  ]
    timeseries = timeseries.loc[(timeseries['worm_index']==worm_id) & 
                                (timeseries['timestamp']>=time_start) &
                                (timeseries['timestamp']<=time_end),
                                tierpsy_timeseries]
    
    for k in range(K):
        assert timeseries.shape[0] == mask_in_state[k].shape[0]
        # Mean
        if 'mean' in stats:
            key = '_mean'
            state_feat = pd.DataFrame(timeseries.loc[mask_in_state[k],:].mean(axis=0)).T
            state_feat.reset_index(drop=True,inplace=True)
            state_feat.columns = [col+key for col in state_feat.columns]
            state_features[k].append(state_feat)
        # Percentiles
        if 'percentiles' in stats:
            percentiles = [10,50,90]
            for perc in percentiles:
                key = '_{}th'.format(perc)
                state_feat = pd.DataFrame(timeseries.loc[mask_in_state[k],:].quantile(perc/100,axis=0)).T
                state_feat.reset_index(drop=True,inplace=True)
                state_feat.columns = [col+key for col in state_feat.columns]
                state_features[k].append(state_feat)
        state_features[k] = pd.concat(state_features[k],axis=1)
    
        
    return state_features
    
    
if __name__=="__main__":
    pass