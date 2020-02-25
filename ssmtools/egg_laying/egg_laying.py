#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:23:19 2020

@author: em812
"""
import pandas as pd
import numpy as np


def slice_long_sequence(event_times, total_time, slice_size=900, 
                        fillmissing=False):
    """
    Takes a sequence of time events recorded in a long video and get the
    time intervals between the events in case the video was sliced in equal
    time slices.
    """
    from math import ceil
    
    n_slices = ceil(total_time/slice_size)
    
    # If there are no events recorded, return nans or slice_size
    if event_times is np.nan:
        if fillmissing:
            sliced_intervals = [[slice_size]]*n_slices
            sliced_intervals[-1] = total_time - slice_size*(n_slices-1)
        else:
            sliced_intervals = [[np.nan]]*n_slices
        return sliced_intervals
    
    # Make breakpoints at slice_size intervals
    add_breakpoints = np.arange(slice_size,total_time,slice_size)
    # Add 0 and total time
    add_breakpoints = np.append(add_breakpoints,total_time)
    add_breakpoints = np.insert(add_breakpoints,0,0.0)
    
    # Sort real event_times and breakpoints
    _event_times = np.sort(np.append(event_times,add_breakpoints))
    
    # Slice _event_times at breakpoints and get intervals
    sliced_intervals = []
    for slc in range(n_slices):
        limstart = slice_size*slc
        limend = slice_size*(slc+1)
        sliced_events = [
                x for x in _event_times if (x>=limstart) and (x<=limend)]
        if (not fillmissing) and (len(sliced_events)==2):
            sliced_intervals.append(np.nan*np.ones(1))
        else:
            sliced_intervals.append(np.diff(sliced_events))
    
    # Make sure the sliced intervals are more than the initial intervals    
    assert np.sum([len(x) for x in sliced_intervals]) >= len(event_times)+1
    
    return sliced_intervals

def slice_long_videos(eggs,slice_size,fillmissing,remove_nan=False):
    """
    Takes a dataframe with rows corresponding to long videos and columns to 
    egg laying event times (events_time) and total length of video (total_time)
    and slices the video in equally sized parts. It outputs the intervals 
    between egg laying events in the sliced videos.
    Input:
        eggs : dataframe, must contain 'events_time' and 'total_time' columns.
        slice_size : size of sliced videos
        fillmissing: if False, a slice that does not contain any events
                will be ignored (removed from data). If True, a slice with no
                events will be stored as an interval of size = slice_size.
        remove_nan : if True, nan values will be removed from the sliced eggs.
    Return:
        eggs_sliced: list of lists of interval sequences 
                (one interval sequence per slice).
    """
    eggs_sliced = [slice_long_sequence(event_times, total_time, 
                                       slice_size=slice_size, 
                                       fillmissing=fillmissing)
                    for indx,(event_times,total_time) in 
                    eggs[['events_time', 'total_time']].iterrows()]
    
    # remove any nans from eggs_sliced
    if remove_nan:
        _eggs_sliced = []
        for video_list in eggs_sliced:
            _eggs_sliced.append(
                    [x for x in video_list if np.sum(np.isnan(x))==0])
        eggs_sliced = _eggs_sliced
    
    return eggs_sliced
    
def read_egg_data(egg_file,sql_file):
    """
    Reads the egglaying data from Avelino and discards samples that do not have
    entries in both files.
    """
    eggs = pd.read_csv(egg_file)
    sqldat = pd.read_csv(sql_file)
    
    # Remove samples that do not have metadata attached based on basename
    common_basenames = set(eggs.basename.unique()).intersection(
            set(sqldat.basename.unique()))
    
    print('{} of the {} samples will be dropped because there is no metadata for \
          the experiments.'.format(
          len(set(eggs.basename.unique())-set(common_basenames)),
          eggs.basename.unique().shape[0]
          ))
    
    keep = set(eggs.basename.unique()).intersection(set(common_basenames))
    
    eggs = eggs[eggs['basename'].isin(keep)]
    sqldat = sqldat[sqldat['basename'].isin(keep)]
    
    eggs = eggs.reset_index(drop=True)
    sqldat = sqldat.reset_index(drop=True)
    
    return eggs,sqldat

def events_seq_parser(events_time,separator=';'):
    """
    Parses string to list of floats.
    """
    events = events_time.apply(
            lambda x:
                x if x is np.nan else
                [float(ix) for ix in x.split(separator)]
                )
    return events

def get_intervals_from_event_times(event_times, total_time, fillmissing=np.nan):
    """
    Gets time intervals from a time event sequence assuming the events where
    recorded in a video with length total_time.
    """
    if event_times is np.nan:
        intervals = fillmissing
    
    else:
        event_times = np.append(event_times,total_time)
        event_times = np.insert(event_times,0,0.0)
        intervals = np.diff(event_times)
    return intervals

def get_interval_series(egg_events,fillmissing=True):
    """
    Derive pandas series with intervals, based on events_time and total_time
    data.
    """
    if fillmissing:
        intervals = egg_events[['events_time','total_time']].apply(
            lambda row: get_intervals_from_event_times(
                    row['events_time'], row['total_time'],
                    fillmissing=row['total_time']
                    )
            ,axis=1)
    else:
        intervals = egg_events[['events_time','total_time']].apply(
            lambda row: get_intervals_from_event_times(
                    row['events_time'], row['total_time'],
                    fillmissing=np.nan
                    )
            ,axis=1)
    
    return intervals
    
def make_egg_dfs(egg_file,sql_file,fillmissing=True):
    """
    1. Reads the egglaying data from Avelino and discards samples that do not have
    entries in both files.
    2. Gets the interval data (fitting data for model) and stores it in a df
    3. Makes a metadata dataframe with matching index, with all the metadata
    for each sample
    """
    # Read the files and keep only samples with samples in both files
    eggs,sqldat = read_egg_data(egg_file,sql_file)
    
    # Get interval data (fitting data) in one dataframe
    egg_events = eggs[['events_time','total_time']].copy()
    egg_events['events_time'] = events_seq_parser(eggs['events_time'])
    
    egg_events['intervals'] = get_interval_series(
            egg_events, fillmissing=fillmissing)
    
    # Make a meta df with matching index with the egg_events df
    meta = eggs.copy()
    for col in sqldat.columns.difference(eggs.columns):
        meta.insert(0,col,
                    meta['basename'].map(
                            dict(sqldat[['basename',col]].values))
                    )
    
    egg_events.reset_index(drop=True,inplace=True)
    meta.reset_index(drop=True,inplace=True)
    
    return egg_events, meta

def expected_n_events(smoothed_z,iy,loglambdas):
    """
    Given an ssm with exponential observations, this functions
    calculates the expected number of events based on a given sequence of 
    expected states.
    It assumes a poisson process while in each state. The rate of the poisson 
    process is the lambda of the fitted exponential observations.
    """
    nev = 0
    for interval,state in zip(iy,smoothed_z):
        nev = nev + np.exp(loglambdas[state])*interval
        
    if isinstance(nev,np.ndarray):
        nev=nev[0]
    return nev

def get_smoothed_n_events(model,y):
    
    loglambdas = model.observations.params.flatten()
    
    smoothed_n_events = []
    for ii,iy in enumerate(y):
        smoothed_z = model.most_likely_states(iy.reshape(-1,1))
        smoothed_n_events.append(
                expected_n_events(smoothed_z, iy, loglambdas))
        
    return smoothed_n_events

def n_events_error(n_events, smoothed_n_events):
    """
    Get the mean error in the expected number of events in each sample.
    """    
    error1 = [np.abs((snev-nev)/nev) if nev>0 else np.nan
              for nev,snev in zip(n_events,smoothed_n_events)]
    error2 = (np.array(smoothed_n_events)-np.array(n_events))/np.array(n_events)
    error2[np.isinf(error2)] = np.nan
    print('Error in expected events per video = ', np.nanmean(error1))
    print('Error in expected events per video = ', np.nanmean(error2))
    
    return error1,error2

def get_egglaying_feat_names(K, rates=True, transitions=True):
    """
    Get the names of the egg laying features based on the number of states
    """
    
    if rates:
        rate_ft = ['lambda_{}'.format(k) for k in range(K)]
    else:
        rate_ft = None    
    
    if transitions:
        transition_ft = []
        for i in range(K):
            for j in range(K):
                transition_ft.append('transition_{}_{}'.format(i,j))
    else:
        transition_ft = None

    return rate_ft, transition_ft

if __name__=="__main__":
    event_times = [0.1,10,200]
    total_time=600
    intervals = get_intervals_from_event_times(event_times, total_time, fillmissing=np.nan)
    
    total_time = 2234
    event_times = np.sort(np.random.uniform(0,2234,size=(40)))
    sliced_intervals = slice_long_sequence(event_times, total_time, slice_size=900, 
                        fillmissing=False)
    