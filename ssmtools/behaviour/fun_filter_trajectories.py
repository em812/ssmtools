#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:15:17 2019

@author: em812
"""

import numpy as np
import pandas as pd

def get_heading_angle(x,y):
    angle = np.diff(y)/np.diff(x)
    angle = np.append(angle,angle[-1])
    angle = np.arctan(angle)
    return angle

def get_anglular_velocity(x,y):
    angle = get_heading_angle(x,y)
    angle_diff = np.diff(angle)
    angle_vel = np.append(angle_diff,angle_diff[-1])
    return angle_vel

def get_velocity(x,y):
    tmp = np.concatenate([[np.diff(x)],[np.diff(y)]],axis=0)
    velocity = np.linalg.norm(tmp,axis=0)
    velocity = np.append(velocity,velocity[-1])
    return velocity

    
def append_timeseries(y,xy,heading_angle=True,velocity=True,angular_velocity=True):
    
    for iy,yworm in enumerate(y):
        xc = xy[iy][:,0]
        yc = xy[iy][:,1]
        if heading_angle:
            angle = get_heading_angle(xc,yc)
            y[iy] = np.append(y[iy],np.reshape(angle,(-1,1)),axis=1)
        if velocity:
            vel = get_velocity(xc,yc)
            y[iy] = np.append(y[iy],np.reshape(vel,(-1,1)),axis=1)
        if angular_velocity:
            ang_vel = get_anglular_velocity(xc,yc)
            y[iy] = np.append(y[iy],np.reshape(ang_vel,(-1,1)),axis=1)
    return y

def get_independent_trajectories(df_out,include_longest=False,only_longest=False):
    """
    Get the maximum number of simultaneously overlapping trajectories (if they are overlapping we
    know that they are different worms)
    """
    if df_out.empty:
        return df_out
    
    unq_ids,counts = np.unique(df_out['new_worm_id'],return_counts=True)
    
    if only_longest:
        longest_id = unq_ids[np.argmax(counts)]
        df_out = df_out[df_out['new_worm_id'].isin([longest_id])]
        df_out.reset_index(drop=True,inplace=True)
        return df_out
    
    if include_longest:
        longest_id = unq_ids[np.argmax(counts)]
        time_range_of_longest_id = df_out.loc[df_out['new_worm_id']==longest_id,'timestamp'].values
        count_per_timestamp = df_out[df_out['timestamp'].isin(time_range_of_longest_id)].groupby(by='timestamp').count()
    else:
        count_per_timestamp = df_out.groupby(by='timestamp').count()
    
    time_with_most_worms = count_per_timestamp.loc[count_per_timestamp.worm_id==count_per_timestamp.worm_id.max(),:].index.values
    overlapping_ids_per_time = df_out[df_out.timestamp.isin(time_with_most_worms)].groupby(by='timestamp').apply(lambda x: [i for i in x.new_worm_id.values])
    unique_overlapping_ids = np.unique([np.sort(iids).tolist() for iids in overlapping_ids_per_time],axis=0)
    
    med_length = []
    for chosen_ids in unique_overlapping_ids:
        med_length.append(np.median(counts[np.isin(unq_ids,chosen_ids)]))
    
    chosen_ids = unique_overlapping_ids[np.argmax(med_length)]
    
    df_out = df_out[df_out['new_worm_id'].isin(chosen_ids)]
    df_out.reset_index(drop=True,inplace=True)
    return df_out


def filter_path_range(df_out,path_range_threshold):
    max_path_range = df_out.groupby(by='new_worm_id').max()['path_range']
    keep_ids = max_path_range[max_path_range>path_range_threshold].index
    df_out = df_out[df_out['new_worm_id'].isin(keep_ids)]
    df_out.reset_index(drop=True,inplace=True)
    return df_out


def filter_length(df_out,min_length_threshold):
    unq_ids,counts = np.unique(df_out['new_worm_id'],return_counts=True)
    keep_ids = unq_ids[np.where(counts>=min_length_threshold)]
    df_out = df_out[df_out['new_worm_id'].isin(keep_ids)]
    df_out.reset_index(drop=True,inplace=True)
    return df_out

def break_at_gaps(df,datacols,max_gap_threshold):
    unq_ids = np.unique(df['worm_id'])
    max_id = np.max(df['worm_id'])
    df_out = pd.DataFrame()
    
    for worm in unq_ids:
        
        df_worm_nonan = df.loc[(df['worm_id']==worm) & (~df['nan_ids']),:].copy()
        df_worm_nonan.insert(0,'new_worm_id',df_worm_nonan.loc[:,'worm_id'].copy())
        
        gaps = np.diff(df_worm_nonan['timestamp'])
        
        gap_times = df_worm_nonan['timestamp'].values[np.where(gaps>1)]
        gap_size = gaps[np.where(gaps>1)]
        
        for igp in range(len(gap_times)):
            if gap_size[igp] > max_gap_threshold:
                # break trajectory (give unique name to broken pieces)
                df_worm_nonan.loc[df_worm_nonan['timestamp']>gap_times[igp],'new_worm_id'] = max_id + 1
                max_id += 1
        
        
            elif gap_size[igp] <= max_gap_threshold:
                # add missing timestamps
                add_df = pd.DataFrame(np.full_like(df_worm_nonan.iloc[0:gap_size[igp]-1].values,np.nan),columns=df_worm_nonan.columns)
                df_worm_nonan = pd.concat([df_worm_nonan[df_worm_nonan.timestamp<=gap_times[igp]],add_df,df_worm_nonan[df_worm_nonan.timestamp>=gap_times[igp]+gap_size[igp]]])
                # interpolate (each datatype separately)
                df_worm_nonan[['worm_id','new_worm_id','timestamp']] = df_worm_nonan[['worm_id','new_worm_id','timestamp']].astype(float).interpolate(method='linear',limit_area='inside').astype(int)
                df_worm_nonan['nan_ids'] = df_worm_nonan['nan_ids'].fillna(value=True)
                df_worm_nonan[['path_range']+list(datacols)] = df_worm_nonan[['path_range']+list(datacols)].interpolate(method='linear',limit_area='inside')
            
        df_out = pd.concat((df_out,df_worm_nonan))
    
    df_out.reset_index(drop=True,inplace=True)
    
    return df_out
    
def filter_trajectories(worm_id,timestamp,nan_ids,path_range,
                        max_gap_threshold,min_length_threshold,path_range_threshold,
                        data=None,include_longest=False,only_longest=False):
    """
     Filter the trajectories and selects the longest one per video
     - breaks trajectories if gap > max_gap_threshold
     - at this point filters out trajectories with length < min_length_threshold
     - also filters out trajectories with path range < path_range_threshold
     - chooses longest filtered trajectory
     - also chooses all trajectories overlapping with the longest
     Param:
         - worm_id : int array-like, worm index
         - timestamp : int array-like, raw timestamp in frame numbers
         - nan_ids : boolean array-like, True indicates nan value in the time series of interest
         - max_gap_threshold : maximum gap in frame numbers (!) that is accepted without breaking the trajectory
         - min_length_threshold : minimum allowed length of a good trajectory in frame numbers (!)
      Return:
    """
    
    assert len(worm_id) == len(timestamp)
    assert len(worm_id) == len(nan_ids)
    
    # make dataframe with the data and the ids (each datatype separately otherwise they are all converted to the same type)
    df = pd.DataFrame(np.array([worm_id,timestamp]).T,columns=['worm_id','timestamp'])
    df['nan_ids'] = nan_ids
    df['path_range'] = path_range
    if data is not None:
        df = pd.concat([df,data],axis=1)
    
    # make sure arrays are sorted by worm_id and by timestamp
    df.sort_values(by=['worm_id','timestamp'],inplace=True)
    
    # Break trajectories at gaps > max_gap_threshold and remove nans
    df = break_at_gaps(df,data.columns,max_gap_threshold)
    
    # Filter out trajectories with length < min_length_threshold
    df = filter_length(df,min_length_threshold)
    
    # Filter out trajectories with path_range < path_range_threshold
    df = filter_path_range(df,path_range_threshold)

    # Choose longest trajectory and all overlapping ones
    df = get_independent_trajectories(df,include_longest=include_longest,only_longest=only_longest)
    
    if not df.empty:
        filtered_ids = []
        for worm in df['new_worm_id'].unique():
            wormids = pd.DataFrame()
            wormids.loc[0,'worm_id'] = df[df['new_worm_id']==worm].iloc[0]['worm_id']
            wormids.loc[0,'new_worm_id'] = df[df['new_worm_id']==worm].iloc[0]['new_worm_id']
            wormids.loc[0,'n_nan_ids'] = df.loc[df['new_worm_id']==worm,'nan_ids'].sum()
            wormids.loc[0,'timestamp_start'] = df.loc[df['new_worm_id']==worm,'timestamp'].min()
            wormids.loc[0,'timestamp_end'] = df.loc[df['new_worm_id']==worm,'timestamp'].max()
            wormids.loc[0,'trajectory_length'] = df.loc[df['new_worm_id']==worm,'timestamp'].shape[0]
            filtered_ids.append(wormids)
        filtered_ids = pd.concat(filtered_ids,axis=0)
    else:
        filtered_ids = pd.DataFrame([])
    
    if data is not None:
        return filtered_ids, df[['new_worm_id'] + list(data.columns)]
    else:
        return filtered_ids
    

def get_path_range(worm_id,coord_x,coord_y):
    
    df = pd.DataFrame(np.array([worm_id,coord_x,coord_y]).T,columns=['worm_id','x','y'])
    centroids = df.groupby(by='worm_id').mean()
    df.insert(0,'centroid_x',df['worm_id'].map(dict(zip(centroids.index,centroids['x']))))
    df.insert(0,'centroid_y',df['worm_id'].map(dict(zip(centroids.index,centroids['y']))))
    path_range = np.sqrt(np.square(df[['x','y']].values-df[['centroid_x','centroid_y']].values).sum(axis=1))
    
    return path_range
        
def mark_nans(eigen):
    
    nan_ids = np.any(np.isnan(eigen),axis=1)
    
    if isinstance(nan_ids,(pd.DataFrame,pd.Series)):
        nan_ids = nan_ids.values
    
    return nan_ids

def read_and_filter_data(filename,max_gap_threshold,min_length_threshold,path_range_threshold,
                         timeseries_names=None,include_longest=False,only_longest=False):
    
    timeseries = pd.read_hdf(filename,'timeseries_data',mode='r')
    
    if timeseries.empty:
        return pd.DataFrame([]),pd.DataFrame([])
        
    if timeseries_names is None:
        allnan = np.all(np.isnan(timeseries),axis=0)
        timeseries_names = timeseries.loc[:,~allnan].columns.difference(
                ['worm_index','timestamp'])
    
    nan_ids = mark_nans(timeseries[timeseries_names])
        
    worm_id = timeseries['worm_index']
    
    timestamp = timeseries['timestamp'].astype(int)
    
    path_range = get_path_range(timeseries['worm_index'],timeseries['coord_x_body'],timeseries['coord_y_body'])
    
    timeseries = timeseries[timeseries_names]
    
    filtered_ids,filtered_data = filter_trajectories(
            worm_id, timestamp, nan_ids, path_range, max_gap_threshold, 
            min_length_threshold, path_range_threshold, data=timeseries,
            include_longest=include_longest, only_longest=only_longest)
#        filtered_ids['filename'] = filename
    return filtered_ids,filtered_data

def get_filtered_trajectories(
        file_list, metadata, timeseries_names, max_gap_threshold,
        min_length_threshold, path_range_threshold, save_to=None, 
        return_data=False, include_longest_trajectory=False, 
        only_longest=False, downsampling_rate=None):
    
    from time import time
    from os.path import join
    
    filtered_meta = []
    if return_data:
        filtered_data = []
    unq_traj_id = 0
    for ifl,file in enumerate(file_list):
        
        start_time = time()
        print('Reading {}...\nFile {} of {}...'.format(file,ifl,metadata.shape[0]))
        
        filtered_ids,filtered_series = read_and_filter_data(
                file, max_gap_threshold, min_length_threshold,
                path_range_threshold, timeseries_names=timeseries_names,
                include_longest=include_longest_trajectory,
                only_longest=only_longest)
        
        if not filtered_ids.empty:
            for worm in filtered_ids['new_worm_id'].unique():
                worm_filtered_series = filtered_series.loc[filtered_series['new_worm_id']==worm,:]
                if downsampling_rate is not None:
                    worm_filtered_series = worm_filtered_series.iloc[::downsampling_rate, :]
                    worm_filtered_series = worm_filtered_series.reset_index(drop=True)
                worm_metadata = pd.concat((metadata.iloc[ifl,:],filtered_ids[filtered_ids['new_worm_id']==worm].T)).T
                worm_metadata['unq_traj_id'] = unq_traj_id
                
                if save_to is not None:
                    print('Saving filtered timeseries...'.format(file,ifl,metadata.shape[0]))
                    worm_filtered_series.to_hdf(join(save_to,'unq_traj_id_{}.hdf5'.format(unq_traj_id)), key='filtered_timeseries', mode='w')
                    worm_metadata['unq_traj_filename'] = 'unq_traj_id_{}.hdf5'.format(unq_traj_id)
                    #filtered_series.loc[filtered_series['new_worm_id']==worm,timeseries_names].to_csv(join(save_to,'unq_traj_id_{}.csv'.format(unq_traj_id)),index=None)
                
                filtered_meta.append(worm_metadata)            
                if return_data:
                    filtered_data.append(worm_filtered_series)
                unq_traj_id += 1
                
        print('Done in {} sec.'.format(time()-start_time))
    
    filtered_meta = pd.concat(filtered_meta,axis=0)
    filtered_meta['max_gap_threshold'] = max_gap_threshold; filtered_meta['min_length_threshold'] = min_length_threshold
    filtered_meta['path_range_threshold'] = path_range_threshold; filtered_meta['only_longest'] = only_longest
    filtered_meta = filtered_meta.reset_index(drop=True)
    
    if return_data:
        return filtered_meta,filtered_data
    else:
        return filtered_meta

def get_savepath(root,max_gap_threshold,min_length_threshold,include_longest_trajectory,only_longest,path_range_threshold):
    from os.path import join
    from os import makedirs
    
    
    if only_longest:
        savepath = join(root,'filtered_trajectories','max_gap={}_min_length={}_only_longest'.format(max_gap_threshold,min_length_threshold))
    elif include_longest_trajectory:
        savepath = join(root,'filtered_trajectories','max_gap={}_min_length={}_include_longest'.format(max_gap_threshold,min_length_threshold))
    else:
        savepath = join(root,'filtered_trajectories','max_gap={}_min_length={}_max_overlapping'.format(max_gap_threshold,min_length_threshold))
    
    makedirs(savepath,exist_ok=True)
    
    return savepath

def read_timeseries(file_list,timeseries_names,verbose=False):
    y = []    
    for ifl,file in enumerate(file_list):
        if verbose:
            print('Reading {}...\nFile {} of {}.'.format(file,ifl+1,len(file_list)))
        y.append((pd.read_hdf(file)).loc[:,timeseries_names].values)
    return y


if __name__ == "__main__":
    worm_id = np.repeat(np.arange(2),20)
    worm_id[25:] = 2
    nan_ids = np.full_like(worm_id,False,dtype=bool)
    nan_ids[5:10] = True
    nan_ids[16] = True
    nan_ids[27:29] = True
    path_range = np.full_like(worm_id,100.0,dtype=float)
    timestamp = np.concatenate((np.arange(20),np.arange(20)))
    timestamp[15:20] = timestamp[15:20]+10
    data = pd.DataFrame(np.random.rand(40,2),columns=['series1','series2'])
    data.iloc[5:10,:] = np.nan
    data.iloc[16] = np.nan
    
    max_gap_threshold = 4
    min_length_threshold = 4 #15
    path_range_threshold = 1
    
    df = pd.DataFrame(np.array([worm_id,timestamp]).T,columns=['worm_id','timestamp'])
    df['nan_ids'] = nan_ids
    df['path_range'] = path_range
    df = pd.concat((df,data),axis=1)
    
#    df_out1 = break_at_gaps(df,data.columns,max_gap_threshold)
#    df_out2 = get_independent_trajectories(df_out1,include_longest=False)
    df_ids,filtered_data = filter_trajectories(worm_id,timestamp,nan_ids,path_range,data,max_gap_threshold,min_length_threshold,path_range_threshold,include_longest=False)
    
    