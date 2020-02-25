#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:15:17 2019

@author: em812
"""

import numpy as np
import pandas as pd

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
        df_out = df_out[df_out['new_worm_id'].isin(longest_id)]
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

def filter_trajectories(
        data, worm_id, timestamp, nan_ids, path_range, max_gap_threshold,
        min_length_threshold, path_range_threshold,
        include_longest=False, only_longest=False):
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
         - data: dataframe, timeseries of interest shape = (len(worm_id , nb of timeseries))
         - max_gap_threshold : maximum gap in frame numbers (!) that is accepted without breaking the trajectory
         - min_length_threshold : minimum allowed length of a good trajectory in frame numbers (!)
      Return:
    """

    assert len(worm_id) == len(timestamp)
    assert len(worm_id) == len(nan_ids)
    assert len(worm_id) == data.shape[0]

    # make dataframe with the data and the ids (each datatype separately otherwise they are all converted to the same type)
    df = pd.DataFrame(np.array([worm_id,timestamp]).T,columns=['worm_id','timestamp'])
    df['nan_ids'] = nan_ids
    df['path_range'] = path_range

    df = pd.concat((df,data),axis=1)

    # make sure arrays are sorted by worm_id and by timestamp
    df.sort_values(by=['worm_id','timestamp'],inplace=True)

    # Break trajectories at gaps > max_gap_threshold and remove nans
    df_out = break_at_gaps(df,data.columns,max_gap_threshold)

    # Filter out trajectories with length < min_length_threshold
    df_out = filter_length(df_out,min_length_threshold)

    # Filter out trajectories with path_range < path_range_threshold
    df_out = filter_path_range(df_out,path_range_threshold)

    # Choose longest trajectory and all overlapping ones
    df_out = get_independent_trajectories(df_out,include_longest=include_longest)

    return df_out[['worm_id','new_worm_id','timestamp','nan_ids']],df_out[data.columns]


def read_and_filter_data(filename,timeseries_names,max_gap_threshold,min_length_threshold,
                         path_range_threshold,include_longest=False,only_longest=False):

    timeseries = pd.read_hdf(filename, key='timeseries_data')

    path_range = get_path_range(
        timeseries['worm_index'], timeseries['coord_x_body'],timeseries['coord_y_body'])

    timeseries = timeseries[['worm_index','timestamp']+timeseries_names]

    nan_ids = mark_nans(timeseries[timeseries_names])

    filtered_ids,filtered_eigen = filter_trajectories(
        timeseries[timeseries_names], timeseries['worm_index'],
        timeseries['timestamp'], nan_ids, path_range, max_gap_threshold,
        min_length_threshold, path_range_threshold,
        include_longest=include_longest)

    filtered_ids['filename'] = filename

    return filtered_ids,filtered_eigen


def get_filtered_data_and_metadata(
        file_list, metadata, timeseries_names, max_gap_threshold,
        min_length_threshold, path_range_threshold,
        include_longest_trajectory = False, only_longest = False,
        return_traj = True, saveto = None,
        meta_filename = 'metadata_filtered_trajectories.csv'):

    from time import time
    from pathlib import Path

    if saveto is None or return_traj:
        grow_y = True
        y = []
    else:
        grow_y = False

    if saveto is not None:
        saveto = Path(saveto)

    ids = [] ; new_ids = []  # read time series and id data
    meta = []; timestamp_start = [] ; timestamp_end = []
    n_nan_ids = []
    itraj = 0
    for ifl,file in enumerate(file_list):
        start_time = time()
        print(
            'Reading {}...\nFile {} of {}...'.format(file, ifl, metadata.shape[0])
            )
        filtered_ids, filtered_eigen = read_and_filter_data(
            file, timeseries_names, max_gap_threshold, min_length_threshold,
            path_range_threshold, include_longest=include_longest_trajectory)
        if not filtered_eigen.empty:
            for worm in filtered_ids['new_worm_id'].unique():
                wormindx = filtered_ids['new_worm_id']==worm

                iy = filtered_eigen[wormindx]
                if grow_y:
                    y.append(iy.values)
                if saveto is not None:
                    iy.to_hdf(
                        saveto / 'unq_traj_id_{}.hdf5'.format(itraj),
                        key = 'filtered_timeseries', mode='w'
                        )
                    itraj += 1

                ids.append(filtered_ids.loc[wormindx, 'worm_id' ].iloc[0])
                new_ids.append(worm)
                timestamp_start.append(
                    filtered_ids.loc[wormindx, 'timestamp'].min()
                    )
                timestamp_end.append(
                    filtered_ids.loc[wormindx, 'timestamp'].max()
                    )
                n_nan_ids.append(filtered_ids.loc[wormindx, 'nan_ids'].sum())
                meta.append(metadata.iloc[ifl,:])
        print('Done in {} sec.'.format(time()-start_time))
    meta = pd.concat(meta,axis=1).T
    meta = meta.reset_index(drop=True)
    meta.insert(0,'new_worm_id',new_ids)
    meta.insert(0,'worm_id',ids)
    meta.insert(0,'timestamp_end',timestamp_end)
    meta.insert(0,'timestamp_start',timestamp_start)
    meta.insert(0,'n_nan_ids',n_nan_ids)

    if saveto is not None:
        meta.to_csv(saveto / meta_filename, index=None)

    if grow_y:
        return y, meta
    else:
        return meta


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

