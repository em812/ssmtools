#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:26:22 2020

@author: em812
"""
import numpy as np

def sample_frames(file_list, timeseries_names, n_samples_per_video = 0.1):
    """
    Randomly sample frames from the timeseries dataframe.

    Parameters
    ----------
    file_list : list
        list of featuresN file paths.
    timeseries_names : list of strings
        columns names of timeseries that we want to read.
    n_samples_per_video : how many samples to return, optional
        If it is a float <1, then it is interpreted as a fraction.
        If it is an int, then it is the exact number of frames to sample.
        The default is 0.1.

    Raises
    ------
    ValueError
        when the n_samples_per video is not one of the recognised data types.

    Returns
    -------
    frames : list of dataframes
        The timeseries_names columns for the sampled frames per video.

    """
    import pdb
    import pandas as pd
    from ssmtools.behaviour.read_and_filter_trajectories import mark_nans

    if isinstance(n_samples_per_video, int):
        n = n_samples_per_video
        frac = None
    elif isinstance(n_samples_per_video, float) and n_samples_per_video < 1:
        n = None
        frac = n_samples_per_video
    else:
        raise ValueError('n_samples_per_video datatype not recognised.')

    frames = []
    for ifl,file in enumerate(file_list):
        print('Sampling from file {} of {}...'.format(ifl+1, len(file_list)))
        timeseries = pd.read_hdf(file, key='timeseries_data', columns=timeseries_names)
        timeseries = timeseries[timeseries_names]

        # drop nan frames
        nan_ids = mark_nans(timeseries)
        timeseries = timeseries.loc[~nan_ids,:]

        # sample
        frames.append(
            timeseries.sample(
                n=min(n, timeseries.shape[0]), frac=frac, axis=0)
            )

    return frames


def eigen2postures(y,syntax):
    """
    Transforms the timeseries of eigenprojections to a discrete space of
    posture sequences, taking the nearest neighbour from the syntax library
    of each real posture at each time.

    Parameters
    ----------
    y : list of arrays of shape = (T x n_eigen)
        Each array is the timeseries of eigenprojections.
    syntax : array shape = (n_postures x n_eigen)
        The library of allowed postures.

    Returns
    -------
    postures : list of arrays of integers of shape = (T x 1)
        Posture sequences. Contain one integer per time point, which defines
        the row of the posture in the syntax matrix.

    """
    from scipy.spatial import cKDTree
    import numpy as np

    mytree = cKDTree(syntax)

    postures = []
    for iy in y :
        dist, iposture = mytree.query(iy, k=1)

        assert np.all(iposture < syntax.shape[0])

        postures.append(iposture.reshape(-1,1))

    return postures

def collapse_posture_seq(postseq):
    """
    Collapse a sequence of posture numbers to remove repeats

    Parameters
    ----------
    postseq : array shape = (T,)
        List or array of posture sequences.

    Returns
    -------
    array shape = (T, 2)
        Array of collapsed posture sequences, with the number of repeats
        stored in the second column.

    """
    from itertools import groupby

    collapsed_postseq = [[p[0],len(list(rep))] for p,rep in groupby(postseq)]

    return np.array(collapsed_postseq)

def expand_posture_seq(postseq):
    """
    Expand a collapsed posture sequence

    Parameters
    ----------
    postseq : array shape = (T, 2)
        Column 0 contains the collapsed posture sequence and column 1 contains
        the number of repeats of the given posture in the raw sequence.

    Returns
    -------
    array shape = (T,)
        The expanded posture sequence (raw sequence).

    """
    assert postseq.shape[1] == 2

    expanded_postseq = []
    for p,counts in postseq:
        expanded_postseq.extend([p]*counts)

    return np.array(expanded_postseq)


def get_ngrams(seq, n, collapsed=False):
    """
    Get all the ngrams from a sequence of integers

    Parameters
    ----------
    seq : array shape = (T,) [if not collapsed] or shape = (T,2) if collapsed
        If not collapsed, then seq is a sequence of integers.
        If collapsed, then the first column of seq is the sequence of integers
        and the second column is the number of repeats.
    n : integer
        Size of grams to return.
    collapsed : bool, optional
        is the sequence collapsed or does it contain repeats of the
        same number? Default is False.

    Returns
    -------
    ngrams : list of tuples shape = (T-n+1, n)
        All the ngrams observed in seq.
    ncounts : list of tuples shape = (T-n+1, n), optional
        This is returned only if collapsed = True. It consists of the
        counts of each posture in the ngram (how many time it is repeated).

    """
    if not collapsed:
        ngrams = zip(*(seq[i:] for i in range(n))) #map(list, )
        ngrams = [g for g in ngrams]
        return ngrams
    else:
        pseq = seq[:,0]
        ngrams = zip(*(pseq[i:] for i in range(n))) #map(list, )
        ngrams = [g for g in ngrams]
        pcounts = seq[:,1]
        ncounts = zip(*(pcounts[i:] for i in range(n))) #map(list, )
        ncounts = [c for c in ncounts]
        return ngrams, ncounts

def get_ngram_accumulation(ngrams, step):
    """
    Get the accumulation curve of unique ngrams stepwise

    Parameters
    ----------
    ngrams : list of tuples shape = (t-n+1, n)
        The sequence of all ngrams from a sequence of integers.
    step : int
        The step at whcih the number of unique grams will be calculated.

    Returns
    -------
    accum : list
        The number of unique ngrams at each step.

    """

    from collections import Counter

    accum = []
    for i in range(step, len(ngrams)+step, step):
        accum.append(len(Counter(ngrams[:i]).keys()))

    return accum

def ngrams2integersequence(ngram_seq, ngram_library):

    from scipy.spatial import cKDTree
    import numpy as np

    mytree = cKDTree(ngram_library)

    int_seq = []
    for seq in ngram_seq:
        dist, igram = mytree.query(seq, k=1)

        if dist == 0:
            int_seq.append(igram)
        else:
            int_seq.append(np.nan)

    return np.array(int_seq)

if __name__=="__main__":
    foo = np.random.randint(0,5,20)
    bar = collapse_posture_seq(foo)
    tmp = expand_posture_seq(bar)
    assert np.all(foo==tmp)

