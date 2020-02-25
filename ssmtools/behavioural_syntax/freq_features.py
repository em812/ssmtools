#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:54:36 2020

@author: em812
"""

import numpy as np
import pdb

def get_pfreq_features(postures, n_syntax):
    """
    Get the postures frequences for a list of posture sequences

    Parameters
    ----------
    postures : list of arrays length = number of sequences (worms)
        A list with all the posture sequences. Each sequence represents a worm
        trajectory and has shape=(T,)
    n_syntax : int
        Number of postures in the posture syntax.

    Returns
    -------
    pfreq : array shape = (number of sequences, n_syntax)
        The frequence of each posture in each sequence.

    """

    pfreq = np.zeros((len(postures),n_syntax))

    for ip, pseq in enumerate(postures):
        pst, count = np.unique(pseq, return_counts=True)
        pfreq[ip,pst] = count/np.sum(count)

    return pfreq

def get_ngram_freq_features(ngram_seq, ngram_library, get_duration=False,
                            ngram_counts=None):
    """
    Get the ngram frequences for a list of ngram sequences

    Parameters
    ----------
    ngram_seq : list of arrays length = number of sequences (worms)
        A list with all the n-gram sequences. Each sequence represents a worm
        trajectory and has shape=(T-n+1, n)
    ngram_library : list of tuples
        All the unique n-grams that will be counted.

    Returns
    -------
    ngr_freq : array shape = (number of sequences, len(ngram_library))
        The frequency of each n-gram in the library in each sequence.

    """

    ## Initialize arrays
    ngr_freq = np.zeros((len(ngram_seq), ngram_library.shape[0]))

    if get_duration:
        if ngram_counts is None:
            raise ValueError('You need to give the ngram counts to get the'+
                             'n-gram duration features.')
        ngr_dur = np.zeros((len(ngram_counts), ngram_library.shape[0]))

    ## Loop over sequences
    for iseq, gseq in enumerate(ngram_seq):
        # Get unique grams observed in the sequence, their indices and their counts
        grams, unq_ind, counts = np.unique(gseq, axis=0, return_index=True,
                                  return_counts=True)
        # Loop over the unique grams observed in the sequence and get the
        # frequency with which they appear.
        # If get_duration=True, then also get the total duration in frames
        # for which each of the unique grams is observed.
        for g, i, c in zip(grams, unq_ind, counts):
            ind = np.all(ngram_library==g,axis=1)
            ngr_freq[iseq, ind] = c
            if get_duration:
                cseq = np.array(ngram_counts[iseq])
                ngr_dur[iseq, ind] = np.sum(cseq[i,:])

    ngr_freq = ngr_freq / np.sum(ngr_freq, axis=1).reshape(-1,1)

    if not get_duration:
        return ngr_freq
    else:
        return np.concat([ngr_freq, ngr_dur], axis=1)