#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:21:05 2020

@author: em812
"""
import numpy as np
import pdb

def get_r2(frames, syntax, eigenworms, all_values=False):

    from ssmtools.behavioural_syntax.discretize import eigen2postures
    from ssmtools.predict_movement.movement import eigen2angle
    from sklearn.metrics import r2_score

    n_eigen = frames.shape[1]
    postures = eigen2postures([frames], syntax)[0]

    angles_syntax = eigen2angle(syntax, n_eigen, eigenworms)

    r2 = []
    for frame, post in zip(frames,postures):
        angles = eigen2angle(frame.reshape(1,-1), n_eigen, eigenworms)
        r2.append(r2_score(angles_syntax[post,:].flatten(), angles.flatten()))

    if all_values:
        return r2
    else:
        return np.mean(r2), np.std(r2)