#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:23:19 2020

@author: em812
"""
import pandas as pd
import numpy as np


def plot_exp_observation_distr(y, K, loglambdas, lims=[None,None], savefig=None):
    import matplotlib.pyplot as plt
    from scipy.stats import expon
    
    yall = np.concatenate(y)
    yall = yall + np.random.randn(*yall.shape)*0.01
    
    if lims[0] is None:
        lims[0] = np.min(yall)
    if lims[1] is None:
        lims[1] = np.max(yall)
    
    yall = yall[(yall>=lims[0])&(yall<=lims[1])]
    
    fig = plt.figure()
    plt.hist(yall,bins=50,density=True)
    xlim = fig.gca().get_xlim()
    ylim = fig.gca().get_ylim()
    
    x = np.linspace(0,xlim[1],1000)
    for state in range(K):
        statelambda = np.exp(loglambdas[state])
        exp_distr = expon.pdf(x,scale=1/statelambda)
        plt.plot(x,exp_distr,label='state = {}'.format(state))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    return

def plot_state_sequence(z,labels=None,savefig=None):
    from hips.plotting.colormaps import gradient_cmap, white_to_color_cmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_style("white")
    
    color_names = [
        "windows blue",
        "red",
        "amber",
        "faded green",
        "dusty purple",
        "orange"
        ]
    
    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors[:np.unique(z).shape[0]])
    
    if len(z.shape)==1:
    
        T = z.shape[0]
        
        plt.figure(figsize=(10, 4))
        plt.subplot(111)
        plt.imshow(z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
        plt.xlim(0, T)
        if labels is None:
            plt.ylabel("$z_{\\mathrm{inferred}}$")
        else:
            plt.ylabel(labels)
        plt.yticks([])
        
    
    else:
        for iz,hmm_z in enumerate(z):
            plt.subplot(int('{}1{}'.format(len(z),iz+1)))
            plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
            plt.xlim(0, hmm_z.shape[0])
            if labels is None:
                plt.ylabel("$z_{\\mathrm{inferred,{}}}$".format(iz))
            else:
                plt.ylabel(labels[iz])
            plt.yticks([])
            plt.xlabel("time")
            
        plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    return

def plot_state_sequence_and_obs(z,y,labels=None,yscale=None,savefig=None):
    from hips.plotting.colormaps import gradient_cmap, white_to_color_cmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_style("white")
    
    color_names = [
        "windows blue",
        "red",
        "amber",
        "faded green",
        "dusty purple",
        "orange"
        ]
    
    
    colors = sns.xkcd_palette(color_names)
    
    if isinstance(z,np.ndarray):
        
        n_states = np.unique(z).shape[0]
        cmap = gradient_cmap(colors[:max(2,n_states)])
    
        assert z.shape == y.flatten().shape
        
        
        T = z.shape[0]
        
        plt.figure(figsize=(20, 4))
        plt.subplot(111)
        plt.imshow(z[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=z.max(),
                   extent=(-0.5, z.shape[0]-0.5, y.min(), y.max()))
        
        if n_states>1:
            bound = np.linspace(0,n_states-1,2*n_states-1)
            plt.colorbar(ticks=np.unique(z),boundaries=bound)
        plt.xlim(0, T)
        
        plt.plot(y,'x',color='black')
        plt.ylabel('Interval size (s)')
        if yscale is not None:
            plt.yscale(yscale)
        if labels is not None:
            plt.title(labels)
        
#        ycent = (y.max()-y.min())/2
#        yt = (y-ycent)/(y.max()-y.min())
#        plt.scatter(range(yt.shape[0]),yt,c='k',marker='x')
        
        
    elif isinstance(z,list):
        n_states = np.unique(np.concatenate(z)).shape[0]
        cmap = gradient_cmap(colors[:max(2,n_states)])
    
        
        plt.figure(figsize=(20, 4))
        for iz,(hmm_z,hmm_y) in enumerate(zip(z,y)):
            assert hmm_z.flatten().shape == hmm_y.flatten().shape
            
            T = hmm_z.shape[0]
            
            plt.subplot(int('{}1{}'.format(len(z),iz+1)))
            plt.imshow(hmm_z[None,:], aspect="auto", cmap=cmap, vmin=0, 
                       vmax=hmm_z.max(), extent=(-0.5, hmm_z.shape[0]-0.5, 
                        hmm_y.min(), hmm_y.max()))
        
            if n_states>1:
                bound = np.linspace(0,n_states-1,2*n_states-1)
                plt.colorbar(ticks=np.unique(hmm_z),boundaries=bound)
            plt.xlim(0, T)
            
            plt.plot(hmm_y,'x',color='black')
            plt.ylabel('Interval size (s)')
            if yscale is not None:
                plt.yscale(yscale)
            if labels is not None:
                plt.title(labels[iz])
        plt.tight_layout()
        
    else:
        ValueError('Data type not understood.')
    
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()

    return

def plot_intervals_and_states(z,y,loglambdas,labels=None,savefig=None):
    from hips.plotting.colormaps import gradient_cmap, white_to_color_cmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_style("white")
    
    color_names = [
        "windows blue",
        "red",
        "amber",
        "faded green",
        "dusty purple",
        "orange"
        ]
    
    colors = sns.xkcd_palette(color_names)
    y = y.flatten()
    lambdas = np.exp(loglambdas)
    
    plt.figure(figsize=(8, 4))
    
    if len(z.shape)==1:
        
        assert z.shape == y.shape
        
        T = np.sum(y)
        
        plt.subplot(111)
        
        x=0
        for interv,state in zip(y,z):
            plt.plot([x,x+interv],lambdas[state]*np.ones(2),color=colors[state],marker='x')
            plt.plot((x+interv)*np.ones(2),lambdas,color='black')
            x = x+interv
        plt.xlim(0, T)
        plt.ylabel("lambda")
        if labels is not None:
            plt.title(labels)
        
    else:
        for iz,(hmm_z,hmm_y) in enumerate(zip(z,y)):
            assert hmm_z.shape == hmm_y.shape
            
            T = np.sum(hmm_y)
            
            plt.subplot(int('{}1{}'.format(len(z),iz+1)))
            
            x=0
            for interv,state in zip(hmm_y,hmm_z):
                plt.plot([x,x+interv],lambdas[state]*np.ones(2),color=colors[state])
                x = x+interv
            plt.xlim(0, T)
            plt.ylabel("lambda")
            if labels is not None:
                plt.title(labels)
            
        plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()

    return



if __name__=="__main__":
    z=[smoothed_z,smoothed_z]
    y = [iy,iy]
    
    plot_state_sequence_and_obs([smoothed_z,smoothed_z],[iy,iy],yscale="log")
