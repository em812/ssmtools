#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:54:24 2019

@author: em812
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from hips.plotting.colormaps import white_to_color_cmap
import pdb

color_names = [
    "windows blue",
    "red",
    "bright lavender",
    "faded green",
    "dusty purple",
    "orange",
    "grey",
    "teal",
    "brick red",
    "dull yellow",
    "purply blue",
    "golden brown"
    ]*10

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          savefig=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()
    if savefig is not None:
        plt.savefig(savefig)

def plot_skeleton(
        thetas, starting_point=[0,0], segment_length=1.0, close_fig=False,
        savefig=None, dpi=None, figsize=None
        ):

    x = np.zeros(thetas.shape[0]+1)
    y = np.zeros(thetas.shape[0]+1)

    x[0],y[0] = starting_point
    for ip,theta in enumerate(thetas):
        if ip==0:
            x[ip+1] = x[ip] + segment_length*np.cos(theta)
            y[ip+1] = y[ip] + segment_length*np.sin(theta)
        else:
            x[ip+1] = x[ip] + segment_length*np.cos(theta)
            y[ip+1] = y[ip] + segment_length*np.sin(theta)
    plt.figure(figsize=figsize,dpi=dpi)
    plt.axis('equal')
    plt.plot(x,y)
    if savefig is not None:
        plt.savefig(savefig,dpi=dpi)
    if close_fig:
        plt.close()
    return

def get_colors(n):
    import random
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

def plot_y_vs_smoothed_y(
        yworm, smoothed_y, smoothed_z=None, timestamp=None, savefig=None
        ):
    """
    Plots all the components of the input timeseries and their smoothed
    version given the most likely state sequence of a fitted HMM.
    param:
        yworm = input timeseries
        smoothed_y = smoothed timeseries from a fitted HMM
        smoothed_z = most likely sequence (optional). If not None, then an
                     extra subplot is created at the bottom with the most
                     likely sequences
        timestamp = real timestamp of worm trajectory in the video (optional).
                    If given, the x axis of the subplots (time) will have the
                    real timestamp values.
        savefig = full path including filename where the figure will be saved
    """
    n_dim = yworm.shape[1]
    if smoothed_z is not None:
        n_subplots = n_dim+1
    else:
        n_subplots = n_dim

    if timestamp is not None:
        xx = timestamp
    else:
        xx = list(range(yworm.shape[0]))

    fig,ax = plt.subplots(n_subplots,1)
    for n in range(n_dim):
        ax[n].plot(xx,yworm[:,n])
        ax[n].plot(xx,smoothed_y[:,n])
    if smoothed_z is not None:
        ax[-1].imshow(smoothed_z.reshape(1,-1), aspect="auto", cmap="jet")

    if savefig is not None:
        fig.savefig(savefig)


def plot_observation_distr_and_path(y, strains_compare, K, model):
    # PCA eigen space
    from sklearn.decomposition import PCA

    pca = PCA(n_components = 2)
    _ = pca.fit_transform(
        np.concatenate([y[strain][worm]
                        for strain in strains_compare
                        for worm in range(len(y[strain]))],axis=0))

    # Get smoothed_z for one N2 worm and one unc3 worm to plot the sequence
    # on top of the distributions
    Ytrace = pca.transform(
        np.concatenate([y[strain][0]
                        for strain in strains_compare],axis=0)
        )
    ztrace = np.concatenate(
        [model.most_likely_states(y[strain][0])
         for strain in strains_compare]
        )
    strain_id = [[strain]*len(y[strain][0]) for strain in strains_compare]
    strain_id = [item for sublist in strain_id for item in sublist]

    # subsample
    Ytrace = Ytrace[1::20,:]
    ztrace = ztrace[1::20]
    strain_id = strain_id[1::20]

    # Plot observation distributions
    lim = 0.0
    for strain in strains_compare:
        for yworm in y[strain]:
            lim = np.max((lim,np.max(np.abs(yworm))))
    lim = .85 * lim
    XX, YY = np.meshgrid(np.linspace(-lim, lim, 100),
                         np.linspace(-lim, lim, 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))
    data_eigen_space = pca.inverse_transform(data)
    input = np.zeros((data_eigen_space.shape[0], 0))
    mask = np.ones_like(data_eigen_space, dtype=bool)
    tag = None
    lls = model.observations.log_likelihoods(
        data_eigen_space, input, mask, tag)

    colors = sns.xkcd_palette(color_names)

    for strain in strains_compare:
        Ystrain = Ytrace[np.isin(strain_id,strain),:]
        zstrain = ztrace[np.isin(strain_id,strain)]
        plt.figure(figsize=(6, 6))
        for k in range(K):
            plt.contour(XX, YY, np.exp(lls[:,k]).reshape(XX.shape),
                        cmap=white_to_color_cmap(colors[k]))
            plt.plot(Ystrain[zstrain==k, 0], Ystrain[zstrain==k, 1],
                     'o', mfc=colors[k], mec='none', ms=4)

        plt.plot(Ystrain[:,0], Ystrain[:,1],
                 '-k', lw=1, alpha=.25,label=strain)
        plt.xlabel("$y_1$")
        plt.ylabel("$y_2$")
        plt.title("Observation Distributions")
        plt.legend()


def plot_observation_distributions_pca(
        y, strains_compare, K, model, observations
        ):

    # PCA eigen space
    from sklearn.decomposition import PCA

    pca = PCA(n_components = 2)
    _ = pca.fit_transform(
        np.concatenate([y[strain][worm]
                        for strain in strains_compare
                        for worm in range(len(y[strain]))],axis=0
                       )
        )

    # Plot observation distributions
    lim = 0.0
    for strain in strains_compare:
        for yworm in y[strain]:
            lim = np.max((lim,np.max(np.abs(yworm))))
    lim = .85 * lim
    XX, YY = np.meshgrid(np.linspace(-lim, lim, 100),
                         np.linspace(-lim, lim, 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))
    data_eigen_space = pca.inverse_transform(data)
    input = np.zeros((data_eigen_space.shape[0], 0))
    mask = np.ones_like(data_eigen_space, dtype=bool)
    tag = None
    lls = model.observations.log_likelihoods(
        data_eigen_space, input, mask, tag)

    colors = sns.xkcd_palette(color_names)

    plt.figure(figsize=(6, 6))
    for k in range(K):
        plt.contour(XX, YY, np.exp(lls[:,k]).reshape(XX.shape),
                    cmap=white_to_color_cmap(colors[k]))
    plt.xlabel("$PC_1$")
    plt.ylabel("$PC_2$")
    plt.title("Observation Distributions - "+observations)
    plt.legend()


def plot_observation_distributions(
        y, strains_compare, K, model, observations
        ):

    import itertools

    ndim = y[strains_compare[0]][0].shape[1]

    # Plot observation distributions
    lim = 0.0
    for strain in strains_compare:
        for yworm in y[strain]:
            lim = np.max((lim,np.max(np.abs(yworm))))
    lim = .85 * lim
    CC = np.meshgrid(*[np.linspace(-lim, lim, 50)]*ndim)
    data = np.column_stack([CC[i].ravel() for i in range(len(CC))])
    input = np.zeros((data.shape[0], 0))
    mask = np.ones_like(data, dtype=bool)
    tag = None
    lls = model.observations.log_likelihoods(data, input, mask, tag)

    colors = sns.xkcd_palette(color_names)

    XX,YY = np.meshgrid(*[np.linspace(-lim, lim, 50)]*2)

    for pair in list(itertools.combinations(range(ndim), 2)):
        plt.figure(figsize=(6, 6))
        for k in range(K):
            plt.contour(XX, YY, np.exp(lls[:,k]).reshape(CC[0].shape),
                        cmap=white_to_color_cmap(colors[k]))
        plt.xlabel('y_{}'.format(pair[0]))
        plt.ylabel('y_{}'.format(pair[1]))
        plt.title("Observation Distributions - "+observations)
        plt.legend()


def plot_observation_distributions_pca2(
        y, metadata, K, model, observations=None, plot_data_points=False,
        z=None, savefig=None
        ):

    # PCA eigen space
    from sklearn.decomposition import PCA

    pca = PCA(n_components = 2)
    Y = pca.fit_transform(np.concatenate(y,axis=0))

    # Plot observation distributions
    lim = 0.0
    lim = np.max(np.abs(Y),axis=0)
    lim = .85 * lim
    XX, YY = np.meshgrid(np.linspace(-lim[0], lim[0], 100),
                         np.linspace(-lim[1], lim[1], 100))
    data = np.column_stack((XX.ravel(), YY.ravel()))
    data_eigen_space = pca.inverse_transform(data)
    input = np.zeros((data_eigen_space.shape[0], 0))
    mask = np.ones_like(data_eigen_space, dtype=bool)
    tag = None
    lls = model.observations.log_likelihoods(
        data_eigen_space, input, mask, tag)

    colors = sns.xkcd_palette(color_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    handles = []
    labels = []
    for k in range(K):
        cntr = ax.contour(XX, YY, np.exp(lls[:,k]).reshape(XX.shape),
                    cmap=white_to_color_cmap(colors[k]))
        h,_ = cntr.legend_elements()
        handles.append(h[-1])
        labels.append('state {}'.format(k))
        if plot_data_points and z is not None:
            if Y[z==K].shape[0]>0:
                plt.plot(*Y[z==K][range(0,Y.shape[0],10000),:].T,
                         'o', mfc=colors[k], mec='none', ms=4)
                plt.plot(*np.max(Y[z==K],axis=0),
                         'o', mfc=colors[k], mec='none', ms=4)
    if plot_data_points and z is None:
        plt.plot(*Y[range(0,Y.shape[0],10000),:].T,
                 'o', mfc='grey', mec='none', ms=4)
        # plot largest pc1 values
        plt.plot(*Y[np.argsort(np.abs(Y[:,0])),:][-100:,:].T,
                 'o', mfc=colors[k], mec='none', ms=4)
        # plot largest pc2 values
        plt.plot(*Y[np.argsort(np.abs(Y[:,1])),:][-100:,:].T,
                 'o', mfc=colors[k], mec='none', ms=4)
    ax.legend(handles, labels)
    ax.set_xlabel("$PC_1$")
    ax.set_ylabel("$PC_2$")
    if observations is not None:
        plt.title("Observation Distributions - "+observations)
    else:
        plt.title("Observation Distributions")
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()


# Joint PDF of eigenprojections
def plot_joint_pdf_of_eigenprojections(
        y, components=[(0,2)], savefig=None, title=None
        ):
    import seaborn as sns
    import pandas as pd
    from itertools import combinations
    from os.path import join

    ndim=y[0].shape[1]

    if components == 'all':
        components = combinations(ndim,2)

    yall = pd.DataFrame(
        np.concatenate([iy[:,np.unique(components)]
                        for iy in y],axis=0),
        columns=list(np.unique(components)))

    for i1,i2 in components:
        fig = sns.jointplot(i1,i2,data=yall,kind='kde')
        if title is not None:
            fig.fig.suptitle(title)
        if savefig is not None:
            figname = join(savefig,title+'_{}_{}.png'.format(i1,i2))
            fig.savefig(figname)

def plot_accuracy_surface(res_df,x,y,z,title=None,saveto=None):
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if title is not None:
        plt.title(title)

    X, Y = np.meshgrid(res_df[x].sort_values().unique().flatten(),
                       res_df[y].sort_values().unique().flatten())
    Z = []
    for ix,iy in zip(X.flatten(),Y.flatten()):
        iz = res_df.loc[res_df[x].isin([ix]) & res_df[y].isin([iy]),z]
        if iz.empty:
            Z.append(np.nan)
        else:
            Z.append(iz.values[0])

    Z = np.array(Z).reshape(X.shape)
    vmin=np.min(Z[~np.isnan(Z)].flatten())
    vmax = np.max(Z[~np.isnan(Z)].flatten())
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=vmin,vmax=vmax,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    fig.colorbar(surf)

    if saveto is not None:
        plt.savefig(saveto)
        plt.close()
    return

