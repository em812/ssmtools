#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:03:48 2020

@author: em812
"""
import numpy as np
from os.path import join
import pdb

def eigen2angle(eigen_projections, n_eigen, eigenworms=None) :
    """
    Recinstructs tangent angles timeseries from eigenprojections

    Parameters
    ----------
    eigen_projections : array-like dim=n_frames x n_eigen
        Time series of the amplitudes of the eigenprojections
    eigenworms : array-like dim=n_eigen x n_segments
        the basis eigenWorms that were used in the decomposition
    n_eigen : integer
        number of eigen wrms to use for the reconstruction

    Returns
    -------
    angles : array-like dim=n_frames x n_segments

    """
    import pandas as pd

    if eigenworms is None:
        from ssmtools import AUX_FILES_DIR
        eigenworms = pd.read_csv(
            join(AUX_FILES_DIR,'eigenWorms.csv'),
            header=None, index_col=None).values

    assert eigenworms.shape[0] >= n_eigen, print(
        'The value of n_eigen is too large.' +
         'Up to {} eigenworms can be used'.format(eigenworms.shape[0]))

    assert eigen_projections.shape[1] >= n_eigen, 'Not enough eigenprojection timeseries given.'

    angles = np.matmul(eigen_projections[:,:n_eigen],eigenworms[:n_eigen,:])

    return angles

def angle2skel(angles, mean_angle=0.0, arclength=1.0):
    """
    Parameters
    ----------
    angles : array-like object dim= n_frames x n_segments
        Timeseries of tngent angles along the worm skeleton.
    mean_angle : array-like object dim = (n_frames,)
        Mean angle per frame that defines orientation of the worm.
        When 0, the skeleton will not rotate.
    arclength : float
        the total arclength of the skeleton to be reconstructed.
        Set to 1 (default) for a normalised skeleton.

    Returns
    -------
    skel_x: array=like object dim = n_frames x (n_segments+1)
        The timeseries of x coordinates of the worm sleleton
    skel_y: array=like object dim = n_frames x (n_segments+1)
        The timeseries of y coordinates of the worm sleleton
    """

    n_frames = angles.shape[0]
    n_angles = angles.shape[1]

    if isinstance(mean_angle,(float,int)):
        mean_angle = mean_angle * np.ones(n_frames)

    skel_x = []
    skel_y = []
    for ii in range(n_frames):
        # add up x-contributions of angleArray, rotated by meanAngle
        skelx = np.insert(
            np.cumsum(
                np.cos(angles[ii, :] + mean_angle[ii] ) * arclength/n_angles),
            0, 0.0)
        skel_x.append(skelx)

        # add up y-contributions of angleArray, rotated by meanAngle
        skely = np.insert(
            np.cumsum(
                np.sin(angles[ii, :] + mean_angle[ii] ) * arclength/n_angles),
            0, 0.0)
        skel_y.append(skely)

    skel_x = np.array(skel_x)
    skel_y = np.array(skel_y)

    return skel_x, skel_y


def plot_skel(
        x, y, saveto,
        make_video=True,
        remove_pngs=True,
        fps=30,
        video_name=None,
        figsize=None,
        plot_traj=True,
        worm_color=None, head_color=None, traj_color=None,
        dpi=None
        ):
    """
    Plot skeletons without regid body movement

    Parameters
    ----------
    x : array-like dim=n_frames x n_segments+1
        x coordinated of the skeleton.
    y : array-like dim=n_frames x n_segments+1
        y coordinated of the skeleton.
    saveto : path
        path where frame images and video will be saved.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    from os import system

    if traj_color is None:
        traj_color = 'grey'
    if worm_color is None:
        worm_color = 'red'
    if head_color is None:
        head_color = 'blue'

    saveto = Path(saveto)
    saveto.mkdir(exist_ok=True)
    for f in saveto.glob('*.png'):
        f.unlink()

    n_frames = x.shape[0]

    for i in range(n_frames):
        fig,ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        plt.xlim([x.min(),x.max()])
        plt.ylim([y.min(),y.max()])
        if plot_traj:
            mid = int(x.shape[1]/2)
            plt.plot(x[:i,mid],y[:i,mid],'.',color=traj_color,markersize=3)
        plt.plot(x[i,:],y[i,:],worm_color)
        plt.plot(x[i,0],y[i,0],'.',color=head_color)
        plt.savefig(join(saveto,'fig_{}'.format(i)),dpi=dpi)
        plt.close()

    if make_video:
        if video_name is None:
            video_name='video'

        system(
            "ffmpeg -r {0} -i ".format(fps) +
            "{0}/fig_%01d.png -vcodec mpeg4 -y {0}/".format(saveto) +
            "{}.mp4".format(video_name)
            )

        if remove_pngs:
            for f in Path(saveto).glob('*.png'):
                f.unlink()
    return


def skel2angles(X, Y, L=1.0):
    """
    Calculate tangent angles from skeletons

    Parameters
    ----------
    X : array-like, size = n_frames x n_segments+1
        x coordinated of the skeleton.
    Y : array-like, size = n_frames x n_segments+1
        y coordinated of the skeleton.
    L : float, optional
        the length of the plotted worm. The default is None. When None, the
        the plotted worm is normalised to a length of 1.0.

    Returns
    -------
    TX, TY : array-like, size = (n_frames) x (n_skel_points)
        x and y componenets of the unit tangent vectors.

    """
    n_skel_points = X.shape[1]

    ds = L/(n_skel_points-1)

    TX = np.zeros_like(X)
    TY = np.zeros_like(X)

    for n in range(n_skel_points):
        if n == 0:
            TX[:, n] = X[:, n+1] - X[:, n]
            TY[:, n] = Y[:, n+1] - Y[:, n]
        elif n == n_skel_points-1:
            TX[:, n] = X[:, n] - X[:, n-1]
            TY[:, n] = Y[:, n] - Y[:, n-1]
        else:
            TX[:, n] = 0.5*( X[:, n+1] - X[:, n-1] )
            TY[:, n] = 0.5*( Y[:, n+1] - Y[:, n-1] )

    TX = TX / ds
    TY = TY / ds
    Tmag = np.sqrt(np.square(TX) + np.square(TY))
    TX = TX / Tmag
    TY = TY / Tmag

    return TX, TY

def getRBM(X, Y, dt, L=1.0):
    """
    Function to get rigid body motion from x,y coordinated of the skeletons.

    Parameters
    ----------
    X : array-like, size = n_frames x n_segments+1
        x coordinated of the skeleton.
    Y : array-like, size = n_frames x n_segments+1
        y coordinated of the skeleton.
    dt: float
        the time between frames.
    L : float, optional
        the length of the plotted worm. The default is None. When None, the
        the plotted worm is normalised to a length of 1.0.

    Returns
    -------
    XCM, YCM : array-like, size = (n_frames,)
        the x and y coordinates of the centre of mass of the skeleton.
    UX, UY : array-like, size = (n_skel_points) x (n_frames - 1)
        velocity components of the skeleton points.
    UXCM, UYCM : array-like, size = (n_frames,)
        the x and y components of the centre of mass velocity
    TX, TY : array-like, size = (n_frames) x (n_skel_points)
        x and y componenets of the unit tangent vectors.
    NX, NY : array-like, size = (n_skel_points) x (n_frames)
        x and y componenets of the unit normal vectors.
    I : array-like, size = (n_frames,)
        moment of inertia of the skeleton.
    OMEG : array-like, size = (n_frames - 1,)
        angular velocity of the skeleton.


    """

    ds = L / (X.shape[1]-1)

    # Center of mass position
    XCM = ds * np.trapz(X,axis=1) / L
    YCM = ds * np.trapz(Y,axis=1) / L

    # Velocity of the centerline from the data
    UX = (X[1:,:] - X[:-1,:]) / dt
    UY = (Y[1:,:] - Y[:-1,:]) / dt

    #Velocity of the center of mass from the data
    UXCM = ds * np.trapz(UX,axis=1) / L
    UYCM = ds * np.trapz(UY,axis=1) / L

    #unit vector tangent to the worm
    TX, TY = skel2angles(X, Y, L)

    TX = -TX
    TY = -TY
    NX = -TY #unit vector normal to the worm
    NY = TX

    #Computing the angular velocity and moment of inertia from the data
    Iint = np.zeros_like(X)
    Omegint = np.zeros_like(X)

    for ii in range(X.shape[0]):
        # get integrand for moment of intertia
        Iint[ii,:] = np.square(X[ii,:] - XCM[ii]) + np.square(Y[ii,:] - YCM[ii])

        if ii > 0:
            # cross product of dX with U (for angular velocity)
            Omegint[ii - 1,:] = \
            (X[ii,:] - XCM[ii]) * (UY[ii - 1,:] - UYCM[ii - 1]) \
            - (Y[ii,:] - YCM[ii]) * (UX[ii - 1,:] - UXCM[ii - 1])

    # moment of inertia
    I = ds * np.trapz(Iint,axis=1)

    # angular velocity
    OMEG = ds*np.trapz(Omegint,axis=1) / I
    OMEG = OMEG[:-1]

    return XCM, YCM, UX, UY, UXCM, UYCM, TX, TY, NX, NY, I, OMEG


def substractRBM(X, Y, XCM, YCM, UX, UY, UXCM, UYCM, OMEG, dt):

    from numpy import matlib

    # initialise
    DX = np.zeros_like(X)
    DY = np.zeros_like(X)
    ODX = np.zeros((X.shape[0]-1, X.shape[1]))
    ODY = np.zeros((X.shape[0]-1, X.shape[1]))

    Xtil = np.zeros_like(X)
    Ytil = np.zeros_like(X)
    THETA = np.zeros(X.shape[0])
    THETA[0] = 0

    # Subtracting the rigid body motion (RBM) from the data
    for ii in range(X.shape[0]):
        DX[ii,:] = X[ii,:] - XCM[ii]
        DY[ii,:] = Y[ii,:] - YCM[ii]
        Xtil[ii,:] = DX[ii,:]
        Ytil[ii,:] = DY[ii,:]

        if ii > 0:
            # cross product of dX with U (for angular velocity)
            ODX[ii-1,:] = OMEG[ii-1] * DX[ii,:]
            ODY[ii-1,:] = OMEG[ii-1] * DY[ii,:]
            THETA[ii] = THETA[ii-1] + OMEG[ii-1] * dt

            Xtil[ii,:] = np.cos(THETA[ii])*DX[ii,:] + np.sin(THETA[ii])*DY[ii,:]
            Ytil[ii,:] = np.cos(THETA[ii])*DY[ii,:] - np.sin(THETA[ii])*DX[ii,:]

    VX = UX - matlib.repmat(UXCM.reshape(-1,1), 1, X.shape[1]) + ODY
    VY = UY - matlib.repmat(UYCM.reshape(-1,1), 1, X.shape[1]) - ODX

    return DX, DY, ODX, ODY, VX, VY, Xtil, Ytil, THETA

def posture2RBM(TX, TY, DX, DY, VX, VY, I, alpha, L=1.0):
    from numpy.linalg import lstsq

    ds = L/(TX.shape[1]-1)

    # initialise
    RBM = np.zeros((TX.shape[0]-1,3))

    # get tangential component of velocity at each skeleton point without
    # centre of mass (the wiggles)
    TdotV = TX[1:,:]*VX + TY[1:,:]*VY

    # get cross product of relative skeleton position with tangent
    DelXcrossT = DX[1:,:] * TY[1:,:] - DY[1:,:] * TX[1:,:]

    for ii in range(TX.shape[0]-1):
        # assemble right hand side
        b1 = (alpha - 1) * ds * np.trapz(TX[ii+1,:] * TdotV[ii,:])
        b2 = (alpha - 1) * ds * np.trapz(TY[ii+1,:] * TdotV[ii,:])
        b3 = (alpha - 1) * ds * np.trapz(DelXcrossT[ii,:] * TdotV[ii,:])

        # the matrix relating rigid body motion to the wiggles
        A11 = alpha * L + (1 - alpha) * ds * np.trapz(np.square(TX[ii+1,:]))
        A12 = (1 - alpha) * ds * np.trapz(TX[ii+1,:] * TY[ii+1,:])
        A13 = (1 - alpha) * ds * np.trapz(TX[ii+1,:] * DelXcrossT[ii,:])

        A22 = alpha * L + (1 - alpha) * ds * np.trapz(np.square(TY[ii+1,:]))
        A21 = A12
        A23 = (1 - alpha) * ds * np.trapz(TY[ii+1,:] * DelXcrossT[ii,:])

        A31 = A13
        A32 = A23
        A33 = alpha * I[ii+1] + (1 - alpha) * ds * np.trapz(np.square(DelXcrossT[ii,:]))

        # solve the linear system
        bvec = np.array([b1, b2, b3])
        Amat = np.array([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])
        RBM[ii,:],_,_,_ = lstsq(Amat.T, bvec, rcond=None)

    return RBM

def lab2body(x, y, theta):
    # initialise

    xp = np.zeros_like(x)
    yp = np.zeros_like(x)

    for ii in range(x.shape[0]):
        xp[ii,:] = np.cos(theta[ii])*x[ii,:] + np.sin(theta[ii])*y[ii,:]
        yp[ii,:] = np.cos(theta[ii])*y[ii,:] - np.sin(theta[ii])*x[ii,:]

    return xp, yp

def body2lab(X, Y, THETA):
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    if len(Y.shape) < 2:
        Y = Y.reshape(-1, 1)

    # initialise
    Xp = np.zeros_like(X);
    Yp = np.zeros_like(X);

    for ii in range(X.shape[0]):
        Xp[ii,:] = np.cos(THETA[ii])*X[ii,:] - np.sin(THETA[ii])*Y[ii,:]
        Yp[ii,:] = np.cos(THETA[ii])*Y[ii,:] + np.sin(THETA[ii])*X[ii,:]

    return Xp, Yp


def integrateRBM(RBM, dt, THETAr):

    Nt = RBM.shape[0]+1
    XCM = np.zeros((Nt,1))
    YCM = np.zeros((Nt,1))
    THETA = np.zeros((Nt,1))

    for ii in range(Nt):
        THETA[ii] = THETA[ii-1] + RBM[ii-1,2] * dt


    THETA = THETA - THETA[0] + THETAr[0]

    # ROTATE VELOCITIES INTO LAB FRAME
    rbm0, rbm1 = body2lab(RBM[:,0], RBM[:,1], THETA)
    RBM[:,0], RBM[:,1] = rbm0.reshape(-1), rbm1.reshape(-1)

    for ii in range(1,Nt):
        XCM[ii] = XCM[ii-1] + RBM[ii-1,0] * dt
        YCM[ii] = YCM[ii-1] + RBM[ii-1,1] * dt


    return XCM, YCM, THETA

def addRBMRotMat(Xtil, Ytil, XCM, YCM, THETA, XCMi, YCMi, THETAi):

    X = np.nan*np.ones(Xtil.shape)
    Y = np.nan*np.ones(Ytil.shape)

    for ii in range(Xtil.shape[0]):

        xt = XCM[ii] - XCM[0] + XCMi[0]
        yt = YCM[ii] - YCM[0] + YCMi[0]
        tht = THETA[ii] - THETA[0] + THETAi[0]

        XNR = Xtil[ii,:]
        YNR = Ytil[ii,:]

        Xtilt = np.cos(tht) * XNR - np.sin(tht) * YNR
        Ytilt = np.cos(tht) * YNR + np.sin(tht) * XNR

        X[ii, :] = Xtilt + xt
        Y[ii, :] = Ytilt + yt

    return X, Y

def get_moving_skeletons_from_posture(
        eigenprojections, n_eigen,
        eigen_worms=None, archlength=1.0, dt=1.0, alpha=100
        ):

    angles = eigen2angle(eigenprojections, n_eigen)
    skel_x, skel_y = angle2skel(angles, mean_angle=0.0, arclength=archlength)

    # saveto='/Users/em812/Documents/Workspace/SSM_slinder/ssmtools/ssmtools/predict_movement/savetest/posture'
    # plot_skel(skel_x[range(0,skel_x.shape[0],50),:],
    #           skel_y[range(0,skel_x.shape[0],50),:],
    #           saveto=saveto, make_video=True, remove_pngs=True, fps=10)

    XCM, YCM, UX, UY, UXCM, UYCM, TX, TY, NX, NY, I, OMEG = getRBM(
        skel_x, skel_y, dt, L=archlength)
    DX, DY, ODX, ODY, VX, VY, Xtil, Ytil, THETA = substractRBM(
        skel_x, skel_y, XCM, YCM, UX, UY, UXCM, UYCM, OMEG, dt=dt)

    TX, TY = lab2body(TX, TY, THETA)
    VX, VY = lab2body(VX, VY, THETA)

    RBM = posture2RBM(TX, TY, DX, DY, VX, VY, I, alpha, L=archlength)

    # calculate the predicted rigid body motion
    XCMrecon, YCMrecon, THETArecon = integrateRBM(RBM, dt, THETA)

    Xrecon, Yrecon = addRBMRotMat(
        Xtil, Ytil, XCMrecon, YCMrecon, THETArecon, XCM, YCM, THETA)


    return Xrecon, Yrecon



if __name__=="__main__":
    import pandas as pd
    from ssmtools import AUX_FILES_DIR

    eigen_projections = pd.read_csv(
        join(AUX_FILES_DIR,'test_data','eigenprojections.csv'),
        header=None).values

    #eigen_projections = eigen_projections.T
    n_eigen = 6

    angles = eigen2angle(eigen_projections, n_eigen)
    skel_x, skel_y = angle2skel(angles, mean_angle=0.0, arclength=1.0)

    # saveto='/Users/em812/Documents/Workspace/SSM_slinder/ssmtools/ssmtools/predict_movement/savetest/posture'
    # plot_skel(skel_x[range(0,skel_x.shape[0],50),:],
    #           skel_y[range(0,skel_x.shape[0],50),:],
    #           saveto=saveto, make_video=True, remove_pngs=True, fps=10)

    XCM, YCM, UX, UY, UXCM, UYCM, TX, TY, NX, NY, I, OMEG = getRBM(skel_x, skel_y, 1, L=1.0)
    DX, DY, ODX, ODY, VX, VY, Xtil, Ytil, THETA = substractRBM(skel_x, skel_y, XCM, YCM, UX, UY, UXCM, UYCM, OMEG, dt=1)

    TX, TY = lab2body(TX, TY, THETA)
    VX, VY = lab2body(VX, VY, THETA)

    alpha = 100 # large alpha corresponds to no-slip during crawling
    RBM = posture2RBM(TX, TY, DX, DY, VX, VY, I, alpha, L=1.0)

    # calculate the predicted rigid body motion
    XCMrecon, YCMrecon, THETArecon = integrateRBM(RBM, 1, THETA)

    Xrecon, Yrecon = addRBMRotMat(Xtil, Ytil, XCMrecon, YCMrecon, THETArecon, XCM, YCM, THETA)

    saveto='/Users/em812/Documents/Workspace/SSM_slinder/ssmtools/ssmtools/predict_movement/savetest/reconstructed'
    plot_skel(Xrecon[range(0,Xrecon.shape[0],50),:],
              Yrecon[range(0,Xrecon.shape[0],50),:],
              saveto=saveto, make_video=True, remove_pngs=True, fps=10)
