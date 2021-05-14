"""Various tools fpr surface detection"""

import numpy as np
import pandas as pd


def detector(rdg, y0=[], winsize=100, method='grima2012', axis=0, **kwargs):
    """Surface detection with the choosen method

    Input
    -----
    rdg: 2d-array
        radargram.

    y0: array
        Initial estimation for the location of the surface.
        Optional.

    winsize: float
        Size of the window around y0 to look for the surface.
        Activated only if y0 > 0.

    method: string
        method to use for surface detection.

    axis: 0 or 1
        Long-time axis.

    Output
    ------
    y: float
        index of the location of the detected echo.

    """
    if axis == 1:
        rdg = np.rot90(rdg)

    xsize = rdg.shape[0]
    ysize = rdg.shape[1]
    y = np.zeros(xsize)

    # Detection
    for xi in np.arange(xsize):
        signal = rdg[xi,:]

        #index vector
        if len(y0) > 0:
            idx = np.arange(winsize)+y0[xi]-winsize/2.
        else:
            idx = np.arange(ysize)
        
        # Method selection
        if method == 'grima2012':
            y[xi], c = grima2012(signal, idx=idx, **kwargs)
        if method == 'mouginot2010':
            y[xi], c = mouginot2010(signal, idx=idx, **kwargs)

    return y


def mouginot2010(signal, idx=[], period=3, window=30, **kwargs):
    """Surface detection using [Mouginot et al. 2010]
    
    Parameters
    ----------
    signal: array
        signal vector

    idx: array
        the indices of the array where to search for the echo

    period: float
        window shift to compute the noise (=1 in the original paper)

    window: float
        size of the window where to compute the noise

    Output
    ------
    y: float
        index of the location of the detected echo

    c: array
        criteria computed with idx
    """

    # array of index where to search for the surface
    idx = np.array(idx)
    if idx.size == 0 :
        idx = np.arange(len(signal)).astype(int)
    else:
        idx = np.array(idx).astype(int) # make idx an integer array

    # Estimator calculation
    noise = pd.Series(signal[idx]).shift(periods=period).rolling(window).mean().values
    #noise = [np.nanmean(signal[i-30:i-3]) for i in idx]
    c = signal[idx]/noise

    # surface index
    try:
        y = idx[np.nanargmax(c)]
    except ValueError:
        y = np.nan
    return y, c


def grima2012(signal, idx=[], **kwargs):
    """Surface detection from [Grima et al. 2012]
    
    Parameters
    ----------
    signal: array
        signal vector

    idx: array
        the indices of the array where to search for the echo

    Output
    ------
    y: float
        index of the location of the detected echo

    c: array
        criteria computed with idx
    """

    # array of index where to search for the surface
    idx = np.array(idx)
    if idx.size == 0 :
        idx = np.arange(len(signal)).astype(int)
    else:
        idx = np.array(idx).astype(int) # make idx an integer array

    # Estimator calculation
    derivative = np.roll(np.gradient(signal[idx]), 2)
    c = signal[idx]*derivative

    # surface index
    try:
        y = idx[np.nanargmax(c)]
    except ValueError:
        y = np.nan
    return y, c
