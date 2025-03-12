"""Various tools for surface detection"""

import numpy as np
import pandas as pd
import scipy.signal
from . import utils


def detector(rdg, y0=None, winsize=100, method='grima2012', axis=0, out_dict=False, **kwargs):
    """Surface detection with the chosen method

    Input
    -----
    rdg: 2d-array
        radargram.

    y0: 1d-array
        Initial estimation for the location of the surface.
        Optional.

    winsize: float
        Size of the window around y0 to look for the surface.
        Activated only if y0 is a nonempty array.

    method: string
        method to use for surface detection.

    axis: 0 or 1
        Indicate which axis is long-time (slow-time) axis. default=0

    out_dict: True or False
        Return a dictionary

    Output
    ------
    y: float
        index of the location of the detected echo.

    """
    if axis == 1:
        rdg = np.transpose(rdg)

    xsize, ysize = rdg.shape
    y = np.empty(xsize)
    c = np.empty(xsize)

    detectors = {
        'maximum': maximum,
        'grima2012': grima2012,
        'mouginot2010': mouginot2010,
    }
    fdetector = detectors[method]

    # Detection
    for xi in range(xsize):
        #index vector
        if y0 is not None and len(y0) > 0:
            idx = np.arange(winsize)+y0[xi]-winsize/2.
        else:
            idx = np.arange(ysize)

        y[xi], cs = fdetector(rdg[xi, :], idx=idx, **kwargs)
        c[xi] = np.nanmax(cs)

    if out_dict:
        return {'y':y, 'c':c}
    else:
        return y


def maximum(signal, idx=(), **kwargs):
    """Surface detection from the maximum within a window
    
    Parameters
    ----------
    signal: array
        signal vector

    idx: array
        the indices of the array where to search for the echo

    Return
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
    c = signal[idx]

    # surface index
    try:
        y = idx[np.nanargmax(c)]
    except ValueError:
        y = np.nan
    return y, c


def mouginot2010(signal, idx=(), period=3, window=30, **kwargs):
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


def grima2012(signal, idx=(), **kwargs):
    """Surface detection from [Grima et al. 2012]

    Parameters
    ----------
    signal: array
        signal vector

    idx: array
        the indices of the array where to search for the echo

    Return
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
    s1 = signal[idx]
    derivative = np.roll(np.gradient(s1), 2)
    c = s1*derivative

    # surface index
    try:
        y = idx[np.nanargmax(c)]
    except ValueError:
        y = np.nan
    return y, c


def gcc(rdg, tau_threshold=2, **kwargs):
    """Surface detection from relative time delay obtained through generalized
    cross-correlation of each contiguous range lines

    Parameters
    ----------
    rdg: 2d-array
        radargram
        axis 0 is slow time, axis 1 is fast time

    Return
    ------
    """
    #---------------
    # Initialization
    xsize, ysize = rdg.shape
    yn = list(range(ysize))
    tau = np.empty(ysize, dtype=int)
    val = np.empty(ysize)
    cc = np.empty_like(rdg) #np.abs(rdg)*0
    #ch = np.empty_like(rdg) #np.abs(rdg)*0
    #offset = np.empty(ysize, dtype=int)

    #-------------------------
    # GCC applied on radargram

    # All records except last
    for i in range(ysize-1):
        _ = utils.gcc(rdg[:, i], rdg[:, i+1], **kwargs)
        tau[i] = _['tau']
        val[i] = _['val']
        cc[:,i] = _['cc']
        #ch[:,i] = _['ch']

    # Question: can we reuse the last value calculated?
    # Last record
    _ = utils.gcc(rdg[:, i], rdg[:, i-1], **kwargs)
    tau[-1] = _['tau']
    val[-1] = _['val']
    cc[:,-1] = _['cc']
    #ch[:,-1] = _['ch']

    # Quality flag when tau gradient higher than dtau_threshold
    #dtau = np.roll( np.gradient( np.abs(tau)) ,-1)
    ok = np.abs(tau) <= tau_threshold


    #----------------------------------------
    # Vertical offset that correponds to tau

    offset = np.cumsum(tau) # tau is an int, so offset should be, too


    #-------------------
    # Corrected offsets

    #Radargram rolled with offset
    rdg2 = np.empty_like(rdg)
    for i in range(ysize):
        rdg2[:,i] = np.roll(rdg[:,i], offset[i])

    # TODO: does this really need to be a nested function?
    # Radargram is divided by chunks that are bounded where ok=0
    def _data_chunks(data, stepsize=1):
        data_id = np.arange(data.size)*data
        pieces = np.split(data_id, np.where(np.diff(data_id) != stepsize)[0]+1)
        chunks = [i for i in pieces if (i.size > 1)]
        return [np.array(chunk, dtype=int) for chunk in chunks]

    chunks = _data_chunks(ok)

    # Cumulative sum of each chunk to assess the average coordinate
    # of the surface echo in each chunk
    chunk_cumsums = [np.abs(rdg2[:, chunk].sum(axis=1)) for chunk in chunks]
    chunk_cumsum_argmaxs = [np.argmax(chunk_cumsum) for chunk_cumsum in chunk_cumsums]

    # Chunks are aligned for their average surface echo coordinate to match
    offset2 = np.empty_like(offset)
    for i, chunk in enumerate(chunks):
        offset2[chunk] = offset[chunk] - chunk_cumsum_argmaxs[i] + chunk_cumsum_argmaxs[0]

    del rdg2
    # NB: if you're doing a del, you should probably make this a function

    #-------------------------------
    # Coordinate of the surface echo

    rdg3 = np.empty_like(rdg)
    for i in range(ysize):
        rdg3[:,i] = np.roll(rdg[:,i], offset2[i])
    y0 = np.argmax( np.abs(rdg3.sum(axis=1)) )
    y = y0 + offset2
    del rdg3

    # TODO: remove yn?  do you really need it?
    return {'tau':tau.astype(int), 'val':val, 'cc':cc, 'ok':ok, 'yn':yn,
    'offset':offset, 'offset2':offset2, 'y':y}
