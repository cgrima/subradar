"""Signal inversion"""

from importlib import import_module
from scipy import integrate
import numpy as np
from numpy import cos, exp, log, log10, pi, sqrt
import matplotlib.pyplot as plt

from . import utils
from .Classdef import Fresnel, Signal



def srf2power_norminc(model, approx, gain=lambda th:1, th_max=np.nan,
                      db=True, kind='isotropic gaussian', **kwargs):
    """Power components over circular footprint at normal incident
    from surface properties

    PARAMETERS
    ==========
    gain: function
        function of the observation angle (radian) and return
        a linear amplitude
    th_max: float
        Maximum observation angle corresponding to the edge of the footprint
    **kwargs: to be passed to the model (do not include theta)

    EXAMPLE
    =======
    In[1]: sr.invert.srf2power_norminc('iem','Small_S', th_max=0.008,
           wf=13.78e9, ep2=1.5, sh=1.5e-3, cl=100e-3)
    Out[1]:
    {'pc': -23.172003557542929,
    'pn': -36.370086693995717,
    'ratio': 13.198083136452789}
    """
    m = import_module('subradar.' + model)
    Scattering = getattr(m, approx)

    # Coherent Signal
    a = Scattering(th=0, **kwargs)
    pc = a.R['nn']**2 * exp( -(2*a.wk*a.sh)**2 )

    # Incoherent Signal
    # Note: nRCS(th=0) approximation all over the footprint.
    # That allows to get nRCS out of the integral for faster computation
    nRCS = lambda th: Scattering(th=th, **kwargs).nRCS(kind=kind)['hh']
    integrand = lambda th: 2* gain(th)**2 * np.arctan(th)/(th**2+1) *nRCS(th)
    pn = integrate.quad(integrand, 0, th_max)[0]

    # Output
    ratio = pc/pn
    if db:
        pc, pn, ratio = 10*log10(pc), 10*log10(pn), 10*log10(ratio)
    return {'pc':pc, 'pn':pn, 'ratio':ratio}



def power2srf_norminc(model, approx, pc, pn, gain=lambda th:1, wf=np.nan,
              th_max=.1, db=True, kind='isotropic gaussian',
              ep_range=(1.4,2.5), cl_logrange=(-1, 2), n=50, verbose=False):
    """Surface properties solutions from Power components [in dB]

    EXAMPLE
    =======
    sr.invert.power2srf_norminc('iem','Small_S', pc, pn, th_max=3/1000.,wf=wf,
    verbose=True, cl_logrange=(5,), n=50)

    NOTE
    ====
    For large correlation length estimation, set cl_logrange to an array of
    1 element with what you think is the cl threshold. (see example above).
    Else, use an array of 2 elements to determine the range.
    """
    pc = 10**(pc/10.)
    s = Signal(wf=wf, wb=np.nan, th=th_max)

    ep = np.linspace(ep_range[0], ep_range[1], n)
    r = utils.R(1, ep, 1, 1, s.th)
    #cl = 10**np.linspace(cl_logrange[0], cl_logrange[1], n)

    sh = sqrt(log(r**2/pc)) / (2*s.wk*cos(s.th))
    sh[np.isnan(sh)] = 0

    # set the iterations for cl
    jn = 1 if len(cl_logrange) == 1 else n
    cl = 10**np.linspace(cl_logrange[0], cl_logrange[-1], jn)
    cl_out = np.full_like(ep, np.nan)

    # Iterations over the field of parameters
    for i, val in enumerate(ep):
        if verbose:
            print('\n')
        if sh[i] == 0: #if no solution for sh, do not compute
            continue
        for j in range(jn-1, -1, -1): # j in reversed(range(0, jn, 1)):
            tmp = srf2power_norminc(model, approx, gain=gain, th_max=th_max,
                  wf=wf, ep2=ep[i], sh=sh[i], cl=cl[j])['pn']
            if verbose:
                print('[%04d - %04d] ep = %05.2f, sh= %09.6f, cl = %08.3f, pn = %05.1f'
                      % (i, j, ep[i], sh[i], cl[j], tmp))
            if (tmp < pn) and ~np.isinf(tmp):
                jn = min(j+1, n)
                cl_out[i] = cl[j]
                break
        if (jn == 1) and (tmp > pn) and ~np.isinf(tmp):
            ep, sh, cl_out = ep[i], sh[i], cl_out[i]
            break
    # Question: why is ep returned in eps and ep?
    return {'eps':ep, 'ep':ep, 'sh':sh, 'cl':cl_out}
