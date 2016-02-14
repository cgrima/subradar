"""Signal inversion"""

__author__ = 'Cyril Grima'

from scipy import integrate
import numpy as np
from numpy import cos, exp, log, log10, pi, sqrt
from importlib import import_module
from subradar import Fresnel, Signal, utils
import matplotlib.pyplot as plt


nan = float('nan')


def power_SAA(model, approx, h=nan, rx=nan, db=True, kind='isotropic gaussian',
              **kwargs):
    """Power components in the small angle approximation over circular footprint
    **kwargs: to be passed to the model (do not include theta)

    EXAMPLE:
    invert.power_SAA('iem', 'Small_S', h=1000, r=100, wf=60e6, ep2=3, sh=1., cl=100)"""

    m = import_module('subradar.' + model)
    func = getattr(m, approx)

    # Coherent Signal
    a = func(th=0, **kwargs)
    pc = a.R['nn']**2 * exp( -(2*a.wk*a.sh*cos(a.th))**2 )

    # Incoherent signal
    pn = integrate.quad(lambda x: func(th=x/h, **kwargs).nRCS(kind=kind)['hh'] \
         *pi*x, 0, rx)[0]
    pn = pn /(pi*h**2)

    ratio = pc/pn

    if db:
        pc, pn, ratio = 10*log10(pc), 10*log10(pn), 10*log10(ratio)

    return {'pc':pc, 'pn':pn, 'ratio':ratio}


def pcpn2srf_SAA(model, approx, pc, pn, wf=nan, h=nan, rx=nan, db=True,
                 kind='isotropic gaussian', ep_range=[1.4,2.5],
                 cl_logrange=[-1, 2], n=50):
    """"""
    pc = 10**(pc/10.)
    s = Signal(wf=wf, bw=nan, th=rx/h)

    ep = np.linspace(ep_range[0], ep_range[1], n)
    r = utils.R(1, ep, 1, 1, s.th)
    cl = 10**np.linspace(cl_logrange[0], cl_logrange[1], n)

    sh = sqrt(log(r**2/pc)) / (2*s.wk*cos(s.th))
    sh[np.isnan(sh)] = 0

    cl_out = np.nan * cl

    jn = n
    for i, val in enumerate(ep):
        if sh[i] != 0: #if no solution for sh, do not compute
            for j in reversed(range(0, jn, 1)):
                tmp = power_SAA(model, approx, h=h, rx=rx, wf=wf, ep2=ep[i],
                      sh=sh[i], cl=cl[j])['pn']
                if tmp/pn > 1:
                    jn = j+1
                    if jn > n:
                        jn = n
                    cl_out[i] = cl[j]
                    break

    return {'ep':ep, 'sh':sh, 'cl':cl_out}
