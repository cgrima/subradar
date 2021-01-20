"""Tools to use the Integral Equation Method (IEM) in the monostatic
(i.e. backscattering) case, when u = k*sin(th), case as described in
Fung et al. [1994].

Note: Argument "m" is subradar.Model class
"""

import numpy as np
from numpy import cos, exp, float64, inf, log10, sin
from scipy.special import factorial
from mpmath import nsum

from . import roughness
from .Classdef import Fresnel, Roughness, Signal


nan = float('nan')


class Common(Fresnel):
    """Common IEM relationships"""
    def __init__(self, **kwargs):
        Fresnel.__init__(self, **kwargs)

    def f(self):
        """f_pp terms"""
        return {'vv': 2*self.R['vv']/cos(self.th),
                'hh':-2*self.R['hh']/cos(self.th)}



class Large_S(Signal, Fresnel):
    """Large roughness single-scatter approximation [sec. 5.5]"""
    pass



class Large_M(Signal, Fresnel):
    """Large roughness multiple-scatter approximation [sec. 5.5]"""
    pass



class Small_S(Common, Signal, Fresnel, Roughness):
    """Small-to-moderate roughness single-scatter approximation [sec. 5.5]"""
    def __init__(self, **kwargs):
        Roughness.__init__(self, **kwargs)
        Fresnel.__init__(self, **kwargs)
        Signal.__init__(self, **kwargs)
        Common.__init__(self, **kwargs)


    def F_sum(self):
        """F_pp(-)+F_pp(+) terms"""

        vv = 2*sin(self.th)**2 * (1+self.R['vv'])**2 / cos(self.th) * \
              ( (1-1/self.ep) + \
                (self.mu*self.ep - sin(self.th)**2 - self.ep*cos(self.th)**2) /
                (self.ep*cos(self.th))**2
              )

        hh = -2*sin(self.th)**2 * (1+self.R['hh'])**2 / cos(self.th) * \
              ( (1-1/self.mu) + \
                (self.mu*self.ep - sin(self.th)**2 - self.mu*cos(self.th)**2) /
                (self.mu*cos(self.th))**2
              )
        return {'vv':vv, 'hh':hh}


    def I(self, n):
        """I_pp terms"""

        def _I(self, pp, n):
            return (2*self.wk_z)**n * self.f()[pp] * exp(-(self.sh*self.wk_z)**2) \
                    + self.wk_z**n * self.F_sum()[pp] /2

        return {'vv':_I(self, 'vv', n),
                'hh':_I(self, 'hh', n)}


    def nRCS(self, n='richardson+shanks', kind='isotropic gaussian', db=False):
        """Normalized Radar cross-section"""

        def _nRCS(self, pp, n, kind):
            n = float(n)
            return self.wk**2 * exp(-2*(self.sh*self.wk_z)**2) /2 * \
                   ( self.sh**(2*n) * np.abs( self.I(n)[pp] )**2 * \
                     roughness.spectrum(self.wk, self.sh, self.cl, self.th, \
                     n=n, kind=kind) / factorial(n) )

        if type(n) is not str:
            vv = nsum(lambda x: _nRCS(self, 'vv', x, kind), [1,n] )
            hh = nsum(lambda x: _nRCS(self, 'hh', x, kind), [1,n] )

        if type(n) is str:
            vv = nsum(lambda x: _nRCS(self, 'vv', x, kind), [1,inf], method=n)
            hh = nsum(lambda x: _nRCS(self, 'hh', x, kind), [1,inf], method=n)

        vv, hh = float64(vv), float64(hh)

        if db:
            vv, hh = 10*log10(vv), 10*log10(hh)

        return {'vv':vv, 'hh':hh}



class Small_M(Signal, Fresnel):
    """Small-to-moderate roughness multiple-scatter approximation [sec. 5.5]"""
    pass
