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

    def ft(self):
        """f_ppt terms
        !! FOR NORMAL INCIDENCE ONLY !!
        """

        nr = np.sqrt(self.mu/self.ep)

        vv = -(1-self.R['vv']) - (1+self.R['vv'])*nr #eq 4D.1
        hh = +(1+self.R['hh']) + (1-self.R['hh'])*nr #eq 4D.2

        return {'vv': vv, 'hh': hh}


class Large_S(Signal, Fresnel):
    """Large roughness single-scatter approximation [sec. 5.5]"""



class Large_M(Signal, Fresnel):
    """Large roughness multiple-scatter approximation [sec. 5.5]"""



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


    def Ft(self):
        """Ft terms
        !! FOR NORMAL INCIDENCE ONLY !!
        """

        k = self.wk
        kt = self.n2*self.wk
        R = self.R['nn']
        er = self.ep
        mr = self.mu
        nr = np.sqrt(mr/er)

        s = 0
        st = 0
        cs = 1
        cst = 1
        sf = 0
        csf = -1

        sq = np.sqrt(self.mu*self.ep)
        rem = sq
        nr = np.sqrt(self.mu/self.ep)
        a1 = (st*s-rem*csf)/(sq*cst)
        a2 = s*(st-s*csf/rem)/cst
        Tv = 1+self.R['vv']
        Tvm = 1-self.R['vv']
        Th = 1+self.R['hh']
        Thm = 1-self.R['hh']
        Tp = 1+self.R['nn']
        Tm = 1-self.R['nn']
        R = (self.R['vv']-self.R['hh'])/2

        #vv = - (cs*Tvm-sq*Tv/self.ep) * (Tv*csf+Tvm*a1) \
        #     - (Tvm**2-Tv*Tvm*cs/sq)*a2# eq 4D.41
        #hh = (cs*Thm-sq*Th/self.mu) * (Th*csf+Thm*a1)*nr \
        #     + (Thm**2-Th*Thm*cs/sq)*a2*nr # eq 4D.42

        #vv = -(Tvm-Tv*nr)*(-Tv+Tvm)
        #hh =  (Thm-Th/nr)*(-Th-Thm)*nr

        # Chris derivation (2022/09/12):

        vv = -k*(
             -(1-R)*( (1+R)/np.sqrt(k**2) + (-1+R)*er/np.sqrt(kt**2) )*nr
             +(1+R)*((-1+R)/np.sqrt(k**2) + ( 1+R)*mr/np.sqrt(kt**2) )
             )

        hh = -k*(
              (1+R)*( (1-R)/np.sqrt(k**2) - ( 1+R)*er/np.sqrt(kt**2) )*nr
             -(1-R)*(-(1+R)/np.sqrt(k**2) - (-1+R)*mr/np.sqrt(kt**2) )
             )


        return {'vv':vv, 'hh':hh}


    def I(self, n):
        """I_pp terms"""

        def _I(self, pp, n):
            return (2*self.wk_z)**n * self.f()[pp] *\
                   exp(-(self.sh*self.wk_z)**2) \
                   + self.wk_z**n * self.F_sum()[pp] /2

        return {'vv':_I(self, 'vv', n),
                'hh':_I(self, 'hh', n)}


    def It(self, n):
        """I_ppt terms
        !! FOR NORMAL INCIDENCE !!
        """

        def _It(self, pp, n):

            kz = self.wk_z
            ktz = self.n2*self.wk_z

            #return (1+self.n2)**n * self.wk_z**n * self.ft()[pp] *\
            #       exp(-self.n2*self.sh**2*self.wk_z**2) \
            #       + (1+self.n2**n) * self.wk_z**n * self.Ft()[pp]

            return (kz+ktz)**n * self.ft()[pp] *\
                   exp(-self.sh**2*kz*ktz) \
                   + (ktz**n*self.Ft()[pp] + kz**n*self.Ft()[pp] )/2

        return {'vv':_It(self, 'vv', n),
                'hh':_It(self, 'hh', n)
                }


    def nRCS(self, n='richardson+shanks', kind='isotropic gaussian', db=False,
            transmission=False):
        """Normalized Radar cross-section"""

        def _nRCS(self, pp, n, kind, transmission):
            n = float(n)

            if not transmission:
                out = self.wk**2 * exp(-2*(self.sh*self.wk_z)**2) /2 * \
                      ( self.sh**(2*n) * np.abs( self.I(n)[pp] )**2 * \
                      roughness.spectrum(self.wk, self.sh, self.cl, self.th, \
                      n=n, kind=kind) / factorial(n) )

            elif transmission:
                out = self.n2**2 * self.wk**2 * \
                      exp(-(1+self.n2**2)*self.sh**2*self.wk_z**2) /2 * \
                      ( self.sh**(2*n) * np.abs( self.It(n)[pp] )**2 * \
                      roughness.spectrum(self.wk, self.sh, self.cl, self.th, \
                      n=n, kind=kind) / factorial(n) )

            return out

        if not isinstance(n, str):
            vv = nsum(lambda x: _nRCS(self, 'vv', x, kind, transmission), [1,n] )
            hh = nsum(lambda x: _nRCS(self, 'hh', x, kind, transmission), [1,n] )
        else:
            vv = nsum(lambda x: _nRCS(self, 'vv', x, kind, transmission), [1,inf], method=n)
            hh = nsum(lambda x: _nRCS(self, 'hh', x, kind, transmission), [1,inf], method=n)

        vv, hh = float64(vv), float64(hh)

        if db:
            vv, hh = 10*log10(vv), 10*log10(hh)

        return {'vv':vv, 'hh':hh}



class Small_M(Signal, Fresnel):
    """Small-to-moderate roughness multiple-scatter approximation [sec. 5.5]"""
