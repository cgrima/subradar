import rsr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal 
from scipy import fft
import scipy.constants as ct
from scipy import integrate
from scipy import special
from scipy import constants
import subradar as sr
import pandas as pd
import glob
from labellines import labelLine, labelLines
import seaborn as sns

# -------------------------------
# TODO: INTEGRATION WITH THE REPO
# -------------------------------


def dB(func):
    """Decorator to convert a linear power to dB power
    """
    def func_wrapper(*args, dB=True, **kwargs):
        out = func(*args, **kwargs)
        if dB is True:
            return 10*np.log10(out)
        else:
            return out
    return func_wrapper


def deg2frac(x):
    """Slope Degree to fraction conversion
    """
    return np.tan(np.deg2rad(x))


def frac2deg(x):
    """Fraction to Slope degree conversion
    """
    return np.rad2deg(np.arctan(x))


class Scattering:
    """Fractal Model"""
    def __init__(self, 
                 wf = None, # Wave frequency
                 wb = None, # Bandwidth
                 Z = None, # Surface altitude
                 Re = None, # Effective reflectivity
                 H = None, # Husrt exponent
                 RMSh_1 = None, # RMS Height at 1m
                 phi = 0. # Backscattering angle
                ):
        self.wf = wf
        self.wb = wb
        self.Z = Z
        self.Re = Re
        self.H = H
        self.RMSh_1 = RMSh_1
        self.phi = phi
        self.wl = constants.c/self.wf

    def RMSh(self, scale=None):
        """RMS Height at the chosen scale (scale = wavelength by default)
        [Shepard et al, 1999(2a)]
        """
        if scale is None:
            scale = self.wl
        out = self.RMSh_1*(scale)**self.H
        return out
    
    def RMSd(self, scale=None):
        """RMS Derivation at the chosen scale (scale = wavelength by default)
        """
        if scale is None:
            scale = self.wl
        out = self.RMSh(scale=scale)*np.sqrt(2)
        return out
    
    def RMSs(self, scale=None, deg=False):
        """RMS slope at the chosen scale (scale = wavelength by defautl)
        [Shepard et al, 1999(1c)]
        """
        if scale is None:
            scale = self.wl
        out = self.RMSd(scale=scale)/scale
        if deg is True:
            return np.rad2deg(np.arctan(out))
        return out
    
    @dB
    def geometric_loss(self):
        """1-way loss due to geometric spreading"""
        out = (1/4/np.pi/self.Z**2)
        return out
    
    @dB
    def pulselimited_footprint(self, area=True, norm=True):
        """Pulse-limited footprint radius OR area
        """
        out = np.sqrt(constants.c*self.Z/self.wb) # Radius
        if area is True:
            out = np.pi*out**2 # Area
        if norm is True:
            out = out/self.wl
        return out
    
    @dB
    def fresnel_zone(self, area=True, norm=True):
        """Fresnel zone radius OR area
        """
        out = np.sqrt(self.Z*self.wl/2)
        if area is True:
            out = np.pi*out**2 # Area
        if norm is True:
            out = out/self.wl
        return out
    
    def coherent_effective_radius(self, n=5, norm=True):
        """Effective radius relative to wavelength
        [Shepard et al, 1999(20)]
        """
        frac = n/(4*np.pi**2*self.RMSs()**2*np.cos(self.phi)**2)
        out = (frac)**(1/2/self.H)
        if norm is not True:
            out = out*self.wl
        return out
    
    def incoherent_limit_radius(self, norm=True):
        """Radius below which the incoherent model is not valid
        """
        pass
    
    @dB
    def flat_RCS(self, norm=False, Re=1.):
        """Perfect-reflector normalized radar cross-section
        """
        #A = self.fresnel_zone(area=True, dB=False)
        #out = 4*np.pi*A**2/self.wl
        out = self.coherent_RCS(Re=Re, H=0., sl=0., norm=False, dB=False)
        if norm is True:
            out = out/A
        return out
    
    
    @dB
    def coherent_RCS(self, model='Campbell2003', norm=False,
                    Re=None, H=None, sl=None, phi=None, rmax=None):
        """Coherent normalized radar cross-section
        """
        # Parameters
        if Re is None:
            Re = self.Re 
        if sl is None:
            sl = self.RMSs()
        if H is None:
            H = self.H
        if phi is None:
            phi = self.phi
        if rmax is None:
            #rmax = self.pulselimited_footprint(area=False, dB=False)
            rmax = self.fresnel_zone(area=False, dB=False, norm=True)
            #rmax = self.coherent_effective_radius()
        
        if model == 'Campbell2003':
            _integrand = lambda x: x * np.exp(-4*np.pi**2*sl**2*x**(2*H)*\
                         np.cos(phi)**2)*special.j0(4*np.pi*x*np.sin(phi))
            K, err = integrate.quad(_integrand, 0, rmax)
            #out = 16*(np.pi*Re*K/rmax)**2 #nRCS
            out = 16*np.pi**3*self.wl**2*Re**2*K**2 #RCS
            
        if norm is True:
            out = out/(np.pi*rmax**2*self.wl**2)
            
        return out

    @dB
    def incoherent_RCS(self, model='Biccari2001', norm=False, 
                      Re=None, H=None, sl=None, phi=None, rmax=None):
        """Incoherent normalized radar cross-section
        """
        # Parameters
        if Re is None:
            Re = self.Re 
        if sl is None:
            sl = self.RMSs()
        if H is None:
            H = self.H
        if phi is None:
            phi = self.phi
        if rmax is None:
            rmax = self.pulselimited_footprint(area=False, dB=False)
        
        if model == 'Biccari2001': #nRCS
            if (H == 0) or (sl == 0):
                return 0.
            elif phi == 0.:
                K = special.gamma(1/H)
            else:
                K = nsum(lambda n: 
                         (-1)**n*np.sin(phi)**(2*n)/\
                         np.math.factorial(n)**2*\
                         ((2*np.pi)**(H-1))**(2*n/H)/\
                         (sl*np.sqrt(2)*np.cos(phi))**(2*n/H)*\
                         np.math.factorial((n+1)/H-1),\
                         [0, inf])
            try:
                out = K*Re**2/H* ((2*np.pi)**(H-1)/sl/np.sqrt(2))**(2/H)
            except ZeroDivisionError:
                out = 0
            out = np.float64(out)
            
        out_RCS = out*(np.pi*rmax)**2*self.wl**2
            
        if norm is False:
            out = out_RCS
        
        if out_RCS > self.flat_RCS(Re=self.Re, dB=False, norm=False):
            out = 0
            
        return out
    
    @dB
    def total_RCS(self):
        """Total normalized radar cross-section
        """
        _sc = self.coherent_RCS(norm=False, dB=False)
        _sn = self.incoherent_RCS(norm=False, dB=False)
        if _sn > self.flat_RCS(norm=False, dB=False, Re=self.Re):
            out = _sc
        else:
            out = _sc + _sn
        return out