"""Common utility functions and radar relationships"""

__author__ = 'Cyril Grima'

from numpy import arcsin, cos, pi, sin, sqrt
import scipy.constants as ct


#-------------------
# Common conversions
#-------------------


def deg2rad(x):
    """Degree to Radian"""
    return x*pi/180


def rad2deg(x):
    """Radian to Degree"""
    return x*180/pi


def wf2wl(x):
    """Wavelength to frequency"""
    return ct.c/x


def wl2wf(x):
    """Wavelength to frequency"""
    return ct.c/x


def wl2wk(x):
    """Wavelength to wave number"""
    return 2*pi/x


def wf2wk(x):
    """Frequency to wave number"""
    return 2*pi/wf2wl(x)


def wk2vec(wk, th):
    """Scalar to Vectorial wave number"""
    return {'x':wk*sin(th), 'z':wk*cos(th)}


#---------------------
# Fresnel/Snell's Laws
#---------------------


def theta_i2t(x, n1, n2):
    """Incident to transmission angle conversion"""
    return arcsin(sin(x)*n1/n2)


def theta_t2i(x, n1, n2):
    """Incident to transmission angle conversion"""
    return arcsin(sin(x)*n2/n1)


def epmu2n(ep, mu):
    """Relative electric perm. and magnetic perm. to optical index"""
    return sqrt(ep*mu)



def R_v(ep1, ep2, mu1, mu2, xi):
    """Vertical (parallel) reflection coeff, funct. of incident angle"""
    n1 = epmu2n(ep1, mu1)
    n2 = epmu2n(ep2, mu2)
    xt = theta_i2t(xi, n1, n2)
    z1, z2 = sqrt(mu1/ep1), sqrt(mu2/ep2)
    return (z2*cos(xt) - z1*cos(xi)) / (z2*cos(xt) + z1*cos(xi))


def T_v(ep1, ep2, mu1, mu2, xi):
    """Vertical (parallel) transmission coeff, funct. of incident angle"""
    n1 = epmu2n(ep1, mu1)
    n2 = epmu2n(ep2, mu2)
    xt = theta_i2t(xi, n1, n2)
    z1, z2 = sqrt(mu1/ep1), sqrt(mu2/ep2)
    return 2*z2*cos(xi) / (z2*cos(xt) + z1*cos(xi))


def R_h(ep1, ep2, mu1, mu2, xi):
    """Horizontal (perpandicular) reflection coeff, funct. of incident angle"""
    n1 = epmu2n(ep1, mu1)
    n2 = epmu2n(ep2, mu2)
    xt = theta_i2t(xi, n1, n2)
    z1, z2 = sqrt(mu1/ep1), sqrt(mu2/ep2)
    return (z2*cos(xi) - z1*cos(xt)) / (z2*cos(xi) + z1*cos(xt))


def T_h(ep1, ep2, mu1, mu2, xi):
    """Horizontal (perpandicular) transmission coeff, funct. of inc. angle"""
    n1 = epmu2n(ep1, mu1)
    n2 = epmu2n(ep2, mu2)
    xt = theta_i2t(xi, n1, n2)
    z1, z2 = sqrt(mu1/ep1), sqrt(mu2/ep2)
    return 2*z2*cos(xi) / (z2*cos(xi) + z1*cos(xt))


def R(ep1, ep2, mu1, mu2, xi):
    """Unpolarized Reflection coefficient"""
    return (R_v(ep1, ep2, mu1, mu2, xi) + R_h(ep1, ep2, mu1, mu2, xi))/2.
