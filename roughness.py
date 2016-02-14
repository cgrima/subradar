"""Various tools for surface desciption"""

__author__ = 'Cyril Grima'

from numpy import pi, exp, sin, cos, sqrt, arcsin, float64
import matplotlib.pyplot as plt


def spectrum(wk, sh, cl, th, n=1, kind='isotropic gaussian', **kwargs):
    """Surface roughness spectrum at at order n
    """

    if kind == 'isotropic gaussian':
        out = float64(cl)**2/n*exp(-(wk*cl*sin(th))**2/n) /2.

    return out
