"""Various tools for electomagntism physics

Variables comonly used in the package
-------------------------------------

bw = Bandwidth
cl : Correlation length
epsilon, ep : Electric permittivity
mp : Magnetic permeability
nRCS : Normalized Radar Cross Section
RCS : Radar Cross Section
n : Optical index
R : Reflection coefficient
sigma_h, sh : Root Mean Square height
theta, th : angle
T : Transmission coefficient
wf : Wave frequency
wk : Wave number
wl : Wavelength
"""

__author__ = 'Cyril Grima'

__all__ = ['iem', 'invert', 'roughness', 'utils']

from Classdef import *

import iem
import invert
import roughness
import utils