"""Various tools for electomagntism physics
Variables comonly used in the package
-------------------------------------

bmw = Beamwidth angle
bw = Bandwidth
cl : Correlation length
epsilon, ep : Electric permittivity
h = Altitude
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

__version__ = "1.0.0"
__author__ = "Cyril Grima"

__all__ = ["iem", "invert", "roughness", "utils"]

from . import iem, invert, roughness, utils
