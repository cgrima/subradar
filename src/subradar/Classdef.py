"""Various Python classes"""

__author__ = 'Cyril Grima'

import numpy as np

from . import utils

NAN = float('nan')


class Signal:
    """Signal relationships"""
    def __init__(self, wf=NAN, bw=NAN, th=0., bmw=NAN, h=NAN, **kwargs):
        self.wf = wf  # Signal central frequency [Hz]
        self.bw = bw  # Signal bandwidth [Hz]
        self.th = th  # Incident angle [rad]
        self.bmw = bmw  # Beamwidth [rad]
        self.h = h  # Altitude [m]

        # Defined from above variables
        self.wl = utils.wf2wl(wf)
        self.wk = utils.wf2wk(wf)
        self.wk_x = utils.wk2vec(self.wk, self.th)['x']
        self.wk_z = utils.wk2vec(self.wk, self.th)['z']
        self.footprint_rad = {'beam':utils.footprint_rad_beam(self.h, self.bmw),
                             'pulse':utils.footprint_rad_pulse(self.h, self.bw),
                             'fresnel':utils.footprint_rad_fresnel(self.h, self.wl)
                             }


class Fresnel:
    """Fresnel and Snell's relationships"""
    def __init__(self, ep1=1., ep2=1., mu1=1., mu2=1., th=0, **kwargs):
        self.ep1 = ep1  # Electric permittivity 1
        self.ep2 = ep2  # Electric permittivity 2
        self.ep = ep2/ep1  # Relative electric permittivity
        self.mu1 = mu1  # Magnetic permeability 1
        self.mu2 = mu2  # Magnetic permeability 2
        self.mu = mu2/mu1  # Relative magnetic permeability 2/1
        self.th = th  # Incident angle [rad]
        self.n1 = utils.epmu2n(ep1, mu1)
        self.n2 = utils.epmu2n(ep2, mu2)
        self.R = {'vv':utils.R_v(self.ep1, self.ep2, self.mu1, self.mu2, self.th),
                 'hh':utils.R_h(self.ep1, self.ep2, self.mu1, self.mu2, self.th),
                 'nn':utils.R(self.ep1, self.ep2, self.mu1, self.mu2, self.th)
                  }
        self.T = {'vv':utils.T_v(self.ep1, self.ep2, self.mu1, self.mu2, self.th),
                 'hh':utils.T_h(self.ep1, self.ep2, self.mu1, self.mu2, self.th)
                  }


class Roughness:
    """Roughness relationships"""
    def __init__(self, wf=NAN, sh=0, cl=np.inf, **kwargs):
        self.wf = wf  # Signal central frequency [Hz]
        self.sh = sh  # RMS  height [m]
        self.cl = cl  # Correlation Length [m]

        # Defined from above variables
        self.wl = utils.wf2wl(wf)
        self.wk = utils.wf2wk(wf)
        self.ks = self.wk*self.sh
        self.kl = self.wk*self.cl
