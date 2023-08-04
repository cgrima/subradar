#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '..')

import subradar as sr


def make_radargram(shape, srfy, walkstd, srfampl, bgampl, rng=None):
    """ Generate a fake radargram with specified dimensions,
    a surface return at, on average, surface location y, with amplitude
    srfampl, and a background amplitude given by bgampl
    shape[0] is the size of the slow time axis and
    shape[1] is the fast time axis """
    assert len(shape) == 2, "Must be a 2D array"
    assert shape[0] > 0 and shape[1] > 0, "Invalid shape"

    if rng is None:
        rng = np.random.default_rng()

    rdg = np.abs(rng.normal(loc=0., scale=bgampl, size=shape))

    # surface walk
    walk = rng.normal(loc=0., scale=walkstd, size=(shape[0],))
    srfloc = srfy + np.cumsum(walk)
    # Round to nearest whole fast time index
    srfloc = np.rint(np.clip(srfloc, 0, shape[1] - 1))

    srfampls = rng.normal(loc=srfampl, scale=srfampl/50., size=(shape[0],))
    for ii in range(shape[0]):
        rdg[ii, int(srfloc[ii])] = srfampls[ii]

    return rdg


class TestClassdef(unittest.TestCase):
    """ Instantiate things in the Classdef file """
    def test_roughness(self):
        x = sr.Classdef.Roughness()
    def test_fresnel(self):
        x = sr.Classdef.Fresnel()
    def test_signal(self):
        x = sr.Classdef.Signal()

class TestIEM(unittest.TestCase):
    """ Test classes/functions in iem.py module """

    def run_common_methods(self, c):
        x = c.f()
        x = c.ft()

    def test_common(self):
        c = sr.iem.Common()
        self.run_common_methods(c)

    def test_large_s(self):
        c = sr.iem.Large_S()

    def test_large_m(self):
        c = sr.iem.Large_M()

    def test_small_m(self):
        c = sr.iem.Small_M()

    def test_small_s(self):
        c = sr.iem.Small_S()
        self.run_common_methods(c)
        x = c.F_sum()
        x = c.Ft()

        for n in range(5):
            x = c.I(n)
            x = c.It(n)

        for db in (False, True):
            for tx in (False, True):
                x = c.nRCS(db=db, transmission=tx)
                y = c.nRCS(db=db, transmission=tx, n=5.1) # valid value for n?



class TestSurface(unittest.TestCase):
    """ Test functions in the surface module """
    def setUp(self):
        self.rng = np.random.default_rng(seed=0xbeef)
        shape = (1000, 800)
        self.rdg = 0.01 + make_radargram(shape, 200, walkstd=2,
                                  srfampl=100, bgampl=0.1, rng=self.rng)

        self.rdg_noise = 0.01 + np.random.normal(0.0, 0.1, size=shape)

    def test_detect_estimate(self):
        """ Test detection using an initial estimate (a good one and a bad one)
        """
        for method in ('grima2012', 'mouginot2010'):
            for y0 in (200, 500):
                with self.subTest(method=method, y0=y0):
                    y0arr = np.full(self.rdg.shape[0:1], y0)
                    x = sr.surface.detector(self.rdg, y0=y0arr, method=method)
                    self.assertEqual(self.rdg.shape[0], len(x))


    def test_detect_method(self):
        for method in ('grima2012', 'mouginot2010'):
            with self.subTest(method=method):
                x = sr.surface.detector(self.rdg, method=method)
                self.assertEqual(self.rdg.shape[0], len(x))

    def test_zeroes(self):
        """ Test with a radargram that is all zeros """
        rdg = np.zeros_like(self.rdg)
        for method in ('grima2012', 'mouginot2010'):
            with self.subTest(method=method):
                x = sr.surface.detector(rdg, method=method)
                self.assertEqual(rdg.shape[0], len(x))

    def test_nosurface(self):
        """ Test with a radargram with no surface in it """
        for method in ('grima2012', 'mouginot2010'):
            with self.subTest(method=method):
                x = sr.surface.detector(self.rdg_noise, method=method)
                self.assertEqual(self.rdg_noise.shape[0], len(x))


    def test_transpose(self):
        rdg = self.rdg.T.copy()
        x = sr.surface.detector(rdg, axis=1)


    def test_gcc(self):
        for method in 'standard wiener roth scot phat ml'.split():
            with self.subTest(weight=method):
                info = sr.surface.gcc(self.rdg, weight=method)

        with self.assertRaises(ValueError):
            info = sr.surface.gcc(self.rdg, weight='unknown!')


class TestRoughness(unittest.TestCase):
    """ test the roughness module """
    def test_isotropic_gaussian(self):
        x = sr.roughness.spectrum(0., 0., 0., 0.)
    def test_unknown(self):
        with self.assertRaises(ValueError):
            x = sr.roughness.spectrum(0., 0., 0., 0., kind='unknown')


class TestUtils(unittest.TestCase):
    """ Test utils that don't otherwise get run """
    def test_geo_loss(self):
        h = np.arange(1, 1000, 0.5)
        y = sr.utils.geo_loss(h)

    def test_funcs1(self):
        x = np.arange(-2*np.pi, 2*np.pi, 0.01)
        y = sr.utils.theta_t2i(x, 1.0, 1.0)
        y = sr.utils.theta_t2i(x, 1.5, 0.5)
    def test_funcs2(self):
        x = np.arange(-np.pi, np.pi, 0.01)
        y = sr.utils.wl2wk(x)
        y = sr.utils.wl2wf(x)
        y = sr.utils.deg2rad(sr.utils.rad2deg(x))
        

def main():
    _ = make_radargram((10, 10), 200, walkstd=2,
                         srfampl=100, bgampl=0.1, rng=None)

    unittest.main()
    #fig, ax = plt. subplots(2, 1, samex=True)
    #ax[0].imshow(rdg.T)
    #ax[1].plot(x)
    #plt.show()

if __name__ == "__main__":
    main()
