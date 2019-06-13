"""Various tools for surface desciption"""

import numpy as np

def spectrum(wk, sh, cl, th, n=1, kind='isotropic gaussian', **kwargs):
    """Surface roughness spectrum at at order n
    """

    if kind == 'isotropic gaussian':
        out = np.float64(cl)**2/n*np.exp(-(wk*cl*np.sin(th))**2/n) /2.
    else:
        raise ValueError("Unsupported value for kind: " + kind)

    return out
