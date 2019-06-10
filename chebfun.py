# chebfun.py: Some routines for dealing with Chebyshev spectral methods
#
# Created Dec 31, 2018 by Christopher L.P. Wolfe (christopher.wolfe@stonybrook.edu)

import numpy as np
import scipy as sp
from numpy import pi

def cheb(N, x1=-1, x2=1, calc_D2=False):
    '''
    Compute Chebyshev differentiation matrix and grid. The grid runs from x1 to x2, which default to -1 and 1.

    If calc_D2 is true, also calculates the second derivative matrix.
    '''

    if N == 0:
        return 0, 1

    if x1 >= x2:
        raise RuntimeError('x1 must be less than x2')

    # find the scaling parameters
    alpha = (x2 + x1)/2
    beta  = (x2 - x1)/2

    c = np.ones((1,N+1))
    c[0, 0] = 2
    c[0,-1] = 2
    c *= (-1)**np.arange(N+1)

    x = np.cos(pi*np.arange(N+1)/N)
    X = np.repeat(x[:,np.newaxis], N+1, axis=-1)
    dX = X - X.T
    D = (c.T@(1/c))/(dX + np.identity(N+1))
    D -= np.diag(D.sum(axis=-1))

    xp = alpha + beta*x[::-1]
    Dp = D[::-1,::-1]/beta

    if not calc_D2:
        return Dp, xp
    else:
        D2 = D@D
        # correct diagonal enties as per Bayless et al. (1994)
        idx = np.diag_indices_from(D2)
        D2[idx] = 0
        D2[idx] = -np.sum(D2, axis=1)

        D2p = D2[::-1,::-1]/beta**2

        return Dp, D2p, xp


def chebyshev_transform(x, u):
    N = len(x) - 1

    V = np.polynomial.chebyshev.chebvander(x, N)
    uT = (u[np.newaxis,1:-1]@V[1:-1,:]).squeeze()

    alt = (-1)**np.arange(N+1) # vector whose entries alternate sign
    a = (2/N)*((u[0]*alt + u[-1])/2 + uT)
    a[0] /= 2

    return a

def apply_operator(operator, fld):
    Ny, Nx = fld.shape

    return np.reshape(operator@fld.flatten(), (Ny, Nx))

def cheb_int(fld, D):
    FLD = sp.linalg.solve(D[1:,1:], fld.flatten()[1:])

    return FLD

def integrate_in_x(fld, D):

    if fld.ndim == 1:
        FLD = sp.linalg.solve(D[:-1,:-1], fld[:-1])
    else:
        FLD = sp.linalg.solve(D[1:,1:], fld[:,1:].T).T

    return FLD

def integrate_in_y(fld, D):
    if fld.ndim == 1:
        FLD = sp.linalg.solve(D[1:,1:], fld[1:])
    else:
        FLD = sp.linalg.solve(D[1:,1:], fld[1:,:])

    return FLD

def interp_field(fld):
    return interpn((y, x), fld,  (yyi, xxi), method='splinef2d')

