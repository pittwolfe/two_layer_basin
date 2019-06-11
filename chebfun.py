# chebfun.py: Some routines for dealing with Chebyshev spectral methods
#
# Created Dec 31, 2018 by Christopher L.P. Wolfe (christopher.wolfe@stonybrook.edu)

import numpy as np
import scipy as sp
from numpy import pi
from scipy.fftpack import dct, dctn

def grid(N, x1=-1, x2=1):
    if N == 0:
        return 0

    if x1 >= x2:
        raise RuntimeError('x1 must be less than x2')

    # find the scaling parameters
    alpha = (x2 + x1)/2
    beta  = (x2 - x1)/2

    x = np.cos(pi*np.arange(N+1)/N)
    xp = alpha + beta*x[::-1]

    return xp

def cheb(N, x1=-1, x2=1, calc_D2=False):
    r'''Compute Chebyshev differentiation matrix and grid.

    Parameters
    ----------
    N : integer
        Order of the Chebeshev grid.
    x1 : float, optional
        Physical location of the first grid point.
    x2 : float, optional
        Physical location of the last grid point.
    calc_D2 : bool, optional
        If `True`, also return second order derivative matrix.

    Returns
    -------
    x : 1D numpy array
        The Chebeshev grid.
    D : 2D numpy array
        First derivative matrix.
    D2 : 2D numpy array
        Second derivative matrix (only if `calc_D2` = `True`)

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
        return xp, Dp
    else:
        D2 = D@D
        # correct diagonal enties as per Bayless et al. (1994)
        idx = np.diag_indices_from(D2)
        D2[idx] = 0
        D2[idx] = -np.sum(D2, axis=1)

        D2p = D2[::-1,::-1]/beta**2

        return xp, Dp, D2p


def transform(u):
    r'''Discrete Chebyshev transform.

    Parameters
    ----------
    u : array_like
        Array of values at :math:`N+1` Chebyshev points.

    Returns
    -------
    v : array_like
        Array of Chebyshev amplitudes.


    Notes
    -----

    The discrete Chebyshev transform is given by

    .. math::
        v_k = \frac{c_k}{N}\left[u_0 + (-1)^k u_N + 2\sum_{i=1}^{N-1} u_i \cos\frac{k\pi i}{N} \right]

    where :math:`c_k = 1/2` for :math:`k = 0,N` and :math:`c_k = 1` otherwise. The transform
    is implmented using the type I discrete cosine transform.

    '''
    N = len(u)-1

    v = dct(u, axis=0, type=1, norm=None)

    # apply the scaling
    v     /= N
    v[0]  /= 2
    v[-1] /= 2

    return v

def inverse_transform(v, ndct=None):
    r'''Inverse discrete Chebyshev transform.

    Parameters
    ----------
    v : array_like
        Array of :math:`N+1` Chebyshev coefficients.
    ndct : integer, optional
        Length of the transform. If `ndct` :math:`> N+1`, `v` is zero padded. If None, then
        `ndct` :math:`= N+1`.

    Returns
    -------
    u : array_like
        Values at chebyshev points.


    Notes
    -----

    The inverse discrete Chebyshev transform is given by

    .. math::
        u_i = \sum_{k=0}^N v_k \cos\frac{k\pi i}{N}.

    The transform is implmented using the type I discrete cosine transform.

    '''
    N = len(v)
    if ndct is not None and ndct > len(v):
        V = np.zeros(ndct)
        V[:N] = v
    else:
        V = np.array(v)

    # pretreat the coefficients
    V[1:-1] /= 2

    u = dct(V, axis=0, type=1, norm=None)

    return u

def transform_2d(u):
    r'''2D discrete Chebyshev transform.

    Parameters
    ----------
    u : 2D array
        Values at the :math:`M+1 \times N+1` Chebyshev points.

    Returns
    -------
    v : 2D array
        Chebyshev amplitudes.


    Notes
    -----

    The discrete Chebyshev transform is given by

    .. math::
        v_{kl} = \frac{4}{MN c_k c_j} \sum_{i=0}^M\sum_{j=0}^N \frac{u_{ij}}{c_ic_j}\cos\frac{k\pi i}{M}\cos\frac{l\pi j}{N}

    where :math:`c_k = 2` for :math:`k = 0,N` and :math:`c_k = 1` otherwise. The transform
    is implmented using the type I discrete cosine transform.

    '''
    N = u.shape[0]-1
    M = u.shape[1]-1

    v = dctn(u, type=1, norm=None)

    # apply the scaling
    v     /= (M*N)
    v[ 0,:] /= 2
    v[-1,:] /= 2
    v[:, 0] /= 2
    v[:,-1] /= 2

    return v

def inverse_transform_2d(v, ndct=None):
    r'''Inverse discrete Chebyshev transform.

    Parameters
    ----------
    v : 2D array
        :math:`M+1 \ times N+1` Chebyshev coefficients.
    ndct : tuple of two integers, optional
        Length of the transforms in two directions. Extra entries are padded. If `None` then
        transforms use the original dimensions of `v`.

    Returns
    -------
    u : 2D array
        Values at chebyshev points.


    Notes
    -----

    The inverse discrete Chebyshev transform is given by

    .. math::
        u_{ij} = \sum_{k=0}^M\sum_{l=0}^N v_{kl} \cos\frac{k\pi i}{M} \cos\frac{l\pi j}{N}.

    The transform is implmented using the type I discrete cosine transform.

    '''
    N = v.shape[0]
    M = v.shape[1]

    if ndct is not None:
        V = np.zeros(ndct)
        V[:M,:N] = v
    else:
        V = np.array(v)

    # pretreat the coefficients
    V[:,1:-1] /= 2
    V[1:-1,:] /= 2

    u = dctn(V, type=1, norm=None)

    return u

def inverse_transform_2d(v, ndct=None):
    r'''Inverse discrete Chebyshev transform.

    Parameters
    ----------
    v : 2D array
        :math:`M+1 \ times N+1` Chebyshev coefficients.
    ndct : tuple of two integers, optional
        Length of the transforms in two directions. Extra entries are padded. If `None` then
        transforms use the original dimensions of `v`.

    Returns
    -------
    u : 2D array
        Values at chebyshev points.


    Notes
    -----

    The inverse discrete Chebyshev transform is given by

    .. math::
        u_{ij} = \sum_{k=0}^M\sum_{l=0}^N v_{kl} \cos\frac{k\pi i}{M} \cos\frac{l\pi j}{N}.

    The transform is implmented using the type I discrete cosine transform.

    '''
    N = v.shape[0]
    M = v.shape[1]

    if ndct is not None:
        V = np.zeros(ndct)
        V[:M,:N] = v
    else:
        V = np.array(v)

    # pretreat the coefficients
    V[:,1:-1] /= 2
    V[1:-1,:] /= 2

    u = dctn(V, type=1, norm=None)

    return u

def interp_1d(u, interp_fac):
    r'''Interpolate onto a refined Chebyshev grid.

    Parameters
    ----------
    u : array_like
        Function on Chebyshev points.
    interp_fac : integer
        Degree of grid refinement.

    Returns
    -------
    ui : array_like
        Values at Chebyshev points.


    Notes
    -----

    Interpolates by computing the Chebyshev transform, padding to new length, then inverting
    the transform.

    '''

    if interp_fac == 1:
        return u
    else:
        return inverse_transform(transform(u), ndct=(len(u)-1)*interp_fac+1)


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

