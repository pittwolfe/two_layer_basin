# chebfun.py: Some routines for dealing with Chebyshev spectral methods
#
# Created Dec 31, 2018 by Christopher L.P. Wolfe (christopher.wolfe@stonybrook.edu)

import numpy as np
import scipy as sp
from numpy import pi
from scipy.fftpack import dct, dctn

def grid(N, type='E', x1=-1, x2=1):
    r'''Generate a Chebyshev colocation grid.

    Parameters
    ----------
    N : integer
        Order of the Chebyshev grid.
    type : one of {'E', 'R'}, optional
        If `type` is 'E', the points are the extrema of the Chebyshev polynomial. If `type`
        is `R`, the points are the roots.
    x1 : float, optional
        Leftmost point of the extremum grid.
    x2 : float, optional
        Rightmost point on the extremum grid. We require x2 > x1.

    Returns
    -------
    x : numpy array
        The desired grid with points in ascending order.

    '''
    if N == 0:
        return 0

    if x1 >= x2:
        raise RuntimeError('x1 must be less than x2')

    # find the scaling parameters
    alpha = (x2 + x1)/2
    beta  = (x2 - x1)/2

    if type.upper() == 'E':
        n = np.arange(N+1)
        x = alpha - beta*np.cos(pi*n/N)
    elif type.upper() == 'R':
        n = np.arange(N)
        x = alpha - beta*np.cos(pi*(n+1/2)/N)
    else:
        RuntimeError('Unknown grid type.')

    return x

def derivative_matrix(N, source_grid='E', target_grid='E', x1=-1, x2=1, second_derivative=False):
    r'''Compute Chebyshev differentiation matrix and grid.

    Parameters
    ----------
    N : integer or array-like
        If integer, order of the Chebyshev grid. If array, assumed to be extrema grid.
        Grid is then inferred from input array and (x1, x2) are ignored.
    source_grid : one of {'E', 'R'}, optional
        Whether source grid is the extremum or root grid.
    target_grid : one of {'E', 'R'}, optional
        Whether target grid is the extremum or root grid.
    x1 : float, optional
        Physical location of the first grid point.
    x2 : float, optional
        Physical location of the last grid point.
    second_derivative : bool, optional
        If `True`, also return second order derivative matrix.

    Returns
    -------
    D : 2D numpy array
        First derivative matrix.
    D2 : 2D numpy array
        Second derivative matrix (only if `second_derivative` = `True` and only for the case
        where the source and target grids are the extremum grids)

    '''

    # We build everything on the "natural" grid, then reverse and scale.
    if not isinstance(N, int):
        x1 = N[0]
        x2 = N[-1]
        N = len(N)-1

    if N == 0:
        return 0

    if x1 >= x2:
        raise RuntimeError('x1 must be less than x2')

    # find the scaling parameters
    alpha = (x2 + x1)/2
    beta  = (x2 - x1)/2

    if source_grid.upper() == 'E' and target_grid.upper() == 'E':
        c = np.ones((1,N+1))
        c[0, 0] = 2
        c[0,-1] = 2
        c *= (-1)**np.arange(N+1)

        x = np.cos(pi*np.arange(N+1)/N)
        X = np.repeat(x[:,np.newaxis], N+1, axis=-1)
        dX = X - X.T
        D = (c.T@(1/c))/(dX + np.identity(N+1))
        D -= np.diag(D.sum(axis=-1))

        if not second_derivative:
            return D[::-1,::-1]/beta
        else:
            D2 = D@D
            # correct diagonal enties as per Bayless et al. (1994)
            idx = np.diag_indices_from(D2)
            D2[idx] = 0
            D2[idx] = -np.sum(D2, axis=1)

            D2p = D2[::-1,::-1]/beta**2

            return D[::-1,::-1]/beta, D2[::-1,::-1]/beta**2

    elif source_grid.upper() == 'E' and target_grid.upper() == 'R':
        i = np.atleast_2d(np.arange(N)).T # rows
        j = np.atleast_2d(np.arange(N+1)) # columns

        xe = np.cos(pi*j/N)         # extremum grid
        xr = np.cos(pi*(i+1/2)/N)   # root grid

        c = np.ones_like(j)
        c[:, 0] = 2
        c[:,-1] = 2

        D = (-1)**(i+j)*(1 - xe*xr)/(N*c*np.sqrt(1 - xr**2)*(xe - xr)**2)

        # correct diagonal enties as per Bayless et al. (1994)
        idx = np.diag_indices_from(D[:,:-1])
        D[idx] = 0
        D[idx] = -np.sum(D, axis=1)

        return D[::-1,::-1]/beta

    elif source_grid.upper() == 'R' and target_grid.upper() == 'E':
        i = np.atleast_2d(np.arange(1,N)).T # rows (boundaries handled as special cases
        j = np.atleast_2d(np.arange(N)) # columns

        xe = np.cos(pi*i/N)         # extremum grid
        xr = np.cos(pi*(j+1/2)/N)   # root grid

        D = np.zeros((N+1, N))
        D[0,   :] = (-1)**j*np.sqrt(1-xr**2)*((1-xr)*N**2-1)/(N*(1-xr)**2)
        D[1:-1,:] = (-1)**(i+j+1)*np.sqrt(1 - xr**2)/(N*(xe - xr)**2)
        D[-1,  :] = (-1)**(N+j)*np.sqrt(1-xr**2)*((1+xr)*N**2-1)/(N*(1+xr)**2)

        # correct diagonal enties as per Bayless et al. (1994)
        idx = np.diag_indices_from(D[:-1,:])
        D[idx] = 0
        D[idx] = -np.sum(D[:-1,:], axis=1)

        D[-1,-1] = 0
        D[-1,-1] = -np.sum(D[-1,:])

        return D[::-1,::-1]/beta

    else:
        raise RuntimeError('unknown or unimplmented combination of source and target grids')


def projection_matrix(N, source_grid='E', target_grid='E'):
    r'''Compute Chebyshev differentiation matrix and grid.

    Parameters
    ----------
    N : integer
        Order of the Chebyshev grid.
    source_grid : one of {'E', 'R'}, optional
        Whether source grid is the extremum or root grid.
    target_grid : one of {'E', 'R'}, optional
        Whether target grid is the extremum or root grid.

    Returns
    -------
    P : 2D numpy array
        Projection matrix.

    '''

    # We build everything on the "natural" grid, then reverse and scale.
    if N == 0:
        return 0

    if source_grid.upper() == 'E' and target_grid.upper() == 'E':
        return np.identity(N+1)

    elif source_grid.upper() == 'R' and target_grid.upper() == 'R':
        return np.identity(N)

    elif source_grid.upper() == 'E' and target_grid.upper() == 'R':
        i = np.atleast_2d(np.arange(N)).T # rows
        j = np.atleast_2d(np.arange(N+1)) # columns

        xj = np.cos(pi*j/N)         # extremum grid
        xi = np.cos(pi*(i+1/2)/N)   # root grid

        c = np.ones_like(j)
        c[:, 0] = 2
        c[:,-1] = 2

        P = (-1)**(i+j+1)*np.sqrt(1-xi**2)/(c*N*(xi-xj))

        return P[::-1,::-1]

    elif source_grid.upper() == 'R' and target_grid.upper() == 'E':
        i = np.atleast_2d(np.arange(N+1)).T # rows (boundaries handled as special cases)
        j = np.atleast_2d(np.arange(N)) # columns

        xi = np.cos(pi*i/N)         # extremum grid
        xj = np.cos(pi*(j+1/2)/N)   # root grid

        P = (-1)**(i+j)*np.sqrt(1-xj**2)/(N*(xi - xj))

        return P[::-1,::-1]

    else:
        raise RuntimeError('unknown or unimplmented combination of source and target grids')


def cheb(N, x1=-1, x2=1, calc_D2=False):
    r'''Compute Chebyshev differentiation matrix and grid.

    Parameters
    ----------
    N : integer
        Order of the Chebyshev grid.
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
        # correct diagonal enties as per Bayliss et al. (1994)
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

def transform_2d(u, type='E'):
    r'''2D discrete Chebyshev transform.

    Parameters
    ----------
    u : 2D array
        Values at the :math:`M+1 \times N+1` Chebyshev points.
    type : one of {'E', 'R'}
        Specifies whether the input field is on extremal ('E') or root ('R') points.

    Returns
    -------
    v : 2D array
        Chebyshev amplitudes.


    Notes
    -----

    The discrete Chebyshev transform of a function defined on extremal points is given by

    .. math::
        v_{kl} = \frac{4}{MN c_k c_j} \sum_{i=0}^M\sum_{j=0}^N \frac{u_{ij}}{c_ic_j}\cos\frac{k\pi i}{M}\cos\frac{l\pi j}{N}

    where :math:`c_k = 2` for :math:`k = 0,N` and :math:`c_k = 1` otherwise. The transform
    is implmented using the type I discrete cosine transform.

    '''
    if type.upper() == 'E':
        N = u.shape[0]-1
        M = u.shape[1]-1

        v = dctn(u, type=1, norm=None)

        # apply the scaling
        v     /= (M*N)
        v[ 0,:] /= 2
        v[-1,:] /= 2
        v[:, 0] /= 2
        v[:,-1] /= 2

    else:
        N = u.shape[0]
        M = u.shape[1]

        v = dctn(u, type=2, norm=None)
        # apply the scaling
        v     /= (M*N)
        v[ 0,:] /= 2
        v[:, 0] /= 2

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

def interp_1d(u, interp_fac, x=None):
    r'''Interpolate onto a refined Chebyshev grid.

    Parameters
    ----------
    u : array_like
        Function on Chebyshev points.
    interp_fac : integer
        Degree of grid refinement.
    x : list or tuple, optional
        First two elements are the left and right endpoints of the Chebyshev grid. Default: None

    Returns
    -------
    x  : array_like
        Refined Cheybshev grid. Only returned if x is not None on entry.
    ui : array_like
        Values at Chebyshev points.


    Notes
    -----

    Interpolates by computing the Chebyshev transform, padding to new length, then inverting
    the transform.

    '''

    N = len(u)-1

    if interp_fac == 1:
        ui = u
    else:
        ui = inverse_transform(transform(u), ndct=N*interp_fac+1)

    if x is not None:
        return grid(N*interp_fac, x1=x[0], x2=x[-1]), ui
    else:
        return ui

def clenshaw_curtis_weight(N, x1=-1, x2=1):
    r'''Clenshaw Curtis integration weights

    Notes
    -----

    From equations (A.48) and (A.49) in Peyret (2002).

    '''
    ω = np.zeros(N+1)

    if np.mod(N, 2) == 0: # N even
        i = np.atleast_2d(np.arange(1,N)).T
        k = np.atleast_2d(np.arange(N//2+1))

        c = np.ones(N//2+1)
        c[ 0] = 2
        c[-1] = 2

        ω[ 0] = 1/((N-1)*(N+1))
        ω[-1] = 1/((N-1)*(N+1))

        ω[1:-1] = (4/N)*(np.cos(2*pi*i*k/N)/c/((1-2*k)*(1+2*k))).sum(axis=-1)
    else:
        i = np.atleast_2d(np.arange(1,N)).T
        k = np.atleast_2d(np.arange((N-1)//2+1))

        c = np.ones((N-1)//2+1)
        c[ 0] = 2
        c[-1] = 2

        ω[ 0] = 1/N**2
        ω[-1] = 1/N**2

        ω[1:-1] = (4/N)*(np.cos(2*pi*i*k/N)/c/((1-2*k)*(1+2*k))).sum(axis=-1) + (2*(-1)**i*np.cos(pi*i/N)/(N**2*(2-N))).flatten()

    beta  = (x2 - x1)/2

    return ω*beta

def apply_operator(operator, fld):
    '''
    Apply an matrix operator to a 2D field.

    Parameters
    ----------
    operator : 2D ndarray
        The matrix operator to apply.
    fld : 2D ndarray
        The 2D field to which the operator is applied.

    Output
    ------
    2D ndarray
        The 2D field with the operator applied.

    '''
    Ny, Nx = fld.shape

    return np.reshape(operator @ fld.flatten(), (Ny, Nx))

def cheb_int(fld, ω=None, x=None):
    '''
    Integrate a field on a Chebysheb grid using Clenshaw-Curtis weights.

    I am not sure what the argument x is supposed to do.
    '''
    N = len(fld) - 1

    if x is None:
        β = 1
    else:
        β = (x[-1] - x[0])/2

    if ω is None:
        ω = clenshaw_curtis_weight(N)

    return np.sum(ω*fld)

def TLn(N, L, type='E', second_derivative=False):
    '''
    Calculate grid and first-order differentiation matrices for the rational Chebyshev functions.

    Parameters
    ----------
    N : int
        Order of the Chebyshev grid.
    L : float
        The scale factor. Should be roughly the size of the spatial variation of the solution.
        L can be negative. In that case, the grid comes out "backward".
    type : one of {'E', 'R'}, optional
        If `type` is 'E', the points are the extrema of the Chebyshev polynomial. If `type`
        is `R`, the points are the roots. Note the extrema grid includes the point at infinity,
        represented as np.inf
    second_derivative : bool, optional
        Whether to return the second derivative matrix (see notes).

    Returns
    -------
    x : (N+1,) numpy array
        The grid on the interval [-1, 1].
    y : (N+1,) numpy array
        The grid on the semi-infinite interval. Includes np.inf if type is 'E'.
    dy : (N+1, N+1) numpy array
        The derivative matrix. The entries for the point at infinity will be zero.
    d2y : (N+1, N+1) numpy array, optional
        The second derivative matrix. Only returned if second_derivative == True

    Notes
    -----
    The rational Chebyshev functions TLn(y) are defined by the map
        TLn(y) = Tn((y-L)/(y+L))
    where Tn are the standard Chebyshev polynomials and L can be positive or negative.

    The mapping
        x = (y-L)/(y+L)  <--> y = L(1+x)/(1-x)
    transforms the interval x in [-1, 1] to the positive or negative half line, depending
    on the sign of L.

    Derivatives transform like
        ∂y = (1/y') ∂x
        ∂2y = (1/y')^2 ∂2x - (y''/y'^3) ∂x

    There are two ways to produce the second derivative matrix: either by squaring the
    first derivative matrix or performing the mapping. The two methods produce results
    which differ by 2N/L**2, but I'm not sure which should be preferred. This routine
    returns the second derivative matrix produced by the mapping.

    '''
    x = cheb.grid(N, type=type)

    y = np.zeros_like(x)
    y[:-1] = L*(1 + x[:-1])/(1 - x[:-1])
    y[-1] = L*np.inf

    dx, d2x = cheb.derivative_matrix(N, source_grid='E', target_grid='E', second_derivative=True)

    X = x[:,np.newaxis]
    Q = (X-1)*(X-1)

    dy = Q*dx/(2*L)

    if second_derivative:
        d2y = (Q*d2x + 2*(X-1)*dx)*Q/(4*L**2)

        return x, y, dy, d2y
    else:
        return x, y, dy

