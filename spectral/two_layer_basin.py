# two_layer_basin.py: Class for two layer model
#
# Created Jan 30, 2019 by Christopher L.P. Wolfe (christopher.wolfe@stonybrook.edu)

import numpy as np
import scipy as sp
import numpy.linalg
import scipy.sparse
from scipy.sparse import kron, identity, diags

from .chebfun import cheb, apply_operator, clenshaw_curtis_weight

class TwoLayerBasin(object):
    r'''A spectral model that solves the steady, linear two layer fluid equations

    Solves the equations

    .. math::
        -f v &= \eta_x - K f \eta_y + \text{Ek}\, \nabla^2 u \\
         f u &= \eta_y + K f \eta_x + \text{Ek}\, \nabla^2 v \\
         u_x + v_y &= r(\eta - \theta)

    with no-slip boundary conditions using Chebshev colocation.

    Parameters
    ----------
    Lx : float, optional
        Nondimensional zonal extend of domain. Default: 1
    Ly : float
        Nondimensional meridional extend of domain
    Nx : int
        Number of colocation points in zonal direction is Nx+1 (including boundary points)
    Ny : int, optional
        Number of colocation points in meridional direction is Ny+1 (including boundary points).
        Default: Ly*Nx/Lx
    b : float
        beta parameter
    Ek : float
        Ekman number
    r : float or 2D ndarry, optional
        Relaxation timescale; spatially variable if a 2D array. If not specified at initialization,
        can be specified by calling `init_forcing`.

    Attributes
    ----------
    x : Nx + 1 ndarray
        Zonal colocation grid.
    y : Ny + 1 ndarray
        Meridional colocation grid.
    X, Y : (Ny + 1) by (Nx + 1) ndarrays
        Kronecker products of the colocation grids.
    u : (Ny + 1) by (Nx + 1) ndarray
        Residual zonal velocity at colocation points. Available after call to :meth:`~two_layer_basin.TwoLayerBasin.solve`.
    v : (Ny + 1) by (Nx + 1) ndarray
        Residual meridional velocity at colocation points. Available after call to :meth:`~two_layer_basin.TwoLayerBasin.solve`.
    η : (Ny + 1) by (Nx + 1) ndarray
        Interface height at colocation points. Available after call to :meth:`~two_layer_basin.TwoLayerBasin.solve`.

    '''
    def __init__(self, Ly, Nx, b, Ek, r=None, Lx=1, Ny=None,
                    corner_bc_hack=False):
        self.Lx = Lx
        self.Ly = Ly
        self.b = b
        self.Ek = Ek
        self.corner_bc_hack = corner_bc_hack

        self.operators_initialized = False
        self.boundary_conditions_initialized = False

        self.init_grid(Nx, Ny)

        if r is not None:
            if isinstance(r, numpy.ndarray):
                self.r = diags(r.flatten())
            else:
                self.r = diags(np.repeat(r, self.N))
        else:
            self.r = None


    def init_grid(self, Nx, Ny):
        r'''Initialize the grid.

        Intializes grid and first and second derivative matrices with calls to :func:`~chebfun.cheb`.
        Sets up logical arrays which are `True` on each of the boundaries and `False`
        everywhere else.
        Defines the array of Coriolis parameter values.

        Parameters
        ----------
        Nx : int
            Number of colocation points in zonal direction is Nx+1 (including boundary points)
        Ny : int
            Number of colocation points in meridional direction is Ny+1 (including boundary points).

        '''
        self.Nx = Nx
        if Ny is None:
            self.Ny = int(self.Ly*Nx/self.Lx)
        else:
            self.Ny = Ny

        self.shape = (self.Ny+1, self.Nx+1)
        self.N = np.prod(self.shape)

        self.x, self.dx, self.dx2 = cheb(self.Nx, x1=0, x2=self.Lx, calc_D2=True)
        self.y, self.dy, self.dy2 = cheb(self.Ny, x1=-self.Ly/2, x2=self.Ly/2, calc_D2=True)

        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.X_flat = self.X.flatten()
        self.Y_flat = self.Y.flatten()

        self.boundary_mask_west  = np.zeros_like(self.X, dtype=np.bool)
        self.boundary_mask_east  = np.zeros_like(self.X, dtype=np.bool)
        self.boundary_mask_west[: ,0 ] = True
        self.boundary_mask_east[: ,-1] = True

        self.boundary_mask_south = np.zeros_like(self.X, dtype=np.bool)
        self.boundary_mask_north = np.zeros_like(self.X, dtype=np.bool)
        self.boundary_mask_south[0, : ] = True
        self.boundary_mask_north[-1,: ] = True

        self.boundary_mask = (self.boundary_mask_east | self.boundary_mask_west |
                             self.boundary_mask_north | self.boundary_mask_south)

        self.f = diags(1 + self.b*self.y)


    def init_forcing(self, K, θ, r=None):
        r'''Intitialize the forcing.

        Parameters
        ----------
        K : float or 2D array
            Nondimensional diffusivities; spacially variable if a 2D array.
        θ : 2D array
            Relaxation target for η
        r : float or 2D arry, optional
            Relaxation timescale; spatially variable if a 2D array. Can be specified at initialization.
        '''
        if isinstance(K, numpy.ndarray):
            self.K = diags(K.flatten())
        else:
            self.K = diags(np.repeat(K, self.N))

        self.θ = θ

        if r is not None:
            if isinstance(r, numpy.ndarray):
                self.r = diags(r.flatten())
            else:
                self.r = diags(np.repeat(r, self.N))
        elif self.r is None:
                raise RuntimeError('Relaxation rate must be specified at initialization or in init_forcing')


    def init_operators(self, reset=False):
        r'''Initialize derivative operators.

        Parameters
        ----------
        reset : bool
            Recalculate operators if `True`

        Notes
        -----
        Initializing the operators is expensive. If called more than once with `reset = False`
        the operators are only initialized on the first call.
        '''
        if self.operators_initialized and not reset:
            return

        self.Ix = identity(self.Nx+1)
        self.Iy = identity(self.Ny+1)

        self.Dx = kron(self.Iy, self.dx, format='csr')
        self.Dy = kron(self.dy, self.Ix, format='csr')
        self.D2x = kron(self.Iy, self.dx2, format='csr')
        self.D2y = kron(self.dy2, self.Ix, format='csr')
        self.Δ = self.D2x + self.D2y

        self.I = identity(self.N)

        self.op_viscU = (self.Ek*self.Δ).tolil()
        self.op_viscV = self.op_viscU.copy()

        self.op_pgfV = (-self.Dy - self.K@kron(self.f,         self.dx)).tolil()
        self.op_pgfU = ( self.Dx - self.K@kron(self.f@self.dy, self.Ix)).tolil()

        self.op_coriU = kron(self.f, self.Ix, format='lil')
        self.op_coriV = self.op_coriU.copy()

        self.op_Dxu = self.Dx.tolil(copy=True)
        self.op_Dyv = self.Dy.tolil(copy=True)
        self.op_rIη = (-self.r).tolil(copy=True)

        self.operators_initialized = True

    def init_boundary_conditions(self, reset=False):
        r'''Initialize boundary conditions.

        Parameters
        ----------
        reset : bool
            Reinitialize boundary conditions if `True`

        Notes
        -----
        Calling more than once with `reset = False` has no additional effect.
        '''
        # setup boundary conditions

        if self.boundary_conditions_initialized and not reset:
            return

        idx_bc = self.boundary_mask.flatten()

        # note that op_CoriU = fv and op_CoriV = fu
        self.op_viscU[idx_bc,:] = 0
        self.op_viscV[idx_bc,:] = 0
        self.op_coriU[idx_bc,:] = 0
        self.op_coriU[idx_bc,idx_bc] = 1
        self.op_coriV[idx_bc,:] = 0
        self.op_coriV[idx_bc,idx_bc] = 1
        self.op_pgfU[idx_bc,:] = 0
        self.op_pgfV[idx_bc,:] = 0


        θrhs = self.θ.copy().flatten()

        if self.corner_bc_hack:
            # Corners are pathological: replace η equation by the requirement that the corner value be the average of
            # the nearby boundary points
            indices = np.reshape(np.arange(self.N), (self.Ny+1, self.Nx+1))

            # northern corners
            idx_corners = ((self.boundary_mask_west | self.boundary_mask_east)
                         & (self.boundary_mask_south | self.boundary_mask_north)).flatten()

            self.op_Dxu[idx_corners,:] = 0
            self.op_Dyv[idx_corners,:] = 0

            # NW corner
            self.op_rIη[indices[-1,0],indices[-1,0]] = -2
            self.op_rIη[indices[-1,0],indices[-2,0]] = 1
            self.op_rIη[indices[-1,0],indices[-1,1]] = 1

            # NE corner
            self.op_rIη[indices[-1,-1],indices[-1,-1]] = -2
            self.op_rIη[indices[-1,-1],indices[-2,-1]] = 1
            self.op_rIη[indices[-1,-1],indices[-1,-2]] = 1

            # SW corner
            self.op_rIη[indices[0,0],indices[0,0]] = -2
            self.op_rIη[indices[0,0],indices[1,0]] = 1
            self.op_rIη[indices[0,0],indices[0,1]] = 1

            # SE corner
            self.op_rIη[indices[0,-1],indices[0,-1]] = -2
            self.op_rIη[indices[0,-1],indices[1,-1]] = 1
            self.op_rIη[indices[0,-1],indices[0,-2]] = 1

            θrhs[idx_corners] = 0

        self.rhs = np.zeros(3*self.N)
        self.rhs[2*self.N:] = -self.r*θrhs

        self.boundary_conditions_initialized = True

    def solve(self):
        r'''Solve the linear system

        Sets up operators and boundary conditions then solves the resulting sparse-ish linear
        system using the umfpack implementation of :func:`~scipy.sparse.linalg.spsolve`.
        '''
        self.init_operators()
        self.init_boundary_conditions()

        L = sp.sparse.bmat([
            [self.op_coriV, -self.op_viscV,  self.op_pgfV],
            [self.op_viscU,  self.op_coriU,  self.op_pgfU],
            [self.op_Dxu,    self.op_Dyv,    self.op_rIη ]
        ], format='csr')

        # hack to get around 32 bit memory limit in umfpack
        L.indices = L.indices.astype(np.int64)
        L.indptr  = L.indptr.astype(np.int64)
        sol = sp.sparse.linalg.spsolve(L, self.rhs.flatten(), use_umfpack=True)
        self.u = np.reshape(sol[:self.N],         (self.Ny+1, self.Nx+1))
        self.v = np.reshape(sol[self.N:2*self.N], (self.Ny+1, self.Nx+1))
        self.η = np.reshape(sol[2*self.N:],       (self.Ny+1, self.Nx+1))


    def calc_derived_quantities(self):
        r'''Calculate quantities derived from u, v, and η

        These are stored as attributes to the model and include

        Attributes
        ----------
        w, wbar, wstar: (Ny + 1) by (Nx + 1) ndarray
           Residual, mean, and eddy vertical velocity at colocation points.
        ubar, ustar : (Ny + 1) by (Nx + 1) ndarray
            Mean and eddy zonal velocities at colocation points.
        vbar, vstar : (Ny + 1) by (Nx + 1) ndarray
            Mean and eddy meridional velocities at colocation points.
            Eddy vertical velocity at colocation points.
        ux, ubarx : (Ny + 1) by (Nx + 1) ndarray
            Derivative of residual and mean zonal velocities at colocation points.
        vy, vbary : (Ny + 1) by (Nx + 1) ndarray
            Derivative of residual and mean meridional velocities at colocation points.
        viscu : (Ny + 1) by (Nx + 1) ndarray
            Viscous term from zonal momentum equation: :math:`\text{Ek} \nabla^2 u`
        viscv : (Ny + 1) by (Nx + 1) ndarray
            Viscous term from meridional momentum equation: :math:`\text{Ek} \nabla^2 v`
        fv : (Ny + 1) by (Nx + 1) ndarray
            Coriolis term from zonal momentum equation: :math:`fv`
        fu : (Ny + 1) by (Nx + 1) ndarray
            Coriolis term from meridional momentum equation: :math:`fu`
        ηx : (Ny + 1) by (Nx + 1) ndarray
            Pressure gradient term from zonal momentum equation: :math:`\eta_x`
        ηy : (Ny + 1) by (Nx + 1) ndarray
            Pressure gradient term from meridional momentum equation: :math:`\eta_y`
        fKηy : (Ny + 1) by (Nx + 1) ndarray
            Form drag term from zonal momentum equation: :math:`f K \eta_y`
        fKηx : (Ny + 1) by (Nx + 1) ndarray
            Form drag term from meridional momentum equation: :math:`f K \eta_x`
        w_yint, wbar_yint : (Nx + 1) array
            Meridionally integrated residual and mean vertical velocities.
        rzoc, mzoc : (Nx + 1) array
            Residual and mean zonal overturning circulations.
        w_xint, wbar_xint : (Ny + 1) array
            Zonally integrated residual and mean vertical velocities.
        rmoc, mmoc : (Nx + 1) array
            Residual and mean meridional overturning circulations.

        '''
       # differential elements in x and y
        dx = clenshaw_curtis_weight(self.Nx, x1=self.x[0], x2=self.x[-1])
        dy = clenshaw_curtis_weight(self.Ny, x1=self.y[0], x2=self.y[-1])[:,np.newaxis]

        self.w = apply_operator(self.r, self.η - self.θ)

        self.ustar = apply_operator(self.K@self.Dx, self.η)
        self.vstar = apply_operator(self.K@self.Dy, self.η)

        self.ustar[self.boundary_mask] = 0
        self.vstar[self.boundary_mask] = 0


        self.ustarx = apply_operator(self.Dx, self.ustar)
        self.vstary = apply_operator(self.Dy, self.vstar)

        self.wstar = self.ustarx + self.vstary

        self.ubar = self.u - self.ustar
        self.vbar = self.v - self.vstar

        self.ubarx = apply_operator(self.Dx, self.ubar)
        self.vbary = apply_operator(self.Dy, self.vbar)
        self.wbar = self.ubarx + self.vbary



        self.viscu = apply_operator(self.Ek*self.Δ, self.u)
        self.viscv = apply_operator(self.Ek*self.Δ, self.v)
        self.fv = apply_operator(kron(self.f, self.Ix), self.v)
        self.fu = apply_operator(kron(self.f, self.Ix), self.u)
        self.ηx = apply_operator(self.Dx, self.η)
        self.ηy = apply_operator(self.Dy, self.η)
        self.fKηx = apply_operator(self.K@kron(self.f, self.dx), self.η)
        self.fKηy = apply_operator(self.K@kron(self.f*self.dy, self.Ix), self.η)
        self.fKηx[self.boundary_mask] = 0
        self.fKηy[self.boundary_mask] = 0

        self.ux = apply_operator(self.Dx, self.u)
        self.vy = apply_operator(self.Dy, self.v)



        # ZOC
        self.w_yint    = np.sum(dy*self.w, axis=0)
        self.wbar_yint = np.sum(dy*self.wbar, axis=0)
        self.rzoc      = np.sum(dy*self.u, axis=0)
        self.mzoc      = np.sum(dy*self.ubar, axis=0)


        # MOC
        self.w_xint    = np.sum(dx*self.w, axis=1)
        self.wbar_xint = np.sum(dx*self.wbar, axis=1)
        self.rmoc      = np.sum(dx*self.v, axis=1)
        self.mmoc      = np.sum(dx*self.vbar, axis=1)


        # PV
        self.ζ = apply_operator(self.Dx, self.v) - apply_operator(self.Dy, self.u)
        self.pv_form_drag = apply_operator(self.Dx, self.fKηx) + apply_operator(self.Dy, self.fKηy)
        self.pv_visc = apply_operator(self.Ek*self.Δ, self.ζ)



    def interp_field1d(self, x, xi, fld):
        r''' Interpolate a field in 1D using piecewise cubic Hermite interpolating polynomials

        Parameters
        ----------
        x : 1D array
            Origin grid
        xi : 1D array
            Destination grid
        fld : 1D array
            Field to be interpolated

        Returns
        -------
         : 1D array
            Interpolated field
        '''

        f = sp.interpolate.PchipInterpolator(x, fld)

        return f(xi)


    def interp_field2d(self, Xi, Yi, fld):
        r''' Interpolate a field in 2D using splines.

        Parameters
        ----------
        Xi : 1D array
            Destination x grid
        Yi : 1D array
            Destination y grid
        fld : 1D array
            Field to be interpolated

        Returns
        -------
         : 2D array
            Interpolated field

        Notes
        -----
        Assumes the origin grid is the internal colocation grid.
        '''
        from scipy.interpolate import interpn
        return interpn((self.y, self.x), fld,  (Yi, Xi), method='splinef2d')


    def interp_grid_1d(self, NXi, NYi=None):
        if NYi is None:
            NYi = int(self.Ly*NXi/self.Lx)

        xi = np.linspace(0, self.Lx, NXi)
        yi = np.linspace(-self.Ly/2, self.Ly/2, NYi)

        return xi, yi

    def interp_grid_2d(self, NXi, NYi=None):
        xi, yi = self.interp_grid_1d(NXi, NYi=None)

        return np.meshgrid(xi, yi)


#     def calc_interpolated_quantities(self, NXi, NYi=None):
#         if NYi is None:
#             NYi = int(self.Ly*NXi/self.Lx)
#
#         self.xi = np.linspace(0, self.Lx, NXi)
#         self.yi = np.linspace(-self.Ly/2, self.Ly/2, NYi)
#
#         Xi, Yi = np.meshgrid(self.xi, self.yi)
#
#         self.ηi     = self.interp_field2d(Xi, Yi, self.η)
#         self.ui     = self.interp_field2d(Xi, Yi, self.u)
#         self.vi     = self.interp_field2d(Xi, Yi, self.v)
#         self.wi     = self.interp_field2d(Xi, Yi, self.w)
#         self.ubari  = self.interp_field2d(Xi, Yi, self.ubar)
#         self.vbari  = self.interp_field2d(Xi, Yi, self.vbar)
#         self.wbari  = self.interp_field2d(Xi, Yi, self.wbar)
#         self.ustari = self.interp_field2d(Xi, Yi, self.ustar)
#         self.vstari = self.interp_field2d(Xi, Yi, self.vstar)
#
#         self.w_yinti = self.interp_field1d(self.x, self.xi, self.w_yint)
#         self.wbar_yinti = self.interp_field1d(self.x, self.xi, self.wbar_yint)
#         self.rzoci = self.interp_field1d(self.x, self.xi, self.rzoc)
#         self.mzoci = self.interp_field1d(self.x, self.xi, self.mzoc)
#
#         self.w_xinti = self.interp_field1d(self.y, self.yi, self.w_xint)
#         self.wbar_xinti = self.interp_field1d(self.y, self.yi, self.wbar_xint)
#         self.rmoci = self.interp_field1d(self.y, self.yi, self.rmoc)
#         self.mmoci = self.interp_field1d(self.y, self.yi, self.mmoc)
#
#         self.Xi = Xi
#         self.Yi = Yi
