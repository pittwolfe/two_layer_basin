# two_layer_basin.py: Class for two layer model
#
# Created Jan 30, 2019 by Christopher L.P. Wolfe (christopher.wolfe@stonybrook.edu)

import numpy as np
import scipy as sp
import numpy.linalg
import scipy.sparse
from scipy.sparse import kron, identity, diags

from .chebfun import *

class two_layer_model(object):
    def __init__(self, Ly, Nx, b, Ek, r, Lx=1, Ny=None, effective_bc=False,
                    corner_bc_hack=True):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.b = b
        self.Ek = Ek
        self.r = r
        self.effective_bc = effective_bc
        self.corner_bc_hack = corner_bc_hack

        if Ny is None:
            self.Ny = int(Ly*Nx/Lx)
        else:
            self.Ny = Ny

        self.boundary_conditions_initialized = False
        self.operators_initialized = False

        self.init_grid()


    def init_grid(self):
        self.dx, self.x = cheb(self.Nx, x1=0, x2=self.Lx)
        self.dy, self.y = cheb(self.Ny, x1=-self.Ly/2, x2=self.Ly/2)

        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.xx_flat = self.xx.flatten()
        self.yy_flat = self.yy.flatten()
        self.N = len(self.yy_flat)

        self.boundary_mask = np.zeros_like(self.xx)
        self.boundary_mask[0, : ] = 1
        self.boundary_mask[-1,: ] = 1
        self.boundary_mask[: ,0 ] = 1
        self.boundary_mask[: ,-1] = 1

        self.f = diags(1 + self.b*self.y)


    def init_forcing(self, K, θ):
        if isinstance(K, numpy.ndarray):
            self.K = diags(K.flatten())
        else:
            self.K = diags(np.repeat(K, self.N))

        self.θ = θ


    def init_operators(self, reset=False):
        if self.operators_initialized and not reset:
            return

        dx2 = self.dx@self.dx
        dy2 = self.dy@self.dy

        self.Ix = identity(self.Nx+1)
        self.Iy = identity(self.Ny+1)

        self.Dx = kron(self.Iy, self.dx, format='csr')
        self.Dy = kron(self.dy, self.Ix, format='csr')
        self.D2x = kron(self.Iy, dx2, format='csr')
        self.D2y = kron(dy2, self.Ix, format='csr')
        self.Δ = self.D2x + self.D2y

        self.I = identity(self.N)

        self.visc = (self.Ek*self.Δ).tolil()

        self.ηy = (-self.Dy - self.K@kron(self.f,         self.dx)).tolil()
        self.ηx = ( self.Dx - self.K@kron(self.f@self.dy, self.Ix)).tolil()

        self.fu = kron(self.f, self.Ix, format='lil')
        self.fv = self.fu.copy()

        self.Dxu = self.Dx.tolil(copy=True)
        self.Dyv = self.Dy.tolil(copy=True)
        self.rIη = (-self.r*self.I).tolil(copy=True)

        self.operators_initialized = True

    def init_boundary_conditions(self, reset=False):
        # setup boundary conditions

        if self.boundary_conditions_initialized and not reset:
            return

        idx_bc = self.boundary_mask.flatten() == 1

        self.visc[idx_bc,:] = 0
        self.fu[idx_bc,:] = 0
        self.fu[idx_bc,idx_bc] = 1
        self.fv[idx_bc,:] = 0
        self.fv[idx_bc,idx_bc] = 1
        self.ηy[idx_bc,:] = 0
        self.ηx[idx_bc,:] = 0

        θrhs = self.θ.copy().flatten()

        if self.corner_bc_hack:
            # Corners are pathological: replace η equation by the requirement that the corner value be the average of
            # the nearby boundary points
            indices = np.reshape(np.arange(self.N), (self.Ny+1, self.Nx+1))

            idx_corners = (((self.xx_flat == 0) | (self.xx_flat == 1))
                            & (np.abs(self.yy_flat) == self.Ly/2))
            self.Dxu[idx_corners,:] = 0
            self.Dyv[idx_corners,:] = 0

            # SW corner
            self.rIη[indices[0,0],indices[0,0]] = -2
            self.rIη[indices[0,0],indices[1,0]] = 1
            self.rIη[indices[0,0],indices[0,1]] = 1

            # NW corner
            self.rIη[indices[-1,0],indices[-1,0]] = -2
            self.rIη[indices[-1,0],indices[-2,0]] = 1
            self.rIη[indices[-1,0],indices[-1,1]] = 1

            # SE corner
            self.rIη[indices[0,-1],indices[0,-1]] = -2
            self.rIη[indices[0,-1],indices[1,-1]] = 1
            self.rIη[indices[0,-1],indices[0,-2]] = 1

            # NE corner
            self.rIη[indices[-1,-1],indices[-1,-1]] = -2
            self.rIη[indices[-1,-1],indices[-2,-1]] = 1
            self.rIη[indices[-1,-1],indices[-1,-2]] = 1

            θrhs[idx_corners] = 0

        self.rhs = np.zeros(3*self.N)
        self.rhs[2*self.N:] = -self.r*θrhs

        self.boundary_conditions_initialized = True

    def solve(self):
        self.init_operators()
        self.init_boundary_conditions()

        L = sp.sparse.bmat([
            [self.fu,   -self.visc,   self.ηy],
            [self.visc,  self.fv,     self.ηx],
            [self.Dxu,   self.Dyv,    self.rIη]
        ], format='csr')

        # hack to get around 32 bit memory limit in umfpack
        L.indices = L.indices.astype(np.int64)
        L.indptr  = L.indptr.astype(np.int64)
        sol = sp.sparse.linalg.spsolve(L, self.rhs.flatten(), use_umfpack=True)
        self.u = np.reshape(sol[:self.N],         (self.Ny+1, self.Nx+1))
        self.v = np.reshape(sol[self.N:2*self.N], (self.Ny+1, self.Nx+1))
        self.η = np.reshape(sol[2*self.N:],       (self.Ny+1, self.Nx+1))


    def calc_derived_quantities(self):
        self.ustar = apply_operator(self.K@self.Dx, self.η)
        self.vstar = apply_operator(self.K@self.Dy, self.η)
        self.ustar[self.boundary_mask == 1] = 0
        self.vstar[self.boundary_mask == 1] = 0

        self.ubar = self.u - self.ustar
        self.vbar = self.v - self.vstar


        self.viscu = apply_operator(self.Ek*self.Δ, self.u)
        self.viscv = apply_operator(self.Ek*self.Δ, self.v)
        self.fv = apply_operator(kron(self.f, self.Ix), self.v)
        self.fu = apply_operator(kron(self.f, self.Ix), self.u)
        self.ηx = apply_operator(self.Dx, self.η)
        self.ηy = apply_operator(self.Dy, self.η)
        self.fKηx = apply_operator(self.K@kron(self.f, self.dx), self.η)
        self.fKηy = apply_operator(self.K@kron(self.f*self.dy, self.Ix), self.η)
        self.fKηx[self.boundary_mask == 1] = 0
        self.fKηy[self.boundary_mask == 1] = 0

        self.ux = apply_operator(self.Dx, self.u)
        self.vy = apply_operator(self.Dy, self.v)

        self.ubarx = apply_operator(self.Dx, self.ubar)
        self.vbary = apply_operator(self.Dy, self.vbar)

        self.w = self.r*(self.η - self.θ)
        self.wbar = self.ubarx + self.vbary

        # ZOC
        self.w_yint    = integrate_in_y(np.reshape(self.w, (self.Ny+1, self.Nx+1)), self.dy)[-1,:]
        self.wbar_yint = integrate_in_y(np.reshape(self.wbar, (self.Ny+1, self.Nx+1)), self.dy)[-1,:]
        self.rzoc      = integrate_in_y(np.reshape(self.u, (self.Ny+1, self.Nx+1)), self.dy)[-1,:]
        self.mzoc      = integrate_in_y(np.reshape(self.ubar, (self.Ny+1, self.Nx+1)), self.dy)[-1,:]
        self.mzoc[ 0] = 0
        self.mzoc[-1] = 0

        # MOC
        self.w_xint    = integrate_in_x(np.reshape(self.w, (self.Ny+1, self.Nx+1)), self.dx)[:,-1]
        self.wbar_xint = integrate_in_x(np.reshape(self.wbar, (self.Ny+1, self.Nx+1)), self.dx)[:,-1]
        self.rmoc = integrate_in_x(np.reshape(self.v, (self.Ny+1, self.Nx+1)), self.dx)[:,-1]
        self.mmoc = integrate_in_x(np.reshape(self.vbar, (self.Ny+1, self.Nx+1)), self.dx)[:,-1]
        self.mmoc[0] = 0
        self.mmoc[-1] = 0

        # PV
        self.ζ = apply_operator(self.Dx, self.v) - apply_operator(self.Dy, self.u)
        self.pv_form_drag = apply_operator(self.Dx, self.fKηx) + apply_operator(self.Dy, self.fKηy)
        self.pv_visc = apply_operator(self.Ek*self.Δ, self.ζ)



    def interp_field1d(self, x, xi, fld):
        f = sp.interpolate.PchipInterpolator(x, fld)

        return f(xi)


    def interp_field2d(self, xxi, yyi, fld):
        from scipy.interpolate import interpn
        return interpn((self.y, self.x), fld,  (yyi, xxi), method='splinef2d')


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
#         xxi, yyi = np.meshgrid(self.xi, self.yi)
#
#         self.ηi     = self.interp_field2d(xxi, yyi, self.η)
#         self.ui     = self.interp_field2d(xxi, yyi, self.u)
#         self.vi     = self.interp_field2d(xxi, yyi, self.v)
#         self.wi     = self.interp_field2d(xxi, yyi, self.w)
#         self.ubari  = self.interp_field2d(xxi, yyi, self.ubar)
#         self.vbari  = self.interp_field2d(xxi, yyi, self.vbar)
#         self.wbari  = self.interp_field2d(xxi, yyi, self.wbar)
#         self.ustari = self.interp_field2d(xxi, yyi, self.ustar)
#         self.vstari = self.interp_field2d(xxi, yyi, self.vstar)
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
#         self.xxi = xxi
#         self.yyi = yyi
