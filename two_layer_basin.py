# two_layer_basin.py: Class for two layer model
#
# Created Jan 30, 2019 by Christopher L.P. Wolfe (christopher.wolfe@stonybrook.edu)

import numpy as np
import scipy as sp
import numpy.linalg

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

        self.f = np.diag(1 + self.b*self.yy_flat)


    def init_forcing(self, K, θ):
        if isinstance(K, numpy.ndarray):
            self.K = np.diag(K.flatten())
        else:
            self.K = np.diag(np.repeat(K, self.N))

        self.θ = θ


    def init_operators(self):
        dx2 = self.dx@self.dx
        dy2 = self.dy@self.dy

        Ix = np.identity(self.Nx+1)
        Iy = np.identity(self.Ny+1)

        self.Dx = np.kron(Iy, self.dx)
        self.Dy = np.kron(self.dy, Ix)
        self.D2x = np.kron(Iy, dx2)
        self.D2y = np.kron(dy2, Ix)
        self.Δ = self.D2x + self.D2y

        self.I = np.identity(self.N)

        self.visc = self.Ek*self.Δ

        self.ηy = -self.Dy - self.K@self.f@self.Dx
        self.ηx = +self.Dx - self.K@self.f@self.Dy

        self.fu = self.f.copy()
        self.fv = self.f.copy()

        self.Dxu = self.Dx.copy()
        self.Dyv = self.Dy.copy()
        self.rIη = -self.r*self.I


    def init_boundary_conditions(self):
        # setup boundary conditions

        idx_bc = self.boundary_mask.flatten() == 1

        # apply the boundary conditions to the sub-blocks
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


    def solve(self, init_operators=True, init_boundary_conditions=True):
        if init_operators:
            self.init_operators()

        if init_boundary_conditions:
            self.init_boundary_conditions()

        L = np.block([
            [self.fu,   -self.visc,   self.ηy],
            [self.visc,  self.fv,     self.ηx],
            [self.Dxu,   self.Dyv,    self.rIη]
        ])

        sol = sp.linalg.solve(L, self.rhs.flatten())
        self.u = np.reshape(sol[:self.N],         (self.Ny+1, self.Nx+1))
        self.v = np.reshape(sol[self.N:2*self.N], (self.Ny+1, self.Nx+1))
        self.η = np.reshape(sol[2*self.N:],       (self.Ny+1, self.Nx+1))

    def calc_derived_quantities(self):
        self.diags2d = {}
        self.ustar = apply_operator(self.K@self.Dx, self.η)
        self.vstar = apply_operator(self.K@self.Dy, self.η)

        self.ustar[self.boundary_mask == 1] = 0
        self.vstar[self.boundary_mask == 1] = 0
        self.ubar = self.u - self.ustar
        self.vbar = self.v - self.vstar

        self.viscu = apply_operator(self.Ek*self.Δ, self.u)
        self.viscv = apply_operator(self.Ek*self.Δ, self.v)
        self.fv = apply_operator(self.f, self.v)
        self.fu = apply_operator(self.f, self.u)
        self.ηx = apply_operator(self.Dx, self.η)
        self.ηy = apply_operator(self.Dy, self.η)
        self.fKηx = apply_operator(self.K@self.f@self.Dx, self.η)
        self.fKηy = apply_operator(self.K@self.f@self.Dy, self.η)
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
        self.pv_form_drag = apply_operator(self.Dx, self.fKηx) + apply_operator(self.Dy, self.fKηy)
        self.pv_visc = apply_operator(self.Dx, self.viscv) - apply_operator(self.Dy, self.viscu)



    def interp_field1d(self, x, xi, fld):
        from scipy.interpolate import interp1d
        f = interp1d(x, fld, kind='cubic', assume_sorted=True)

        return f(xi)


    def interp_field2d(self, xxi, yyi, fld):
        from scipy.interpolate import interpn
        return interpn((self.y, self.x), fld,  (yyi, xxi), method='splinef2d')


    def calc_interpolated_quantities(self, NXi, NYi=None):
        if NYi is None:
            NYi = int(self.Ly*NXi/self.Lx)

        self.xi = np.linspace(0, self.Lx, NXi)
        self.yi = np.linspace(-self.Ly/2, self.Ly/2, NYi)

        xxi, yyi = np.meshgrid(self.xi, self.yi)

        self.ηi     = self.interp_field2d(xxi, yyi, self.η)
        self.ui     = self.interp_field2d(xxi, yyi, self.u)
        self.vi     = self.interp_field2d(xxi, yyi, self.v)
        self.wi     = self.interp_field2d(xxi, yyi, self.w)
        self.ubari  = self.interp_field2d(xxi, yyi, self.ubar)
        self.vbari  = self.interp_field2d(xxi, yyi, self.vbar)
        self.wbari  = self.interp_field2d(xxi, yyi, self.wbar)
        self.ustari = self.interp_field2d(xxi, yyi, self.ustar)
        self.vstari = self.interp_field2d(xxi, yyi, self.vstar)

        self.w_yinti = self.interp_field1d(self.x, self.xi, self.w_yint)
        self.wbar_yinti = self.interp_field1d(self.x, self.xi, self.wbar_yint)
        self.rzoci = self.interp_field1d(self.x, self.xi, self.rzoc)
        self.mzoci = self.interp_field1d(self.x, self.xi, self.mzoc)

        self.w_xinti = self.interp_field1d(self.y, self.yi, self.w_xint)
        self.wbar_xinti = self.interp_field1d(self.y, self.yi, self.wbar_xint)
        self.rmoci = self.interp_field1d(self.y, self.yi, self.rmoc)
        self.mmoci = self.interp_field1d(self.y, self.yi, self.mmoc)

        self.xxi = xxi
        self.yyi = yyi
