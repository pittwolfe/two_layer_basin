# one_layer_jet.py: Classes implementing the 1.5-layer Rossby-Zhang jet
#
# Created May 19, 2022 by Christopher L.P. Wolfe (christopher.wolfe@stonybrook.edu)

import numpy as np
import scipy as sp

from . import chebfun as cheb

# Class to represent the background flow
class OneLayerJet_BackgroundFlow(object):
    def __init__(self, δ, Ro, F, b):
        self.δ = δ     # asymmetry parameter
        self.Ro = Ro   # Rossby number
        self.F = F     # baroclinic inverse Burger number
        self.b = b     # beta parameter

        self.Δζ = 2/(1 - self.δ**2) # The jump in ζ at y = 0.

    def f(self, y):
        if self.b == 0:
            return np.ones_like(y)
        else:
            return 1 + self.b*y

    def u(self, y):
        δ = self.δ

        if δ < 1:
            u = np.exp(-np.abs(y)/(1-np.sign(y)*δ))
            u[y == 0] = 1
        else:
            u = np.zeros_like(y)
            u[y<=0] = np.exp(-np.abs(y[y<=0])/2)

        return u

    def dudy_south(self, y):
        ret = np.exp(y/(1+self.δ))/(1+self.δ)
        ret[y > 0] = np.nan
        return ret

    def dudy_north(self, y):
        ret = -np.exp(-y/(1-self.δ))/(1-self.δ)
        ret[y < 0] = np.nan
        return ret

    def u_inv_south(self, Y):
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y, dtype=float)

        Y = np.abs(Y)
        δ = self.δ
        y = np.zeros_like(Y)

        y[Y >  0] = (1+δ)*np.log(Y[Y > 0])
        y[Y == 0] = -np.inf
        y[Y <  0] = np.nan

        return y

    def u_inv_north(self, Y):
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y, dtype=float)

        Y = np.abs(Y)
        δ = self.δ
        y = np.zeros_like(Y)

        y[Y >  0] = -(1-δ)*np.log(Y[Y > 0])
        y[Y == 0] = np.inf
        y[Y <  0] = np.nan

        return y

    def η(self, y):
        δ = self.δ
        Ro = self.Ro
        b = self.b

        f = self.f(y)
        S = np.sign(y)
        u = self.u(y)

        η = -(u*(δ - S)*(f - b*Ro*(δ - S)) + S + 2*(1 - S)*b*Ro*δ)
        η[y==0] = -(δ - Ro*b*(1 - δ)**2)

        return η

    def h(self, y):
        return 1 + self.Ro*self.F*self.η(y)

    def ζ_jump(self, y):
        '''
        This is a discontinuous version of ζ with a jump at y = 0.
        '''
        δ = self.δ
        S = np.sign(y)

        if (δ < 1):
            ζ = self.u(y)/(S - δ)
            ζ[y == 0] = np.nan
        else:
            ζ = np.zeros_like(y)
            ζ[y < 0] = self.u(y[y<0])/2
            ζ[y==0] = np.nan

        return ζ

    def ζ(self, y):
        '''
        This is a continuous version of ζ with the value at y = 0 the average of the two limits.
        '''
        δ = self.δ

        ζ = np.zeros_like(y)
        if (δ < 1):
            ζ[y < 0] = -self.u(y[y < 0])/(1+δ)
            ζ[y > 0] =  self.u(y[y > 0])/(1-δ)
            ζ[y == 0] = δ/(1-δ**2)
        else:
            ζ[y < 0] = -self.u(y[y<0])/2
            ζ[y==0] = -1/4

        return ζ

    def q(self, y):
        Ro = self.Ro
        f = self.f(y)
        ζ = self.ζ(y)
        h = self.h(y)

        return (Ro*ζ + f)/h

##########################################################################################
# The solution in the long wave limit
class OneLayerJet_Longwave(object):
    def __init__(self, δ):
        self.δ = δ     # asymmetry parameter
        self.Ro = 2*δ   # Rossby number
        self.F = 1/(1+δ**2)   # baroclinic inverse Burger number
        self.b = 0     # beta parameter

        self.bg = OneLayerJet_BackgroundFlow(self.δ, self.Ro, self.F, self.b)

    def c(self, k):
        return k**2*self.δ*(3 + self.δ**2)/3

    def η0(self, y):
        δ = self.δ
        η = np.zeros_like(y)

        yp = y[y >= 0]
        yn = y[y <  0]

        η[y >= 0] = np.exp(-yp/(1-δ))
        η[y <  0] = np.exp( yn/(1+δ))

        return η

    def η2(self, y):
        δ = self.δ
        η = np.zeros_like(y)

        yp = y[y >= 0]
        yn = y[y <  0]

        η[y >= 0] = np.exp(-yp/(1-δ))*(1-δ)*(-3 - 3*yp + 5*δ - 8*δ*np.exp(-yp/(1-δ)))/6
        η[y <  0] = np.exp( yn/(1+δ))*(1+δ)*(-3 + 3*yn - 5*δ + 8*δ*np.exp( yn/(1+δ)))/6

        return η


    def η(self, k, y):
        δ = self.δ

        return self.η0(y) + k**2*self.η2(y)

    def u0(self, y):
        δ = self.δ
        u = np.zeros_like(y)

        yp = y[y > 0]
        yn = y[y < 0]

        u[y > 0] =  np.exp(-yp/(1-δ))/(1-δ)
        u[y < 0] = -np.exp( yn/(1+δ))/(1+δ)
        u[y == 0] = np.nan

        return u

    def u2(self, y):
        δ = self.δ
        u = np.zeros_like(y)

        yp = y[y >= 0]
        yn = y[y <  0]

        u[y >= 0] = -(3*yp - 2*δ + 4*δ*np.exp(-yp/(1-δ)))*np.exp(-yp/(1-δ))/6
        u[y <  0] = -(3*yn - 2*δ + 4*δ*np.exp( yn/(1+δ)))*np.exp( yn/(1+δ))/6

        return u

    def u(self, k, y):
        return self.u0(y) + k**2*self.u2(y)

    def ζ0(self, y):
        δ = self.δ
        ζ = np.zeros_like(y)

        yp = y[y > 0]
        yn = y[y < 0]

        ζ[y > 0] = np.exp(-yp/(1-δ))/(1-δ)**2
        ζ[y < 0] = np.exp( yn/(1+δ))/(1+δ)**2
        ζ[y == 0] = np.nan

        return ζ

    def ψ0(self, y):
        δ = self.δ
        ψ = np.zeros_like(y)

        yp = y[y >= 0]
        yn = y[y < 0]

        ψ[y >= 0] = np.exp(-yp/(1-δ))
        ψ[y <  0] = np.exp( yn/(1+δ))

        return ψ

    def ψ2(self, y):
        δ = self.δ
        c2 = (3+δ**2)/6

        ψ = (self.η2(y) + (2*δ*self.bg.u(y)*self.u2(y) - c2*self.u0(y)))/(self.bg.h(y)*self.bg.q(y))
        ψ[y==0] = -c2

        return ψ

    def ψ(self, k, y):
        return self.ψ0(y) + k**2*self.ψ2(y)

    def v0(self, y):
        return 1j*self.ψ0(y)

    def v2(self, y):
        return 1j*self.ψ2(y)

    def v(self, k, y):
        return k*(self.v0(y) + k**2*self.v2(y))

##########################################################################################
# This one uses a logarithmic map. Boyd says this is a bad idea.
class OneLayerJet_cheb(object):
    '''
    A class implementing a one-layer Rossby-Zhang jet with constant PV and a solution in the long wave limit.
    '''

    def __init__(self, δ, Ro, F, b, N):
        self.δ = δ     # asymmetry parameter
        self.Ro = Ro   # Rossby number
        self.F = F     # baroclinic inverse Burger number
        self.b = b     # beta parameter
        self.N = N

        self.bg = OneLayerJet_BackgroundFlow(self.δ, self.Ro, self.F, self.b)

        self.grid_initialized = False
        self.background_flow_initialized = False

    def init_grid(self):
        N = self.N # number of grid points in each subgrid
        if (N == 0):
            raise RuntimeError('Number of grid points must be greater than zero!')

        # The suffixes S and N refer to the southern and northern subdomains, respectively.
        # We leave off the points "at infinity".
        yS = self.bg.u_inv_south(cheb.grid(N, x1=0, x2=1)[1:])
        yN = self.bg.u_inv_north(cheb.grid(N, x1=0, x2=1)[:0:-1])
        yS[-1] = 0
        yN[0] = 0

        # This grid has no extra points
        y = np.hstack([yS, yN[1:]])

        # This grid duplicates y = 0
        yd = np.hstack([yS, yN])

        dyS = np.diag(self.bg.dudy_south(yS)) @ cheb.derivative_matrix(N, x1=0, x2=1)[1:,1:]
        dyN = np.diag(self.bg.dudy_north(yN)) @ cheb.derivative_matrix(N, x1=0, x2=1)[:0:-1,:0:-1]

        # This operator is for functions with continuous derivatives at y = 0
        # There are two different ways to construct the derivative matrix for functions with continuous derivatives
        # Picking one or the other leads to weird asymmetries in the derivative matrix, so we average the two together
        dyC1 = np.block([[dyS, np.zeros((N, N-1))],
                        [np.zeros((N-1, N-1)), dyN[1:,:]]])
        dyC2 = np.block([[dyS[:-1,:], np.zeros((N-1, N-1))],
                        [np.zeros((N, N-1)), dyN]])
        dyC = (dyC1+dyC2)/2


        # This operator is for functions with discontinuous derivatives at y = 0
        dyD = np.block([[dyS, np.zeros((N, N-1))],
                        [np.zeros((N, N-1)), dyN]])


        # projects quantities from the unique grid to the duplicate grid
        PtoD = np.block([
            [np.identity(N), np.zeros((N, N-1))],
            [np.zeros((N, N-1)), np.identity(N)]
        ])

        # projects quantities from the duplicate grid to the unique grid
        PfromD = np.block([
            [np.identity(N), np.zeros((N, N))],
            [np.zeros((N-1, N+1)), np.identity(N-1)]
        ])
        PfromD[N-1, N-1] = 0.5
        PfromD[N-1, N] = 0.5

        # Various zero and identity matrices
        Z      = np.zeros((2*N-1, 2*N-1))
        ZtoD   = np.zeros((2*N,   2*N-1))
        ZfromD = np.zeros((2*N-1, 2*N))

        I  = np.identity(2*N-1)
        ID = np.identity(2*N)

        self.idx_hu = np.arange(2*N)
        self.idx_ψ = self.idx_hu[-1] + 1 + np.arange(2*N-1)
        self.idx_η = self.idx_ψ[-1] + 1 + np.arange(2*N-1)

        self.y = y
        self.yd = yd
        self.yS = yS
        self.yN = yN

        self.dyC = dyC
        self.dyD = dyD

        self.PtoD = PtoD
        self.PfromD = PfromD

        self.Z = Z
        self.ZtoD = ZtoD
        self.ZfromD = ZfromD

        self.I = I
        self.ID = ID

        self.grid_initialized = True

    def init_background_flow(self):
        if not self.grid_initialized:
            self.init_grid()

        # background flow on the unique grid
        self.f_u    = self.bg.f(self.y)
        self.ubar_u = self.bg.u(self.y)
        self.hbar_u = self.bg.h(self.y)

        # background flow on the duplicate grid
        self.f_d    = self.bg.f(self.yd)
        self.ubar_d = self.bg.u(self.yd)
        self.hbar_d = self.bg.h(self.yd)

        self.ubarp_d = np.hstack((self.bg.dudy_south(self.yS), self.bg.dudy_north(self.yN)))

        self.background_flow_initialized = True
    def modes(self, k):
        if not self.background_flow_initialized:
            self.init_background_flow()

        self.k = k

        self.lhs_U = np.block([
            [np.diag(self.Ro*self.ubar_d),
             np.diag(self.Ro*self.ubarp_d - self.f_d) @ self.PtoD,
             np.diag(self.hbar_d) @ self.PtoD ],
        ])
        self.rhs_U = np.block([
            [self.ID,     self.ZtoD,    self.ZtoD   ],
        ])


        # Everything is multiplied through by -1 to make the RHS positive semidefinite
        self.lhs_V = np.block([
            [np.diag(-self.f_u) @ self.PfromD,
             np.diag(k**2*self.Ro*self.ubar_u),
             np.diag(-self.hbar_u) @ self.dyC    ],
        ])
        self.rhs_V = np.block([
            [self.ZfromD, k**2*self.I, self.Z],
        ])


        self.lhs_H = np.block([
            [self.PfromD,
             self.PfromD @ self.dyD,
             np.diag(self.Ro*self.F*self.ubar_u)],
        ])
        self.rhs_H = np.block([
            [self.ZfromD, self.Z,       self.F*self.I]
        ])


        self.lhs = np.block([
            [self.lhs_U],
            [self.lhs_V],
            [self.lhs_H]
        ])

        self.rhs = np.block([
            [self.rhs_U],
            [self.rhs_V],
            [self.rhs_H]
        ])


        λ, X = sp.linalg.eig(self.lhs, self.rhs)
        # idx = np.isfinite(λ)
        # λ = λ[idx]
        # X = X[:,idx]
        c = λ/jet.Ro

        # We'll magntiude
        idx = np.argsort(np.abs(c))
        self.c = c[idx]
        self.X = X[:,idx]

        self.u = self.X[self.idx_hu,:]/self.hbar_d[:,np.newaxis]
        self.v = 1j*k*self.X[self.idx_ψ,:]/self.hbar_u[:,np.newaxis]
        self.η = self.X[self.idx_η]

        return self.c

##########################################################################################
# Based on my reading of Boyd, I had thought that the appearance of large complex eigenvalues
# for $k > 0$ was from imposing too many boundary conditions at infinity. The original version
# of this class discards the point at infinity and imposes zero values on everything. This
# version keeps the points at infinity and only imposes a boundary condition on the
# meridional velocity. However, the resulting eigenspectrum is exactly the same except for
# two infinite eigenvalues associated with the boundary condition.
class OneLayerJet_cheb_inf(object):
    '''
    A class implementing a one-layer Rossby-Zhang jet with constant PV and a solution in the long wave limit.
    '''

    def __init__(self, δ, Ro, F, b, N, radiation_bc=False):
        self.δ = δ     # asymmetry parameter
        self.Ro = Ro   # Rossby number
        self.F = F     # baroclinic inverse Burger number
        self.b = b     # beta parameter
        self.N = N
        self.radiation_bc = radiation_bc # Whether to use radiation BCs

        self.bg = OneLayerJet_BackgroundFlow(self.δ, self.Ro, self.F, self.b)

        self.grid_initialized = False
        self.background_flow_initialized = False

    def init_grid(self):
        N = self.N # number of grid points in each subgrid
        if (N == 0):
            raise RuntimeError('Number of grid points must be greater than zero!')

        # The suffixes S and N refer to the southern and northern subdomains, respectively.
        # The capital 'Y' refers to the actual Chebyshev grid.
        # Note that the
        YS = cheb.grid(N, x1=-1, x2=0)
        YN = cheb.grid(N, x1= 0, x2=1)
        # The lower case 'y' refers to the remapped (physical) grid.
        # Note that the first entry of yS is np.inf and the last entry of yN is -np.inf
        yS = self.bg.u_inv_south(1+YS)
        yN = self.bg.u_inv_north(1-YN)
        yS[-1] = 0
        yN[0] = 0

        # This grid has no extra points (unique grid)
        yu = np.hstack([yS, yN[1:]])
        self.Yu = np.hstack([YS, YN[1:]])

        # This grid duplicates y = 0
        yd = np.hstack([yS, yN])
        self.Yd = np.hstack([YS, YN])

        # Note negative sign
        dyS =  np.diag(self.bg.dudy_south(yS)) @ cheb.derivative_matrix(N, x1=-1, x2=0)
        dyN = -np.diag(self.bg.dudy_north(yN)) @ cheb.derivative_matrix(N, x1= 0, x2=1)

        # This operator is for functions with continuous derivatives at y = 0
        # There are two different ways to construct the derivative matrix for functions with continuous derivatives
        # Picking one or the other leads to weird asymmetries in the derivative matrix, so we average the two together
        dyC1 = np.block([[dyS, np.zeros((N+1, N))],
                        [np.zeros((N, N)), dyN[1:,:]]])
        dyC2 = np.block([[dyS[:-1,:], np.zeros((N, N))],
                        [np.zeros((N+1, N)), dyN]])
        dyC = (dyC1+dyC2)/2


        # This operator is for functions with discontinuous derivatives at y = 0
        dyD = np.block([[dyS, np.zeros((N+1, N))],
                        [np.zeros((N+1, N)), dyN]])


        # projects quantities from the unique grid to the duplicate grid
        PtoD = np.block([
            [np.identity(N+1), np.zeros((N+1, N))],
            [np.zeros((N+1, N)), np.identity(N+1)]
        ])

        # projects quantities from the duplicate grid to the unique grid
        PfromD = np.block([
            [np.identity(N+1), np.zeros((N+1, N+1))],
            [np.zeros((N, N+2)), np.identity(N)]
        ])
        PfromD[N, N] = 0.5
        PfromD[N, N+1] = 0.5

        # Various zero and identity matrices
        Z      = np.zeros((2*N+1, 2*N+1))
        ZtoD   = np.zeros((2*N+2, 2*N+1))
        ZfromD = np.zeros((2*N+1, 2*N+2))

        I  = np.identity(2*N+1)
        ID = np.identity(2*N+2)

        # u is on the full duplicate grid: 2*N+2 points
        self.idx_hu = np.arange(2*N+2)
        # ψ is on the unique grid: 2*N+1 points
        self.idx_ψ = self.idx_hu[-1] + 1 + np.arange(2*N+1)
        self.idx_η = self.idx_ψ[-1] + 1 + np.arange(2*N+1)

        self.yu = yu
        self.yd = yd
        self.yS = yS
        self.yN = yN

        self.dyC = dyC
        self.dyD = dyD

        self.PtoD = PtoD
        self.PfromD = PfromD

        self.Z = Z
        self.ZtoD = ZtoD
        self.ZfromD = ZfromD

        self.I = I
        self.ID = ID

        self.grid_initialized = True

    def init_background_flow(self):
        if not self.grid_initialized:
            self.init_grid()

        # background flow on the unique grid
        self.f_u    = self.bg.f(self.yu)
        self.ubar_u = self.bg.u(self.yu)
        self.hbar_u = self.bg.h(self.yu)

        # background flow on the duplicate grid
        self.f_d    = self.bg.f(self.yd)
        self.ubar_d = self.bg.u(self.yd)
        self.hbar_d = self.bg.h(self.yd)

        self.ubarp_d = np.hstack((self.bg.dudy_south(self.yS), self.bg.dudy_north(self.yN)))

        # We will never actually need the value of f at infinity, so we just set it to zero
        # self.f_u[[0, -1]] = 0
        # self.f_d[[0, -1]] = 0

        self.background_flow_initialized = True

    def modes(self, k):
        if not self.background_flow_initialized:
            self.init_background_flow()

        self.k = k

        self.lhs_U = np.block([
            [np.diag(self.Ro*self.ubar_d),
             np.diag(self.Ro*self.ubarp_d - self.f_d) @ self.PtoD,
             np.diag(self.hbar_d) @ self.PtoD ],
        ])
        self.rhs_U = np.block([
            [self.ID,     self.ZtoD,    self.ZtoD   ],
        ])


        # Everything is multiplied through by -1 to make the RHS positive semidefinite
        self.lhs_V = np.block([
            [-np.diag(self.f_u) @ self.PfromD,
             -np.diag(-k**2*self.Ro*self.ubar_u),
             -np.diag(self.hbar_u) @ self.dyC    ],
        ])
        self.rhs_V = np.block([
            [self.ZfromD, k**2*self.I, self.Z],
        ])

        # We replace the first and last row with the boundary condition
        # self.lhs_V[[0,-1],:] = 0
        # self.rhs_V[[0,-1],:] = 0
        # if self.radiation_bc:
        #     self.lhs_V = self.lhs_V.astype(complex)
        #     # set the northward-propagating invariant i k ψ + sqrt(H/F)η = 0
        #     self.lhs_V[ 0, self.idx_ψ[ 0]] = 1j*k
        #     self.lhs_V[ 0, self.idx_η[ 0]] = np.sqrt(self.hbar_u[0]/self.F)
        #     # set the southward-propagating invariant i k ψ - sqrt(H/F)η = 0
        #     self.lhs_V[-1, self.idx_ψ[ 0]] = 1j*k
        #     self.lhs_V[-1, self.idx_η[-1]] = -np.sqrt(self.hbar_u[-1]/self.F)
        # else:
        #     self.lhs_V[ 0, self.idx_ψ[ 0]] = 1
        #     self.lhs_V[-1, self.idx_ψ[-1]] = 1

        # We replace the first and last row with the boundary condition
        if self.radiation_bc:
            self.lhs_V[[0,-1],:] = 0
            self.rhs_V[[0,-1],:] = 0
            self.lhs_V[ 0, self.idx_ψ[ 0]] = 1
            self.lhs_V[-1, self.idx_ψ[-1]] = 1

        self.lhs_H = np.block([
            [self.PfromD,
             self.PfromD @ self.dyD,
             np.diag(self.Ro*self.F*self.ubar_u)],
        ])
        self.rhs_H = np.block([
            [self.ZfromD, self.Z,       self.F*self.I]
        ])


        lhs = np.block([
            [self.lhs_U],
            [self.lhs_V],
            [self.lhs_H]
        ])

        rhs = np.block([
            [self.rhs_U],
            [self.rhs_V],
            [self.rhs_H]
        ])


        c, X = sp.linalg.eig(lhs, rhs)
        # idx = np.isfinite(λ)
        # λ = λ[idx]
        # X = X[:,idx]
        # c = λ/jet.Ro

        idx = np.argsort(np.abs(c))
        self.c = c[idx]
        self.X = X[:,idx]

        self.u = self.X[self.idx_hu,:]/self.hbar_d[:,np.newaxis]
        self.v = 1j*k*self.X[self.idx_ψ,:]/self.hbar_u[:,np.newaxis]
        self.η = self.X[self.idx_η]

        return self.c

##########################################################################################
class OneLayerJet_TL(object):
    '''
    A class implementing a one-layer Rossby-Zhang jet using the rational Chebyshev functions, TL.
    '''

    def __init__(self, δ, Ro, F, b, N, Ln, Ls=None):
        self.δ = δ     # asymmetry parameter
        self.Ro = Ro   # Rossby number
        self.F = F     # baroclinic inverse Burger number
        self.b = b     # beta parameter
        self.N = N     # order of the Chebyshev grid
        self.mapping_Ln = Ln # mapping scale for the northern domain
        if Ls is None:
            self.mapping_Ls = Ln # mapping scale for the southern domain
        else:
            self.mapping_Ls = Ls # mapping scale for the southern domain


        self.bg = OneLayerJet_BackgroundFlow(self.δ, self.Ro, self.F, self.b)

        self.grid_initialized = False
        self.background_flow_initialized = False
        self.operators_initialized = False

    def init_grid(self):
        N = self.N
        if (self.N == 0):
            raise RuntimeError('Number of grid points must be greater than zero!')

        # The suffixes S and N refer to the southern and northern subdomains, respectively.
        # The x refers to the actual Chebyshev grid and y refers to the mapped grid
        self.xS, self.yS, self.dyS = cheb.TLn(self.N, -self.mapping_Ls)
        self.xN, self.yN, self.dyN = cheb.TLn(self.N,  self.mapping_Ln)

        # flip the southern grid and operators around
        self.xS  = self.xS[::-1]
        self.yS  = self.yS[::-1]
        self.dyS = self.dyS[::-1,::-1]

        # remap xN to [0, 1] and xS to [-1, 0]
        self.xS = -(self.xS + 1)/2
        self.xN =  (self.xN + 1)/2


        # make sure the point at zero is actually at zero
        self.xS[-1] = 0
        self.yS[-1] = 0
        self.xN[ 0] = 0
        self.yN[ 0] = 0

        # projects quantities from the unique grid to the duplicate grid
        self.PtoD_full = np.block([
            [np.identity(N+1), np.zeros((N+1, N))],
            [np.zeros((N+1, N)), np.identity(N+1)]
        ])

        # projects quantities from the duplicate grid to the unique grid
        self.PfromD_full = np.block([
            [np.identity(N+1), np.zeros((N+1, N+1))],
            [np.zeros((N, N+2)), np.identity(N)]
        ])
        self.PfromD_full[N, N  ] = 0.5
        self.PfromD_full[N, N+1] = 0.5

        # There are two different ways to construct the derivative matrix for functions with continuous derivatives
        # Picking one or the other leads to weird asymmetries in the derivative matrix, so we average the two together
        # This turns out to be the same as applying the projection operator to dyD
        # dyC1 = np.block([[self.dyS, np.zeros((N+1, N))],
        #                 [np.zeros((N, N)), self.dyN[1:,:]]])
        # dyC2 = np.block([[self.dyS[:-1,:], np.zeros((N, N))],
        #                 [np.zeros((N+1, N)), self.dyN]])
        # self.dyC = (dyC1+dyC2)/2


        # This operator is for functions with discontinuous derivatives at y = 0
        self.dyD_full = np.block([[self.dyS, np.zeros((N+1, N))],
                                  [np.zeros((N+1, N)), self.dyN]])

        self.dyC_full = self.PfromD_full @ self.dyD_full

        self.dyD = self.dyD_full[1:-1,1:-1]
        self.dyC = self.dyC_full[1:-1,1:-1]
        self.PtoD = self.PtoD_full[1:-1,1:-1]
        self.PfromD = self.PfromD_full[1:-1,1:-1]

        # Various zero and identity matrices
        self.Z      = np.zeros((2*N-1, 2*N-1))
        self.ZtoD   = np.zeros((2*N,   2*N-1))
        self.ZfromD = np.zeros((2*N-1, 2*N))

        self.I  = np.identity(2*N-1)
        self.ID = np.identity(2*N)

        self.idx_U = np.arange(2*N-1)
        self.idx_ψ = self.idx_U[-1] + 1 + np.arange(2*N-1)
        self.idx_η = self.idx_ψ[-1] + 1 + np.arange(2*N-1)

        # The duplicate grid (points at infinity and duplicates 0)
        self.xd = np.hstack([self.xS[1:], self.xN[:-1]])
        self.yd = np.hstack([self.yS[1:], self.yN[:-1]])

        # The unique grid (no points at infinity, no duplicates)
        self.xu = np.hstack([self.xS[1:], self.xN[1:-1]])
        self.yu = np.hstack([self.yS[1:], self.yN[1:-1]])

        # make the unique grid the default grid
        self.x = self.xu
        self.y = self.yu

        self.grid_initialized = True

    def init_background_flow(self):
        if not self.grid_initialized:
            self.init_grid()

        # background flow on unique points
        self.f    = self.bg.f(self.y)
        self.ubar = self.bg.u(self.y)
        self.hbar = self.bg.h(self.y)
        self.ζbar = self.bg.ζ(self.y)

        self.background_flow_initialized = True

    def init_operators(self):
        '''
        This method constructs the k-indepdendent parts of the operators.
        '''
        if not self.background_flow_initialized:
            self.init_background_flow()

        # u-equation
        # Note that the u-equation at y = 0 is the average of the two limits
        self.adv_U = np.diag(self.Ro*self.ubar)
        # this is actually the nonlinear coriolis term
        self.cor_U = -np.diag(self.f + self.Ro*self.ζbar)
        self.pgf_U = np.diag(self.hbar)
        self.ten_U = self.I

        # v-equation
        # Note: the v-equation at y = 0 is the average of the v-equations at 0+ and 0-, using the fact
        # that v is continuous
        self.adv_V = -self.Ro*np.diag(self.ubar)  # this term should get multiplied by k**2
        self.cor_V = np.diag(self.f)
        self.pgf_V = np.diag(self.hbar) @ self.dyC
        self.ten_V = -self.I  # this term should get multiplied by k**2

        # η-equation
        self.adv_H = self.Ro*self.F*np.diag(self.ubar)
        self.divUH = self.I
        self.divVH = self.dyC
        self.ten_H = self.F*self.I

        self.operators_initialized = True

    def modes(self, k):
        if not self.operators_initialized:
            self.init_operators()

        self.k = k

        lhs = np.block([
            [self.adv_U, self.cor_U,      self.pgf_U],
            [self.cor_V, self.adv_V*k**2, self.pgf_V],
            [self.divUH, self.divVH,      self.adv_H]
        ])

        sz = self.idx_η[-1] + 1
        rhs = np.zeros((sz, sz))
        rhs[np.ix_(self.idx_U,self.idx_U)] = self.ten_U
        rhs[np.ix_(self.idx_ψ,self.idx_ψ)] = self.ten_V*k**2
        rhs[np.ix_(self.idx_η,self.idx_η)] = self.ten_H

        self.c, self.X = sp.linalg.eig(lhs, rhs)
#         # idx = np.isfinite(λ)
#         # λ = λ[idx]
#         # X = X[:,idx]
#         # c = λ/self.Ro

        idx = np.argsort(self.c.real)
        self.c = self.c[idx]
        self.X = self.X[:,idx]

        self.U_nojump = self.X[self.idx_U,:]
        self.ψ = self.X[self.idx_ψ,:]
        self.η = self.X[self.idx_η]

        self.u_nojump = self.U_nojump/self.hbar[:,np.newaxis]
        self.v = 1j*k*self.ψ/self.hbar[:,np.newaxis]

        # put back in jump in u
        ΔU = self.bg.Δζ*self.ψ[self.N-1,:]/(1 - self.c)
        Up = self.U_nojump[self.N-1,:] + ΔU/2
        Um = self.U_nojump[self.N-1,:] - ΔU/2

        self.U = np.zeros((2*self.N, 3*(2*self.N-1)), dtype=complex)
        self.U[:self.N-1,:] = self.U_nojump[:self.N-1,:]
        self.U[ self.N-1,:] = Um
        self.U[ self.N,  :] = Up
        self.U[self.N+1:,:] = self.U_nojump[self.N:,:]

        self.u =  np.zeros((2*self.N, 3*(2*self.N-1)), dtype=complex)
        self.u[:self.N,:] = self.U[:self.N,:]/self.hbar[:self.N,np.newaxis]
        self.u[self.N:,:] = self.U[self.N:,:]/self.hbar[self.N-1:,np.newaxis]

        return self.c


##########################################################################################
# Utility functions
def normalize(u, v, η):
    maxvals = np.array([np.abs(fld).max() for fld in (u, v, η)])
    idx_fld = np.argmax(maxvals)

    if idx_fld == 0: # u is maximal
        idx = np.argmax(np.abs(u))
        norm = 1/u[idx]
    elif idx_fld == 1: # v is maximal
        idx = np.argmax(np.abs(v))
        norm = 1/v[idx]
    elif idx_fld == 2: # η is maximal
        idx = np.argmax(np.abs(η))
        norm = 1/η[idx]

    return norm*u, norm*v, norm*η

def winnow(k, jet1, jet2):
    c1 = jet1.modes(k)
    c2 = jet2.modes(k)

    # intermodal separation
    σ = np.zeros_like(c1, dtype=float)
    σ[0] = np.abs(c1[0] - c1[1])
    σ[1:-1] = .5*(np.abs(c1[1:-1] - c1[0:-2]) + np.abs(c1[2:] - c1[1:-1]))
    σ[-1] = np.abs(c1[-1] - c1[-2])

    idx_nearest = np.argmin(np.abs(np.atleast_2d(c1).T - np.atleast_2d(c2)), axis=1)
    δ_nearest = np.min(np.abs(np.atleast_2d(c1).T - np.atleast_2d(c2)), axis=1)/σ

    idx1_good = np.nonzero(δ_nearest < 1e-6)[0]

    return c1[idx1_good]
