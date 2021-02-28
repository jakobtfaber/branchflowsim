
# Phase Screens for Wave Optics Simulation (Branched Flow Honors Thesis 2021)
# Author: Jakob Faber

from imports import *

class Screen():

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                    inner=0.001, ns=256, nf=256, dlam=0.25, nx=None, 
                    ny=None, dx=None, dy=None):

        self.mb2 = mb2
        self.rf = rf
        self.dx = dx if dx is not None else ds
        self.dy = dy if dy is not None else ds
        self.alpha = alpha
        self.ar = ar
        self.psi = psi
        self.inner = inner
        self.nx = nx if nx is not None else ns
        self.ny = ny if ny is not None else ns
        self.nf = nf
        self.dlam = dlam
        self.lamsteps = lamsteps
        self.seed = seed

    def kolmogorov(self):

        """
        Get phase screen in x and y
        """
        random.seed(self.seed)  # Set the seed, if any

        nx_ = int(self.nx/2 + 1)
        ny_ = int(self.ny/2 + 1)

        w = np.zeros([self.nx, self.ny])  # initialize array
        dqx = 2*np.pi/(self.dx*self.nx)
        dqy = 2*np.pi/(self.dy*self.ny)

        # first do ky=0 line
        k = np.arange(2, nx_+1)
        w[k-1, 0] = self.swdsp(kx=(k-1)*dqx, ky=0)
        w[self.nx+1-k, 0] = w[k, 0]

        # then do kx=0 line
        ll = np.arange(2, ny_+1)
        w[0, ll-1] = self.swdsp(kx=0, ky=(ll-1)*dqy)
        w[0, self.ny+1-ll] = w[0, ll-1]

        # now do the rest of the field
        kp = np.arange(2, nx_+1)
        k = np.arange((nx_+1), self.nx+1)
        km = -(self.nx-k+1)
        for il in range(2, ny_+1):
            w[kp-1, il-1] = self.swdsp(kx=(kp-1)*dqx, ky=(il-1)*dqy)
            w[k-1, il-1] = self.swdsp(kx=km*dqx, ky=(il-1)*dqy)
            w[self.nx+1-kp, self.ny+1-il] = w[kp-1, il-1]
            w[self.nx+1-k, self.ny+1-il] = w[k-1, il-1]

        # done the whole screen weights, now generate complex gaussian array
        xyp = np.multiply(w, np.add(randn(self.nx, self.ny),
                                    1j*randn(self.nx, self.ny)))


        xyp = np.real(fft2(xyp))
        self.w = w
        self.xyp = xyp

        return self.xyp, self.w #returns screen and weights

