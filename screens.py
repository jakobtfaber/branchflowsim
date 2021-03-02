
# Phase Screens for Wave Optics Simulation (Branched Flow Honors Thesis 2021)
# Author: Jakob Faber

from imports import *

from scipy.special import gamma, kv
from scipy import linalg
from scipy.interpolate import interp2d
import numpy as np
from numpy import pi

# Numba compiles python code to machine code for faster execution
try:
    import numba
except:
    numba = None  

class Screen():

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                    inner=0.001, outer = 1e3, ns=256, nf=256, dlam=0.25, nx=None, 
                    ny=None, dx=None, dy=None, seed = None):

        """
        Portions of this simulation have been borrowed from Scintools by D. Reardon
        which is, in turn, based on the Coles et al. 2010 simulation.

        mb2: Max Born parameter for strength of scattering
        rf: Fresnel scale
        ds (or dx,dy): Spatial step sizes with respect to rf
        alpha: Structure function exponent (Kolmogorov = 5/3)
        ar: Anisotropy axial ratio
        psi: Anisotropy orientation
        inner: Inner scale w.r.t rf - should generally be smaller than ds
        ns (or nx,ny): Number of spatial steps
        nf: Number of frequency steps.
        dlam: Fractional bandwidth relative to centre frequency
        lamsteps: Boolean to choose whether steps in lambda or freq
        seed: Seed number, or use "-1" to shuffle
        """

        self.mb2 = mb2
        self.rf = rf
        self.dx = dx if dx is not None else ds
        self.dy = dy if dy is not None else ds
        self.alpha = alpha
        self.ar = ar
        self.psi = psi
        self.inner = float(inner)
        self.outer = float(outer)
        self.nx = nx if nx is not None else ns
        self.ny = ny if ny is not None else ns
        self.nf = nf
        self.dlam = dlam
        self.seed = seed

        if seed != None:
            self.nscreen = 1

        ns = 1
        lenx = self.nx*self.dx
        leny = self.ny*self.dy
        dqx = 2*np.pi/lenx
        dqy = 2*np.pi/leny
        a2 = self.alpha*0.5
        aa = 1.0+a2
        ab = 1.0-a2
        cdrf = 2.0**(self.alpha)*np.cos(self.alpha*np.pi*0.25)\
            * gamma(aa)/self.mb2

        cmb2 = self.alpha*self.mb2 / (4*np.pi *
                                      gamma(ab)*np.cos(self.alpha *
                                                       np.pi*0.25)*ns)
        self.consp = cmb2*dqx*dqy/(self.rf**self.alpha)

    def discrete_sample_spectrum(self, kx=0, ky=0):
        
        cs = np.cos(self.psi*np.pi/180)
        sn = np.sin(self.psi*np.pi/180)
        r = self.ar
        con = np.sqrt(self.consp)
        alf = -(self.alpha+2)/4
        # anisotropy parameters
        a = (cs**2)/r + r*sn**2
        b = r*cs**2 + sn**2/r
        c = 2*cs*sn*(1/r-r)
        q2 = a * np.power(kx, 2) + b * np.power(ky, 2) + c*np.multiply(kx, ky)
        # isotropic inner scale
        S_phi_q = con*np.multiply(np.power(q2, alf),
                              np.exp(-(np.add(np.power(kx, 2),
                                              np.power(ky, 2))) *
                                     self.inner**2/2))
        self.S_phi_q = S_phi_q

        return S_phi_q

    def kolmogorov(self, delta):

        """
        Generate a periodic phase screen in x and y
        """
        random.seed(self.seed)  # Set the seed, if any

        nx_ = int(self.nx/2 + 1)
        ny_ = int(self.ny/2 + 1)

        w = np.zeros([self.nx, self.ny])  # initialize array
        #dqx = 2*np.pi/(self.dx*self.nx)
        #dqy = 2*np.pi/(self.dy*self.ny)
        dqx = 2*np.pi/(delta*self.nx)
        dqy = 2*np.pi/(delta*self.ny)

        # first do ky=0 line
        k = np.arange(2, nx_+1)
        w[k-1, 0] = self.discrete_sample_spectrum(kx=(k-1)*dqx, ky=0)
        w[self.nx+1-k, 0] = w[k, 0]

        # then do kx=0 line
        ll = np.arange(2, ny_+1)
        w[0, ll-1] = self.discrete_sample_spectrum(kx=0, ky=(ll-1)*dqy)
        w[0, self.ny+1-ll] = w[0, ll-1]

        # now do the rest of the field
        kp = np.arange(2, nx_+1)
        k = np.arange((nx_+1), self.nx+1)
        km = -(self.nx-k+1)
        for il in range(2, ny_+1):
            w[kp-1, il-1] = self.discrete_sample_spectrum(kx=(kp-1)*dqx, ky=(il-1)*dqy)
            w[k-1, il-1] = self.discrete_sample_spectrum(kx=km*dqx, ky=(il-1)*dqy)
            w[self.nx+1-kp, self.ny+1-il] = w[kp-1, il-1]
            w[self.nx+1-k, self.ny+1-il] = w[k-1, il-1]

        # done the whole screen weights, now generate complex gaussian array
        xyp = np.multiply(w, np.add(randn(self.nx, self.ny),
                                    1j*randn(self.nx, self.ny)))


        xyp = np.real(fft2(xyp))
        self.w = w
        self.xyp = xyp

        return self.xyp, self.w #returns screen and weights

    def ft_phase_screen(self, seed=None):
        """
        Creates a random phase screen with Von Karmen statistics.
        (Schmidt 2010)
        
        Parameters:
            rF (float): fresnel scale
            N (int): Size of phase scrn in pxls
            delta (float): size in Metres of each pxl
        Returns:
            ndarray: np array representing phase screen
        """

        np.random.seed(seed)

        del_f = 1./(self.nx*self.dx)

        fx = np.arange(-self.nx/2., self.nx/2.) * del_f
        (fx, fy) = np.meshgrid(fx,fx)
        f = np.sqrt(fx**2. + fy**2.)
        fm = 5.92/self.inner/(2*np.pi)
        f0 = 1./self.outer

        PSD_phi = (0.023*self.rf**(-self.alpha) * np.exp(-1*((f/fm)**2)) / (((f**2) + (f0**2))**(11./6)))
        PSD_phi[int(self.nx/2), int(self.nx/2)] = 0

        Cn = ((np.random.normal(size=(self.nx, self.ny))+1j * np.random.normal(size=(self.nx, self.ny))) * np.sqrt(PSD_phi)*del_f)
        xyp = (np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Cn))) * (Cn.shape[0] * del_f) ** 2.).real

        self.xyp = xyp        

        return self.xyp

    def phase_covariance(self, separations):
        """
        Calculate the phase covariance between two points seperated by `r`, 
        in turbulence with a given `r0 and `L0`.
        Uses equation 5 from Assemat and Wilson, 2006.
        Parameters:
            r (float, ndarray): Seperation between points in metres (can be ndarray)
            r0 (float): Fried parameter of turbulence in metres
            L0 (float): Outer scale of turbulence in metres
        """
        # Make sure everything is a float to avoid nasty surprises in division!
        sep = np.float32(separations)
        self.sep = sep

        # Get rid of any zeros
        r += 1e-40

        A = (self.outer / self.inner) ** (5. / 3)

        B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (np.pi ** (8. / 3))
        B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)

        C = (((2 * np.pi * self.sep) / self.outer) ** (5. / 6)) * kv(5. / 6, (2 * np.pi * self.sep) / self.outer)

        cov = A * B1 * B2 * C
        self.cov = cov

        return self.cov

