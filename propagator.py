
# KDI for Wave Optics Simulation (Branched Flow Honors Thesis 2021)
# i.e., solving the Kirchhoff Diffraction Integral using Split Step FFT Methods
# Author: Jakob Faber

from imports import *

class Propagator():

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0, nscreen = 2,
                    inner=0.001, ns=256, nf=256, nx=None, ny=None, dx=None, dy=None):

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
        self.lamsteps = lamsteps
        self.seed = seed
        self.nscreen = nscreen
        # For convolution with Frensel integral
        ns = 1
        lenx = self.nx*self.dx
        leny = self.ny*self.dy
        self.ffconx = (2.0/(ns*lenx*lenx))*(np.pi*self.rf)**2
        self.ffcony = (2.0/(ns*leny*leny))*(np.pi*self.rf)**2

    def plane_fresnel(self, xye, scale):

        nx2 = int(self.nx / 2) + 1
        ny2 = int(self.ny / 2) + 1
        filt = np.zeros([nx2, ny2], dtype=np.dtype(np.csingle))
        q2x = np.linspace(0, nx2-1, nx2)**2 * scale * self.ffconx
        for ly in range(0, ny2):
            q2 = q2x + (self.ffcony * (ly**2) * scale)
            filt[:, ly] = np.cos(q2) - 1j * np.sin(q2)

        xye[0:nx2, 0:ny2] = np.multiply(xye[0:nx2, 0:ny2], filt[0:nx2, 0:ny2])
        xye[self.nx:nx2-1:-1, 0:ny2] = np.multiply(
            xye[self.nx:nx2-1:-1, 0:ny2], filt[1:(nx2 - 1), 0:ny2])
        xye[0:nx2, self.ny:ny2-1:-1] =\
            np.multiply(xye[0:nx2, self.ny:ny2-1:-1], filt[0:nx2, 1:(ny2-1)])
        xye[self.nx:nx2-1:-1, self.ny:ny2-1:-1] =\
            np.multiply(xye[self.nx:nx2-1:-1, self.ny:ny2-1:-1],
                        filt[1:(nx2-1), 1:(ny2-1)])
        return xye


    def plane_free_space(self, verbose=False):

        efield = np.zeros([self.nx, self.nf],
               dtype=np.dtype(np.csingle)) + \
               1j*np.zeros([self.nx, self.nf],
                           dtype=np.dtype(np.csingle))
        for ifreq in range(0, self.nf):
            if verbose:
                if ifreq % round(self.nf/100) == 0:
                    print(int(np.floor((ifreq+1)*100/self.nf)), '%')
            else:
                frfreq = 1.0 +\
                    self.dlam * (-0.5 + ifreq / self.nf)
                scale = 1 / frfreq
            scaled = scale
            
            # Propagate through multiple (nscreen) screens
            for jscr in range(0, self.nscreen):
                if jscr == 0:
                    xye = fft2(np.exp(1j * self.xyp * scaled))
                else:
                    xye = fft2(xye * np.exp(1j * self.xyp * scaled))
                xye = self.frfilt3(xye, scale) #propagate through free space to the next screen
                xye = ifft2(xye)
            gam = 0
            spe[:, ifreq] = xye[:, int(np.floor(self.ny / 2))] / scale**gam

        xyi = np.real(np.multiply(xye, np.conj(xye)))

        self.xyi = xyi
        self.spe = spe
        return self.xyi, self.spe #returns intensity and electric field

# N.B.:
# - A point source in an extended medium ought to be modeled differently
#   from a plane wave, since there is a periodicity implied in the DFT. We
#   must therefore recast the split-step FFT algorithm in a spherically
#   diverging cordinate system.





