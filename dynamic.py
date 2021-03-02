# Generate Dynamic Spectrum

from imports import *

class Dynamic():

    def __init__(self, ds=0.01, ns=256, nf=256, dlam=0.25, nx=None, ny=None, dx=None, dy=None):

        self.dx = dx if dx is not None else ds
        self.dy = dy if dy is not None else ds
        self.nx = nx if nx is not None else ns
        self.ny = ny if ny is not None else ns
        self.nf = nf
        self.dlam = dlam

    def spectrum(self):

        if self.nf == 1:
            print('no spectrum because nf=1')

        # dynamic spectrum
        dynspec = np.real(np.multiply(self.spe, np.conj(self.spe)))
        self.dynspec = dynspec

        self.x = np.linspace(0, self.dx*(self.nx), (self.nx+1))
        ifreq = np.arange(0, self.nf+1)
        lam_norm = 1.0 + self.dlam * (ifreq - 1 - (self.nf / 2)) / self.nf
        self.lams = lam_norm / np.mean(lam_norm)
        frfreq = 1.0 + self.dlam * (-0.5 + ifreq / self.nf)
        self.freqs = frfreq / np.mean(frfreq)

        return self.dynspec, self.freqs, self.x