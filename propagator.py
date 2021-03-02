
# KDI for Wave Optics Simulation (Branched Flow Honors Thesis 2021)
# i.e., solving the Kirchhoff Diffraction Integral using Split Step FFT Methods
# Author: Jakob Faber

from imports import *
import sources as srcs
import screens as scrs
from numba import jit

scrkol = scrs.Screen()
s = srcs.Source()

class Propagator():

    def __init__(self, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0, nscreen=2, PropDist=10e3, dlam=0.25, wavelen=1,
                    inner=0.001, ns=256, nf=256, nx=None, ny=None, dx=None, dy=None, del_x=5e-3, 
                    Rdel_x=5e-3, loon=1, seed = None):

        self.rf = rf
        self.dx = dx if dx is not None else ds
        self.dy = dy if dy is not None else ds
        self.alpha = alpha
        self.ar = ar
        self.psi = psi
        self.inner = inner
        self.nx = nx if nx is not None else ns
        self.ny = ny if ny is not None else ns
        self.nxf = nf
        self.dlam = dlam 
        self.nscreen = nscreen    # Number of phase screens
        self.PropDist = PropDist  # Propagation distance(Path Length) [m]
        
        # Include sub-harmonic compensation?
        self.loon = loon
        # Simulation output
        self.Output = np.zeros((self.nx, self.ny))

        # For convolution with Frensel integral
        ns = 1
        lenx = self.nx*self.dx
        leny = self.ny*self.dy
        self.ffconx = (2.0/(ns*lenx*lenx))*(np.pi*self.rf)**2
        self.ffcony = (2.0/(ns*leny*leny))*(np.pi*self.rf)**2

        #self.ispace = ispace
        #self.icomplex = icomplex
        #self.ospace = ospace
        #self.dz = dz
        self.wavelen = wavelen
        self.k = 2 * np.pi / self.wavelen  # Optical wavenumber [rad/m]
        self.del_x = del_x # Sampling interval at source plane
        self.Rdel_x = Rdel_x  # Sampling interval at receiver plane

        # Place holders for geometry/source variables
        self.Source = None
        self.r1 = None
        self.x1 = None
        self.y1 = None
        self.rR = None
        self.xR = None
        self.yR = None
        self.Uout = None

        x = np.linspace(-self.nx / 2, (self.nx / 2) - 1, self.nx) * self.del_x
        y = np.linspace(-self.nx / 2, (self.nx / 2) - 1, self.nx) * self.del_x
        self.x1, self.y1 = np.meshgrid(x, y)
        self.r1 = np.sqrt(self.x1 ** 2 + self.y1 ** 2)

        x = np.linspace(-self.nx / 2, (self.nx / 2) - 1, self.nx) * self.Rdel_x
        y = np.linspace(-self.nx / 2, (self.nx / 2) - 1, self.nx) * self.Rdel_x
        self.xR, self.yR = np.meshgrid(x, y)
        self.rR = np.sqrt(self.xR ** 2 + self.yR ** 2)

        # Set Propagation Geometry / Screen placement
        self.dzProps = np.ones(self.nscreen + 2) * (self.PropDist / self.nscreen)
        self.dzProps[0:2] = 0.5 * (self.PropDist / self.nscreen)
        self.dzProps[self.nscreen:self.nscreen + 2] = 0.5 * (self.PropDist / self.nscreen)

        self.PropLocs = np.zeros(self.nscreen + 3)

        for zval in range(0, self.nscreen + 2):
            self.PropLocs[zval + 1] = self.PropLocs[zval] + self.dzProps[zval]

        self.ScrnLoc = np.concatenate((self.PropLocs[1:self.nscreen],
                                       np.array([self.PropLocs[self.nscreen + 1]])), axis=0)

        self.FracPropDist = self.PropLocs / self.PropDist

        self.PropSampling = (self.Rdel_x - self.del_x) * self.FracPropDist + self.del_x

        self.SamplingRatioBetweenScreen = \
            self.PropSampling[1:len(self.PropSampling)] \
            / self.PropSampling[0:len(self.PropSampling) - 1]

    def get_screen(self, delta, scr=scrkol):

        kolscr, weights = scr.kolmogorov(delta)
        #self.kolscr = kolscr
        return kolscr

    def get_source(self, source='point', src=s):
        
        if source == 'point':
            sc = src.PointSource()
        elif source == 'plane':
            sc = src.PlaneSource()

        return sc

    def MakeSGB(self):

        # Construction of Super Gaussian Boundary
        rad = self.r1 * (self.nx);
        w = 0.55 * self.nx
        sg = np.exp(- ((rad / w) ** 16.0))

        return sg


    def plane_fresnel(self, xye, scale):

        nx_ = int(self.nx / 2) + 1
        ny_ = int(self.ny / 2) + 1
        filt = np.zeros([nx_, ny_], dtype=np.dtype(np.csingle))
        q2x = np.linspace(0, nx_-1, nx_)**2 * scale * self.ffconx
        for ly in range(0, ny_):
            q2 = q2x + (self.ffcony * (ly**2) * scale)
            filt[:, ly] = np.cos(q2) - 1j * np.sin(q2)

        xye[0:nx_, 0:ny_] = np.multiply(xye[0:nx_, 0:ny_], filt[0:nx_, 0:ny_])
        xye[self.nx:nx_-1:-1, 0:ny_] = np.multiply(
            xye[self.nx:nx_-1:-1, 0:ny_], filt[1:(nx_ - 1), 0:ny_])
        xye[0:nx_, self.ny:ny_-1:-1] =\
            np.multiply(xye[0:nx_, self.ny:ny_-1:-1], filt[0:nx_, 1:(ny_-1)])
        xye[self.nx:nx_-1:-1, self.ny:ny_-1:-1] =\
            np.multiply(xye[self.nx:nx_-1:-1, self.ny:ny_-1:-1],
                        filt[1:(nx_-1), 1:(ny_-1)])
        return xye

    def plane_free_space(self, verbose=False):

        efield = np.zeros([self.nx, self.nxf],
               dtype=np.dtype(np.csingle)) + \
               1j*np.zeros([self.nx, self.nxf],
                             dtype=np.dtype(np.csingle))

        # Generate a pack of delta-correlated Kolmogorov turbulent phase screens
        screen_pack = np.zeros((self.nscreen, self.nx, self.ny), dtype = 'complex128')
        for jscr in range(0, self.nscreen):
            kolscr = self.get_screen()
            screen_pack[jscr] = kolscr

        # Setup an array to temporarily store the fields during iterative convolutions
        field_stack = np.zeros((self.nscreen, self.nx, self.ny), dtype = 'complex128')
        for ifreq in range(0, self.nxf):
            if verbose:
                if ifreq % round(self.nxf/100) == 0:
                    print(int(np.floor((ifreq+1)*100/self.nxf)), '%')
            else:
                frfreq = 1.0 +\
                    self.dlam * (-0.5 + ifreq / self.nxf)
                scale = 1 / frfreq
            scaled = scale
            
            if self.nscreen == 1:
            # Propagate through a single phase screen
                sc = self.get_source()
                xye = fft2(sc * np.exp(1j * screen_pack[0] * scaled))
                xye = self.plane_fresnel(xye, scale) #propagate through free space to the next screen
                field_stack[0] = ifft2(xye)

            else:
            # Propagate through multiple (nscreen) screens
                for jscr in range(0, self.nscreen):
                    if jscr == 0:
                        sc = self.get_source()
                        field_stack[0] = fft2(sc * np.exp(1j * screen_pack[jscr] * scaled))
                    else:
                        xye = fft2(field_stack[jscr-1] * np.exp(1j * screen_pack[jscr] * scaled))
                        xye = self.plane_fresnel(xye, scale) #propagate through free space to the next screen
                        field_stack[jscr] = ifft2(xye)

            gam = 0
            efield[:, ifreq] = field_stack[self.nscreen-1][:, int(np.floor(self.ny / 2))] / scale**gam

        amp = np.real(np.multiply(field_stack[self.nscreen-1], np.conj(field_stack[self.nscreen-1])))

        self.amp = amp
        self.efield = efield
        return self.amp, self.efield #returns intensity and electric field


    #def angularSpectrum(inputComplexAmp, wvl, inputSpacing, outputSpacing, z):
    #    """
    #    Propogates light complex amplitude using an angular spectrum algorithm
#
    #    Parameters:
    #        inputComplexAmp (ndarray): Complex array of input complex amplitude
    #        wvl (float): Wavelength of light to propagate
    #        inputSpacing (float): The spacing between points on the input array in metres
    #        outputSpacing (float): The desired spacing between points on the output array in metres
    #        z (float): Distance to propagate in metres
#
    #    Returns:
    #        ndarray: propagated complex amplitude
    #    """
    #    
    #    # If propagation distance is 0, don't bother 
    #    if z==0:
    #        return self.icomplex
#
    #    N = self.icomplex.shape[0] #Assumes Uin is square.
    #    k = 2*numpy.pi/self.wavelen     #optical wavevector
#
    #    (x,y) = numpy.meshgrid(self.ispace*numpy.arange(-N/2,N/2),
    #                             self.ispace*numpy.arange(-N/2,N/2))
    #    rsq = (x**2 + y**2) + 1e-10
#
    #    #Spatial Frequencies (of source plane)
    #    df = 1. / (N*inputSpacing)
    #    fX,fY = numpy.meshgrid(df*numpy.arange(-N/2,N/2),
    #                           df*numpy.arange(-N/2,N/2))
    #    fsq = fX**2 + fY**2
#
    #    #Scaling Param
    #    mag = float(self.ospace)/self.ospace
#
    #    #Observation Plane Co-ords
    #    x_,y_ = numpy.meshgrid( self.ospace*numpy.arange(-N/2,N/2),
    #                            self.ospace*numpy.arange(-N/2,N/2) )
    #    r_sq = x_**2 + y_**2
#
    #    #Quadratic phase factors
    #    Q1 = numpy.exp( 1j * k/2. * (1-mag)/z * rsq)
#
    #    Q2 = numpy.exp(-1j * numpy.pi**2 * 2 * z/mag/k*fsq)
#
    #    Q3 = numpy.exp(1j * k/2. * (mag-1)/(mag*z) * r_sq)
#
    #    #Compute propagated field
    #    outputComplexAmp = Q3 * fouriertransform.ift2(
    #                    Q2 * fouriertransform.ft2(Q1 * self.ispace/mag,self.ispace), df1)
    #    return outputComplexAmp
#
#
    #def oneStepFresnel(Uin, wvl, d1, z):
    #    """
    #    Fresnel propagation using a one step Fresnel propagation method.
#
    #    Parameters:
    #        Uin (ndarray): A 2-d, complex, input array of complex amplitude
    #        wvl (float): Wavelength of propagated light in metres
    #        d1 (float): spacing of input plane
    #        z (float): metres to propagate along optical axis
#
    #    Returns:
    #        ndarray: Complex ampltitude after propagation
    #    """
    #    N = Uin.shape[0]    #Assume square grid
    #    k = 2*numpy.pi/wvl  #optical wavevector
#
    #    #Source plane coordinates
    #    x1,y1 = numpy.meshgrid( numpy.arange(-N/2.,N/2.) * d1,
    #                            numpy.arange(-N/2.,N/2.) * d1)
    #    #observation plane coordinates
    #    d2 = wvl*z/(N*d1)
    #    x2,y2 = numpy.meshgrid( numpy.arange(-N/2.,N/2.) * d2,
    #                            numpy.arange(-N/2.,N/2.) * d2 )
#
    #    #evaluate Fresnel-Kirchoff integral
    #    A = 1/(1j*wvl*z)
    #    B = numpy.exp( 1j * k/(2*z) * (x2**2 + y2**2))
    #    C = fouriertransform.ft2(Uin *numpy.exp(1j * k/(2*z) * (x1**2+y1**2)), d1)
#
    #    Uout = A*B*C
#
    #    return Uout


    def splitstep(self, Uin, PhaseScreenStack):
        # Propagation/Fresnel Diffraction Integral

        sg = self.MakeSGB()  # Generates SGB

        SamplingRatio = self.SamplingRatioBetweenScreen

        a = int(self.nx / 2)

        nx, ny = np.meshgrid(range(-a, a), range(-a, a))

        # Initial Propagation from source plane to first screen location
        P0 = np.exp(1j * (self.k / (2 * self.dzProps[0])) * (self.r1 ** 2) * (1 - SamplingRatio[0]))

        Uin = P0 * self.Source * np.exp(1j * PhaseScreenStack[:, :, 0])

        for pcount in range(1, len(self.PropLocs) - 2):
            UinSpec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Uin)))

            # Set spatial frequencies at propagation plane
            deltaf = 1 / (self.nx * self.PropSampling[pcount])
            fX = nx * deltaf
            fY = ny * deltaf
            fsq = fX ** 2 + fY ** 2

            # Quadratic Phase Factor
            QuadPhaseFac = np.exp(-1j * np.pi * self.wvl * self.dzProps[pcount] \
                                  * SamplingRatio[pcount] * fsq)

            Uin = np.fft.ifftshift(np.fft.ifft2( \
                np.fft.ifftshift(UinSpec * QuadPhaseFac)))

            Uin = Uin * sg * np.exp(1j * PhaseScreenStack[:, :, pcount - 1])

        PF = np.exp(1j * (self.k / (2 * self.dzProps[-1])) * (self.rR ** 2) * (SamplingRatio[-1]))

        Uout = PF * Uin

        return Uout

    def ism_prop(self):
        """
        Perform a turbulence simulation that accounts for the changing sampling planes over the course of the propagation
        through the atmosphere

        :return:
        """
        # initialize phase screen array
        phz = np.zeros(shape=(self.nx, self.nx, self.nscreen))
        phz_lo = np.zeros(shape=(self.nx, self.nx, self.nscreen))
        phz_hi = np.zeros(shape=(self.nx, self.nx, self.nscreen))

        for idxscr in range(0, self.nscreen, 1):

            # Retrieve the current sampling at the current propagation distance
            deltax_dz = self.PropSampling[idxscr]

            phz_hi[:, :, idxscr] = self.get_screen(deltax_dz)
            # FFT-based phase screens

            #phz_lo[:, :, idxscr] = self.SubHarmonicComp(self.nxumSubHarmonics,  deltax_dz)
            # sub harmonics
            phz[:, :, idxscr] = phz_hi[:, :, idxscr] #self.loon * phz_lo[:, :, idxscr] + phz_hi[:, :, idxscr]
            # subharmonic compensated phase screens

        # Simulating propagation
        s = self.get_source()
        self.Output = self.splitstep(s, phz)#np.exp(1j * phz))

        return self.Output


# N.B.:
# - A point source in an extended medium ought to be modeled differently
#   from a plane wave, since there is a periodicity implied in the DFT. We
#   must therefore recast the split-step FFT algorithm in a spherically
#   diverging cordinate system.





