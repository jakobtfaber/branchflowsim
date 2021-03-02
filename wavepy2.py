"""

Changes made by Greg Badura and Cody Fernandez, © 2019 Georgia Tech Research Corporation, 7/16/2019

Updates include:
- Correction for propagation plane sampling at the locations of the phase screens
- Ability to evolve wind screens over time to model turbulence propagation through a windy atmosphere
- Ability to use an expanding Gaussian beam, as defined by a radius of curvature

"""


import numpy as np
from math import pi, gamma, cos, sin
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import scipy.ndimage
from scipy.stats import gaussian_kde
from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.fftpack as sf
import skimage.restoration
from matplotlib.patches import Circle


class wavepy():
    def __init__(self, simOption=0, N=256, SideLen=1.0, NumScr=10, DRx=0.1, dx=5e-3,
                 wvl=1e-6, PropDist=10e3, Cn2=1e-16, loon=1, aniso=1.0, Rdx=5e-3, f_curv = -10e3,
                 L0 = 1e3, W0 = 5e-2, vx_list = None, vy_list = None, beam_rider_quadrant = 1):
        self.N = N  # number of grid points per side
        self.SideLen = SideLen  # Length of one side of square phase secreen [m]
        self.dx = dx  # Sampling interval at source plane
        self.Rdx = Rdx  # Sampling interval at receiver plane
        self.L0 = L0  # Turbulence outer scale [m]
        self.l0 = 1e-3  # Turbulence inner scale [m]
        self.NumScr = NumScr  # Number of screens Turn into input variable
        self.DRx = DRx  # Diameter of aperture [m]
        self.wvl = wvl  # Wavelength [m]
        self.PropDist = PropDist  # Propagation distance(Path Length) [m]
        self.Cn2 = Cn2
        self.simOption = simOption  # Simulation type (i.e. spherical, plane)
        self.theta = 0  # Angle of anisotropy [deg]
        self.aniso = aniso  # Anisotropy magnitude
        self.alpha = 22.0  # Power Law exponent 22 = 11/3 (Kolmogorov)
        self.k = 2 * pi / self.wvl  # Optical wavenumber [rad/m]
        self.NumSubHarmonics = 5  # Number of subharmonics
        self.w0 = W0 # the gaussian width of the beam
        self.DTx = np.sqrt(2) * self.w0 #0.1  # Transmitting aperture size for Gauss, as defined by beam waist [m]
        # self.w0 = W0 # (np.exp(-1) * self.DTx)
        self.fcurv = f_curv # the radius of curvature of the beam at the aperture of the source, negative is diverging
        self.vx_list = vx_list # the wind speed tranverse to wave in x direction in m/s at the phase screen locations
        self.vy_list = vy_list # the wind speed tranverse to wave in y direction in m/s at the phase screen locations
        self.update_time = None # Placeholder for the phase screen update time
        self.alpha_memory = 0.995 # A default alpha memory term for wind screen evolution
        self.number_evol_tint = None # the number of time evolutions in a time simulation
        self.evol_dir_stub = None # the directory for holding the output phase screens that are evolved over time
        self.beam_rider_quadrant = beam_rider_quadrant # 1,2,3,4 denotes top, right, bottom, left, respectively


        # Include sub-harmonic compensation?
        self.loon = loon
        # Simulation output
        self.Output = np.zeros((N, N))

        # Place holders for geometry/source variables
        self.Source = None
        self.r1 = None
        self.x1 = None
        self.y1 = None
        self.rR = None
        self.xR = None
        self.yR = None
        self.Uout = None

        x = np.linspace(-self.N / 2, (self.N / 2) - 1, self.N) * self.dx
        y = np.linspace(-self.N / 2, (self.N / 2) - 1, self.N) * self.dx
        self.x1, self.y1 = np.meshgrid(x, y)
        self.r1 = np.sqrt(self.x1 ** 2 + self.y1 ** 2)

        if simOption == 0:
            # Plane Wave source (default)
            self.Source = self.PlaneSource()


        elif simOption == 1:
            # Spherical Wave Source
            self.Source = self.PointSource()


        elif simOption == 2:
            # Collimated Gaussian Source
            self.Source = self.CollimatedGaussian()

        elif simOption == 3:
            # Flatte Point Source
            self.Source = self.FlattePointSource()

        elif simOption == 100:
            # A diverging gaussian beam
            self.Source = self.FocusedBeam()

        elif simOption == 200:
            # A beam rider, specified by a given quadrant
            self.Source = self.BeamRider()


        x = np.linspace(-self.N / 2, (self.N / 2) - 1, self.N) * self.Rdx
        y = np.linspace(-self.N / 2, (self.N / 2) - 1, self.N) * self.Rdx
        self.xR, self.yR = np.meshgrid(x, y)
        self.rR = np.sqrt(self.xR ** 2 + self.yR ** 2)

        # Set Propagation Geometry / Screen placement
        self.dzProps = np.ones(self.NumScr + 2) * (self.PropDist / self.NumScr)
        self.dzProps[0:2] = 0.5 * (self.PropDist / self.NumScr)
        self.dzProps[self.NumScr:self.NumScr + 2] = 0.5 * (self.PropDist / self.NumScr)

        self.PropLocs = np.zeros(self.NumScr + 3)

        for zval in range(0, self.NumScr + 2):
            self.PropLocs[zval + 1] = self.PropLocs[zval] + self.dzProps[zval]

        self.ScrnLoc = np.concatenate((self.PropLocs[1:self.NumScr],
                                       np.array([self.PropLocs[self.NumScr + 1]])), axis=0)

        self.FracPropDist = self.PropLocs / self.PropDist

        self.PropSampling = (self.Rdx - self.dx) * self.FracPropDist + self.dx

        self.SamplingRatioBetweenScreen = \
            self.PropSampling[1:len(self.PropSampling)] \
            / self.PropSampling[0:len(self.PropSampling) - 1]

        # Set derived values
        self.r0 = (0.423 * (self.k) ** 2 * self.Cn2 * self.PropDist) ** (-3.0 / 5.0)
        self.r0scrn = (0.423 * ((self.k) ** 2) * self.Cn2 * (self.PropDist / self.NumScr)) ** (-3.0 / 5.0)
        self.log_ampl_var = 0.3075 * ((self.k) ** 2) * ((self.PropDist) ** (11.0 / 6.0)) * self.Cn2
        self.phase_var = 0.78 * (self.Cn2) * (self.k ** 2) * self.PropDist * (self.L0 ** (-5.0 / 3.0))
        self.rho_0 = (1.46 * self.Cn2 * self.k ** 2 * self.PropDist) ** (-5.0 / 3.0)
        self.rytovNum = np.sqrt(1.23 * self.Cn2 * (self.k ** (7 / 6)) * (self.PropDist ** (11 / 6)))
        self.rytovVar = self.rytovNum ** 2


    def FocusedBeam(self ):
        """
        Create a focused gaussian beam using equation (2.148) from the Smith's (2010) textbook. This equation assumes
        that the radius of curvature of the beam is focused at the end propagation distance.

        Note that a negative value for the radius of curvature denotes diverging beam and a positive value is a
        converging beam.
        """
        source = np.exp(-(self.r1 ** 2)*((1/ self.w0 ** 2) + 1j*(np.pi/(self.wvl* self.fcurv)) ))

        source = source * self.MakePupil(self.DTx)

        return source

    def BeamRider(self):
        """
        Create a beam rider defined by the input quadrant to the simulation parameter. The beam rider is a collimated
        gaussian within a region of the input diameter plane.

        Essentially, I am creating a shifted gaussian into the top, bottom, right and left hand sides of the source
        aperture by a distance of DTx/2
        """
        if self.beam_rider_quadrant == 1: # top
            source = np.exp(-(self.x1 ** 2 + (self.y1 + self.DTx/2)** 2.0) / self.w0 ** 2)
            self.beam_rider_center = [0, self.DTx/2 ]
            pupil = self.MakePupil(self.DTx/4)
            shift_pupil = scipy.ndimage.shift(pupil, [ -int(self.DTx/2/self.dx), 0 ])
            source = source * shift_pupil

        elif self.beam_rider_quadrant == 2: # right
            source = np.exp(-(self.y1 ** 2 + (self.x1 - self.DTx/2)** 2.0) / self.w0 ** 2)
            self.beam_rider_center = [self.DTx/2, 0]
            pupil = self.MakePupil(self.DTx / 4)
            shift_pupil = scipy.ndimage.shift(pupil, [ 0, int(self.DTx/2/self.dx)])
            source = source * shift_pupil

        elif self.beam_rider_quadrant == 3: # bottom
            source = np.exp(-(self.x1 ** 2 + (self.y1 - self.DTx/2)** 2.0) / self.w0 ** 2)
            self.beam_rider_center = [0, -self.DTx/2 ]
            pupil = self.MakePupil(self.DTx / 4)
            shift_pupil = scipy.ndimage.shift(pupil, [int(self.DTx/2/self.dx), 0])
            source = source * shift_pupil

        elif self.beam_rider_quadrant == 'all': # for ppt presentation

            pupil = self.MakePupil(self.DTx / 4)

            source_left = np.exp(-(self.y1 ** 2 + (self.x1 + self.DTx/2)** 2.0) / self.w0 ** 2)
            shift_pupil_left = scipy.ndimage.shift(pupil, [0, -int(self.DTx/2/self.dx) ])
            source_left = source_left * shift_pupil_left

            source_bottom = np.exp(-(self.x1 ** 2 + (self.y1 - self.DTx/2)** 2.0) / self.w0 ** 2)
            shift_pupil_bottom = scipy.ndimage.shift(pupil, [int(self.DTx/2/self.dx), 0])
            source_bottom = source_bottom * shift_pupil_bottom

            source_right = np.exp(-(self.y1 ** 2 + (self.x1 - self.DTx/2)** 2.0) / self.w0 ** 2)
            shift_pupil_right = scipy.ndimage.shift(pupil, [ 0, int(self.DTx/2/self.dx)])
            source_right = source_right * shift_pupil_right

            source_top = np.exp(-(self.x1 ** 2 + (self.y1 + self.DTx/2)** 2.0) / self.w0 ** 2)
            shift_pupil_top = scipy.ndimage.shift(pupil, [ -int(self.DTx/2/self.dx), 0 ])
            source_top = source_top * shift_pupil_top

            source = source_bottom + source_left + source_right + source_top

            fig, ax = plt.subplots(1)
            ax.imshow(source, extent=([-self.dx*self.N/2., self.dx*self.N/2.,
                                              -self.dx * self.N / 2., self.dx * self.N / 2.]))
            DTX_circle = Circle((0,0),
                                 self.DTx/2,
                                 facecolor='none',
                                 edgecolor='white',
                                 fill=False,
                                 linestyle='-')
            ax.add_patch(DTX_circle)
            plt.show()


        else: # left
            source = np.exp(-(self.y1 ** 2 + (self.x1 + self.DTx/2)** 2.0) / self.w0 ** 2)
            self.beam_rider_center = [-self.DTx/2, 0]
            pupil = self.MakePupil(self.DTx / 4)
            shift_pupil = scipy.ndimage.shift(pupil, [0, -int(self.DTx/2/self.dx) ])
            source = source * shift_pupil


        self.beam_rider_radius = self.DTx/4

        return source

    def PlaneSource(self):
        # Uniform plane wave
        plane = np.ones([self.N, self.N])

        return plane

    def PointSource(self):
        # Schmidt Point Source
        DROI = 4.0 * self.DRx  # Observation plane region [m]
        D1 = self.wvl * self.PropDist / DROI  # Central Lobe width [m]
        R = self.PropDist  # Radius of curvature at wavefront [m
        temp = np.exp(-1j * self.k / (2 * R) * (self.r1 ** 2)) / (D1 ** 2)
        pt = temp * np.sinc((self.x1 / D1)) * np.sinc((self.y1 / D1)) * np.exp(-(self.r1 / (4.0 * D1)) ** 2)

        return pt

    def FlattePointSource(self):

        fpt = np.exp(-(self.r1 ** 2) / (10 * (self.dx ** 2))) \
              * np.cos(-(self.r1 ** 2) / (10 * (self.dx ** 2)))


        return fpt

    def CollimatedGaussian(self):

        source = np.exp(-(self.r1 ** 2 / self.w0 ** 2))

        source = source * self.MakePupil(self.DTx)

        # Debug
        # plt.imshow(np.abs(source)**2.0)
        # plt.show()

        # Source return
        return source

    def MakeSGB(self):

        # Construction of Super Gaussian Boundary
        rad = self.r1 * (self.N);
        w = 0.55 * self.N
        sg = np.exp(- ((rad / w) ** 16.0))

        return sg

    def MakePupil(self, D_eval):
        # Target pupil creation
        boundary1 = -(self.SideLen / 2)  # sets negative coord of sidelength
        boundary2 = self.SideLen / 2  # sets positive coord of sidelength

        A = np.linspace(boundary1, boundary2, self.N)  # creates a series of numbers evenly spaced between
        # positive and negative boundary
        A = np.array([A] * self.N)  # horizontal distance map created

        base = np.linspace(boundary1, boundary2, self.N)  # creates another linspace of numbers
        set_ones = np.ones(self.N)  # builds array of length N filled with ones
        B = np.array([set_ones] * self.N)
        for i in range(0, len(base)):
            B[i] = B[i] * base[i]  # vertical distance map created

        A = A.reshape(self.N, self.N)
        B = B.reshape(self.N, self.N)  # arrays reshaped into matrices

        x_coord = A ** 2
        y_coord = B ** 2

        rad_dist = np.sqrt(x_coord + y_coord)  # now radial distance has been defined

        mask = []
        for row in rad_dist:
            for val in row:
                if val < D_eval:
                    mask.append(1.0)
                elif val > D_eval:
                    mask.append(0.0)
                elif val == D_eval:
                    mask.append(0.5)
        mask = np.array([mask])
        mask = mask.reshape(self.N, self.N)  # mask created and reshaped into a matrix

        return mask  # returns the pupil mask as the whole function's output

    def PhaseScreen(self,
                    delta,
                    N_override = None):
        """
        A phase screen sampling that accounts for varying sampling at the propagation planes of interest. You can create
        a larger phase screen while maintaining the sampling by using the N_override to set the N value to a larger
        value than originally derived.

        Note: the original wavepy had a bug where the same sampling was used at all phase screen planes. This function
              corrects for this by accepting a sampling parameter in meters

        :param delta: the sampling at the current propagation plane
        :param N_override: if not None, then the screen will be a different sized N to account for phase screen wind
                            evolution being affected by periodicity
        :return: a phase screen that is randomly drawn using random coefficients
        """


        # Constants used in this code
        b = self.aniso
        c = 1.0
        thetar = (pi / 180.0) * self.theta
        na = self.alpha / 6.0  # Normalized alpha value
        Bnum = gamma(na / 2.0)
        Bdenom = 2.0 ** (2.0 - na) * pi * na * gamma(-na / 2.0)
        # c1 Striblings Consistency parameter. Evaluates to 6.88 in Kolmogorov turb.
        cone = (2.0 * (8.0 / (na - 2.0) * gamma(2.0 / (na - 2.0))) ** ((na - 2.0) / 2.0))
        # Charnotskii/Bos generalized phase consistency parameter
        Bfac = (2.0 * pi) ** (2.0 - na) * (Bnum / Bdenom)
        a = gamma(na - 1.0) * cos(na * pi / 2.0) / (4.0 * pi ** 2.0)
        # Toselli's inner-scale intertial range consistency parameter
        c_a = (gamma(0.5 * (5.0 - na)) * a * 2.0 * pi / 3.0) ** (1.0 / (na - 5.0))
        fm = c_a / self.l0  # Inner scale frequency(1/m)
        # Set up parameters for Kolmogorov PSD
        nae = 22 / 6.0  # Normalized alpha value
        Bnume = gamma(nae / 2.0)
        Bdenome = 2.0 ** (2.0 - nae) * pi * nae * gamma(-nae / 2.0)
        conee = (2.0 * (8.0 / (nae - 2.0) * gamma(2.0 / (nae - 2.0))) ** ((nae - 2.0) / 2.0))
        Bface = (2.0 * pi) ** (2.0 - nae) * (Bnume / Bdenome)
        ae = gamma(nae - 1.0) * cos(nae * pi / 2.0) / (4.0 * pi ** 2.0)
        c_ae = (gamma(0.5 * (5.0 - nae)) * ae * 2.0 * pi / 3.0) ** (1.0 / (nae - 5.0))
        fme = c_ae / self.l0  # Inner scale frequency(1/m)
        f0 = 1.0 / self.L0  # Outer scale frequency

        # If no override, then just make a phase screen of the same size as original plane
        if N_override is None:

            # Derive the frequency sampling at the current plane
            del_f = 1.0 / (self.N * delta)  # Frequency grid spacing(1/m)

            # The zero-frequency center of the phase screen
            cen = np.floor(self.N / 2)

            # Create frequency sample grid
            fx = np.arange(-self.N / 2.0, self.N / 2.0) * del_f
            fx, fy = np.meshgrid(fx, -1 * fx)

            # Apply affine transform
            tx = fx * cos(thetar) + fy * sin(thetar)
            ty = -1.0 * fx * sin(thetar) + fy * cos(thetar)

            # Scalar frequency grid
            f = np.sqrt((tx ** 2.0) / (b ** 2.0) + (ty ** 2.0) / (c ** 2.0))

            # Sample Turbulence PSD
            PSD_phi = (cone * Bfac * ((b * c) ** (-na / 2.0)) * (self.r0scrn ** (2.0 - na)) * np.exp(-(f / fm) ** 2.0) \
                       / ((f ** 2.0 + f0 ** 2.0) ** (na / 2.0)))

            tot_NOK = np.sum(PSD_phi)

            # Kolmogorov equivalent and enforce isotropy
            PSD_phie = (conee * Bface * (self.r0scrn ** (2.0 - nae)) * np.exp(-(f / fme) ** 2.0) \
                        / ((f ** 2.0 + f0 ** 2.0) ** (nae / 2.0)))

            tot_OK = np.sum(PSD_phie)

            PSD_phi = (tot_OK / tot_NOK) * PSD_phi

            # PSD_phi = cone*Bfac* (r0**(2-na)) * f**(-na/2)  # Kolmogorov PSD
            PSD_phi[np.int(cen), np.int(cen)] = 0.0

            # Create a random field that is circular complex Guassian
            cn = (np.random.randn(self.N, self.N) + 1j * np.random.randn(self.N, self.N))

            # Filter by turbulence PSD
            cn = cn * np.sqrt(PSD_phi) * del_f

            # Inverse FFT
            phz_temp = np.fft.ifft2(np.fft.fftshift(cn)) * ((self.N) ** 2)

            # Phase screens
            phz1 = np.real(phz_temp)

        else:

            # Derive the frequency sampling at the current plane
            del_f = 1.0 / (N_override * delta)  # Frequency grid spacing(1/m)

            # Debug:
            print("Min/Max frequencies original grid: ")
            print(str(1/(self.N*delta)) + ", " + str(1/(2*delta)))
            print("Min/Max frequencies new grid: ")
            print(str(del_f) + ", " + str(1/(2*delta)))
            print()

            cen = np.floor(N_override / 2)

            # Create a larger frequency grid than self.N by using the override parameter
            fx = np.arange(-N_override / 2.0, N_override/ 2.0) * del_f
            fx, fy = np.meshgrid(fx, -1 * fx)

            # Apply affine transform
            tx = fx * cos(thetar) + fy * sin(thetar)
            ty = -1.0 * fx * sin(thetar) + fy * cos(thetar)

            # Scalar frequency grid
            f = np.sqrt((tx ** 2.0) / (b ** 2.0) + (ty ** 2.0) / (c ** 2.0))

            # Sample Turbulence PSD
            PSD_phi = (cone * Bfac * ((b * c) ** (-na / 2.0)) * (self.r0scrn ** (2.0 - na)) * np.exp(-(f / fm) ** 2.0) \
                       / ((f ** 2.0 + f0 ** 2.0) ** (na / 2.0)))

            tot_NOK = np.sum(PSD_phi)

            # Kolmogorov equivalent and enforce isotropy
            # Sample Turbulence PSD
            PSD_phie = (conee * Bface * (self.r0scrn ** (2.0 - nae)) * np.exp(-(f / fme) ** 2.0) \
                        / ((f ** 2.0 + f0 ** 2.0) ** (nae / 2.0)))

            tot_OK = np.sum(PSD_phie)

            PSD_phi = (tot_OK / tot_NOK) * PSD_phi

            # PSD_phi = cone*Bfac* (r0**(2-na)) * f**(-na/2)  # Kolmogorov PSD
            PSD_phi[np.int(cen), np.int(cen)] = 0.0

            # Mask out values that are outside of the power spectrum under consideration
            #mask = (f < (1 / self.L0) ** 2) | (f > fme ** 2)
            #PSD_phi[mask] = 0.0


            # Create a random field that is circular complex Guassian
            cn = (np.random.randn(N_override,N_override) + 1j * np.random.randn(N_override,N_override))

            # Filter by turbulence PSD
            cn = cn * np.sqrt(PSD_phi) * del_f

            # Inverse FFT
            phz_temp = np.fft.ifft2(np.fft.fftshift(cn)) * ((N_override) ** 2)

            # Phase screens
            phz1 = np.real(phz_temp)

        return phz1

    def SubHarmonicComp(self, nsub, delta, N_override = None):
        """
        Sub-harmonic phase screen generation that accounts for the expansion of the beam at different propagation
        distances. You can use the N_override parameter to create a larger phase screen than self.N, which is necessary
        to create a time-evolved phase screen that is not affected by periodicity concerns.

        :param nsub: the number of sub-harmonics
        :param delta: the sampling in terms of meters at the current propagation plane
        :param N_override: override the number of samples per side while maintaining the same sample spacing
        :return:
        """

        if N_override is None:
            # Derive the sidelength and sampling frequency for the current plane
            SideLen_dz = self.N * delta
            dq = 1 / SideLen_dz
            na = self.alpha / 6.0

            Bnum = gamma(na / 2.0)
            Bdenom = (2 ** (2 - na)) * pi * na * gamma(-na / 2)
            Bfac = (2 * pi) ** (2 - na) * (Bnum / Bdenom)

            # c1 Striblings Consistency parameter. Evaluates to 6.88 in Kolmogorov turb.
            cone = (2 * (8 / (na - 2) * gamma(2 / (na - 2))) ** ((na - 2) / 2))

            # Anisotropy factors
            b = self.aniso
            c = 1
            f0 = 1 / self.L0
            lof_phz = np.zeros((self.N, self.N))

            temp_m = np.linspace(-0.5, 0.5, self.N)

            m_indices, n_indices = np.meshgrid(temp_m, -1 * np.transpose(temp_m))

            temp_mp = np.linspace(-2.5, 2.5, 6)

            m_prime_indices, n_prime_indices = np.meshgrid(temp_mp, -1 * np.transpose(temp_mp))

            for Np in range(1, nsub + 1):

                temp_phz = np.zeros((self.N, self.N))
                # Subharmonic frequency
                dqp = dq / (3.0 ** Np)
                # Set samples

                f_x = 3 ** (-Np) * m_prime_indices * dq
                f_y = 3 ** (-Np) * n_prime_indices * dq

                f = np.sqrt((f_x ** 2) / (b ** 2) + (f_y ** 2) / (c ** 2))
                # Sample PSD
                PSD_fi = cone * Bfac * ((b * c) ** (-na / 2)) * (self.r0scrn) ** (2 - na) * (f ** 2 + f0 ** 2) ** (-na / 2)

                # Generate normal circ complex RV
                w = np.random.randn(6, 6) + 1j * np.random.randn(6, 6)
                # Covariances
                cv = w * np.sqrt(PSD_fi) * dqp
                # Sum over subharmonic components
                temp_shape = np.shape(cv)
                for n in range(0, temp_shape[0]):
                    for m in range(0, temp_shape[1]):
                        indexMap = (m_prime_indices[n][m] * m_indices +
                                    n_prime_indices[n][m] * n_indices)

                        temp_phz = temp_phz + cv[m][n] * np.exp(1j * 2 * pi * (3 ** (-Np)) * indexMap)

                # Accumulate components to phase screen
                lof_phz = lof_phz + temp_phz

            lof_phz = np.real(lof_phz) - np.mean(np.real(lof_phz))
        else:

            # Derive the sidelength and sampling frequency for the current plane
            SideLen_dz = self.N * delta
            dq = 1 / SideLen_dz
            na = self.alpha / 6.0

            Bnum = gamma(na / 2.0)
            Bdenom = (2 ** (2 - na)) * pi * na * gamma(-na / 2)
            Bfac = (2 * pi) ** (2 - na) * (Bnum / Bdenom)

            # c1 Striblings Consistency parameter. Evaluates to 6.88 in Kolmogorov turb.
            cone = (2 * (8 / (na - 2) * gamma(2 / (na - 2))) ** ((na - 2) / 2))

            # Anisotropy factors
            b = self.aniso
            c = 1
            f0 = 1 / self.L0
            fme = 5.92 / self.l0 / (2*np.pi)
            lof_phz = np.zeros((N_override, N_override))

            temp_m = np.linspace(-0.5, 0.5,N_override)

            m_indices, n_indices = np.meshgrid(temp_m, -1 * np.transpose(temp_m))

            temp_mp = np.linspace(-2.5, 2.5, 6)

            m_prime_indices, n_prime_indices = np.meshgrid(temp_mp, -1 * np.transpose(temp_mp))

            for Np in range(1, nsub + 1):

                temp_phz = np.zeros((N_override, N_override))
                # Subharmonic frequency
                dqp = dq / (3.0 ** Np)
                # Set samples

                f_x = 3 ** (-Np) * m_prime_indices * dq
                f_y = 3 ** (-Np) * n_prime_indices * dq

                f = np.sqrt((f_x ** 2) / (b ** 2) + (f_y ** 2) / (c ** 2))
                # Sample PSD
                PSD_fi = cone * Bfac * ((b * c) ** (-na / 2)) * (self.r0scrn) ** (2 - na) * (f ** 2 + f0 ** 2) ** (
                            -na / 2)


                # Mask out the portions of the power spectrum that are outside of the frequency bounds
                mask = (f < (1 / self.L0) ** 2) | (f > fme ** 2)
                PSD_fi[mask] = 0


                # Generate normal circ complex RV
                w = np.random.randn(6, 6) + 1j * np.random.randn(6, 6)
                # Covariances
                cv = w * np.sqrt(PSD_fi) * dqp
                # Sum over subharmonic components
                temp_shape = np.shape(cv)
                for n in range(0, temp_shape[0]):
                    for m in range(0, temp_shape[1]):
                        indexMap = (m_prime_indices[n][m] * m_indices +
                                    n_prime_indices[n][m] * n_indices)

                        temp_phz = temp_phz + cv[m][n] * np.exp(1j * 2 * pi * (3 ** (-Np)) * indexMap)

                # Accumulate components to phase screen
                lof_phz = lof_phz + temp_phz

            lof_phz = np.real(lof_phz) - np.mean(np.real(lof_phz))


        return lof_phz


    def VacuumProp(self):
        # Vacuum propagation (included for source valiation)
        sg = self.MakeSGB()  # Generates SGB

        SamplingRatio = self.SamplingRatioBetweenScreen

        a = int(self.N / 2)

        nx, ny = np.meshgrid(range(-a, a), range(-a, a))

        # Initial Propagation from source plane to first screen location
        P0 = np.exp(1j * (self.k / (2 * self.dzProps[0])) * (self.r1 ** 2) * (1 - SamplingRatio[0]))

        Uin = P0 * self.Source

        for pcount in range(1, len(self.PropLocs) - 2):
            UinSpec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Uin)))

            # Set spatial frequencies at propagation plane
            deltaf = 1 / (self.N * self.PropSampling[pcount])
            fX = nx * deltaf
            fY = ny * deltaf
            fsq = fX ** 2 + fY ** 2

            # Quadratic Phase Factor
            QuadPhaseFac = np.exp(-1j * np.pi * self.wvl * self.dzProps[pcount] \
                                  * SamplingRatio[pcount] * fsq)

            Uin = np.fft.ifftshift(np.fft.ifft2( \
                np.fft.ifftshift(UinSpec * QuadPhaseFac)))

            Uin = Uin * sg

        PF = np.exp(1j * (self.k / (2 * self.dzProps[-1])) * (self.rR ** 2) * (SamplingRatio[-1]))

        Uout = PF * Uin

        return Uout


    def SplitStepProp(self, Uin, PhaseScreenStack):
        # Propagation/Fresnel Diffraction Integral
        sg = self.MakeSGB()  # Generates SGB

        SamplingRatio = self.SamplingRatioBetweenScreen

        a = int(self.N / 2)

        nx, ny = np.meshgrid(range(-a, a), range(-a, a))

        # Initial Propagation from source plane to first screen location
        P0 = np.exp(1j * (self.k / (2 * self.dzProps[0])) * (self.r1 ** 2) * (1 - SamplingRatio[0]))

        Uin = P0 * self.Source * np.exp(1j * PhaseScreenStack[:, :, 0])

        for pcount in range(1, len(self.PropLocs) - 2):
            UinSpec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Uin)))

            # Set spatial frequencies at propagation plane
            deltaf = 1 / (self.N * self.PropSampling[pcount])
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

    def TurbSim(self):
        """
        Perform a turbulence simulation that accounts for the changing sampling planes over the course of the propagation
        through the atmosphere

        :return:
        """
        # initialize phase screen array
        phz = np.zeros(shape=(self.N, self.N, self.NumScr))
        phz_lo = np.zeros(shape=(self.N, self.N, self.NumScr))
        phz_hi = np.zeros(shape=(self.N, self.N, self.NumScr))

        for idxscr in range(0, self.NumScr, 1):

            # Retrieve the current sampling at the current propagation distance
            deltax_dz = self.PropSampling[idxscr]

            phz_hi[:, :, idxscr] = self.PhaseScreen(deltax_dz)
            # FFT-based phase screens

            phz_lo[:, :, idxscr] = self.SubHarmonicComp(self.NumSubHarmonics,  deltax_dz)
            # sub harmonics
            phz[:, :, idxscr] = self.loon * phz_lo[:, :, idxscr] + phz_hi[:, :, idxscr]
            # subharmonic compensated phase screens

        # Simulating propagation
        self.Output = self.SplitStepProp(self.Source, np.exp(1j * phz))

    def SetCn2Rytov(self, UserRytov):
        # Change rytov number and variance to user specified value
        self.rytovNum = UserRytov
        self.rytov = self.rytovNum ** 2

        rytov_denom = 1.23 * (self.k) ** (7.0 / 6.0) * (self.PropDist) ** (11.0 / 6.0)

        # Find Cn2
        self.Cn2 = self.rytov / rytov_denom

        log_ampl_var = np.zeros(100)
        for d in range(0, self.PropDist, self.PropDist//100):
            log_ampl_var[d] = 0.3075 * ((self.k) ** 2) * ((d) ** (11.0 / 6.0)) * self.Cn2
        self.log_ampl_var = log_ampl_var
        # Set derived values
        self.r0 = (0.423 * (self.k) ** 2 * self.Cn2 * self.PropDist) ** (-3.0 / 5.0)
        self.r0scrn = (0.423 * ((self.k) ** 2) * self.Cn2 * (self.PropDist / self.NumScr)) ** (-3.0 / 5.0)
        #self.log_ampl_var = 0.3075 * ((self.k) ** 2) * ((self.PropDist) ** (11.0 / 6.0)) * self.Cn2
        self.phase_var = 0.78 * (self.Cn2) * (self.k ** 2) * self.PropDist * (self.L0 ** (-5.0 / 3.0))
        self.rho_0 = (1.46 * self.Cn2 * self.k ** 2 * self.PropDist) ** (-5.0 / 3.0)

    def EvalSI(self):

        temp_s = (np.abs(self.Output) ** 2) * self.makePupil(self.DRx)
        temp_s = temp_s.ravel()[np.flatnonzero(temp_s)]
        s_i = (np.mean(temp_s ** 2) / (np.mean(temp_s) ** 2)) - 1

        return s_i

    def StructFunc(self, ph):

        # Define mask construction
        mask = self.MakePupil(self.SideLen / 4)
        delta = self.SideLen / self.N

        N_size = np.shape(ph)  # Make sure to reference 0th element later
        ph = ph * mask

        P = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ph))) * (delta ** 2)
        S = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ph ** 2))) * (delta ** 2)
        W = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mask))) * (delta ** 2)
        delta_f = 1 / (N_size[0] * delta)

        fft_size_a = np.shape(W * np.conjugate(W))
        w2 = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(W * np.conjugate(W)))) * ((fft_size_a[0] * delta_f) ** 2)

        fft_size_b = np.shape(np.real(S * np.conjugate(W)) - np.abs(P) ** 2)
        D = 2 * ((np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(np.real(S * np.conjugate(W)) - np.abs(P) ** 2)))) * (
                    (fft_size_b[0] * delta_f) ** 2))

        D = D / w2

        D = np.abs(D) * mask

        return D


    def Validate(self, nruns):

        self.r0scrn = 0.5 * self.SideLen / 20
        self.N = 512

        phz_FT = np.zeros((self.N, self.N))
        phz_FT_temp = phz_FT
        phz_SH = np.zeros((self.N, self.N))
        phz_SH_temp = phz_SH

        # Generating multiple phase screens
        for j in range(0, nruns):
            phz_FT_temp = self.PhaseScreen()
            # using phase screens from ^ so that time isn't wasted generating
            # screens for the SubHarmonic case
            phz_SH_temp = self.SubHarmonicComp(1) + phz_FT_temp

            phz_FT_temp = self.StructFunc(phz_FT_temp)
            phz_SH_temp = self.StructFunc(phz_SH_temp)
            phz_FT = phz_FT + phz_FT_temp
            phz_SH = phz_SH + phz_SH_temp

        # Averaging the runs and correct bin size
        phz_FT = phz_FT / nruns
        phz_SH = phz_SH / nruns
        m, n = np.shape(phz_FT)
        centerX = round(m / 2) + 1

        phz_FT_disp = np.ones(self.N / 2)
        phz_FT_disp = phz_FT[:, centerX]
        phz_SH_disp = np.ones(self.N / 2)
        phz_SH_disp = phz_SH[:, centerX]

        phz_FT_disp = phz_FT_disp[0:(self.N / 2)]
        phz_FT_disp = phz_FT_disp[::-1]
        phz_SH_disp = phz_SH_disp[0:(self.N / 2)]
        phz_SH_disp = phz_SH_disp[::-1]

        # array of values for normalized r to plot x-axis
        cent_dist = np.zeros(self.N / 2)
        r_size = (0.5 * self.SideLen) / (0.5 * self.N)
        for i in range(0, (self.N / 2)):
            cent_dist[i] = (i * r_size) / (self.r0scrn)

        # Defining theoretical equation
        theory_val = np.zeros(self.N / 2)
        theory_val = 6.88 * (cent_dist) ** (5.0 / 3.0)

        # Plotting 3 options, with blue=theory, green=FT, and red=SH in current order
        plt.plot(cent_dist, theory_val)
        plt.plot(cent_dist, phz_FT_disp)
        plt.plot(cent_dist, phz_SH_disp)
        plt.xlim((0, 10))
        plt.ylim((0, 400))


    def evolve_phase_screen_stack(self,
                                  output_dir_stub,
                                  timeintervals,
                                  delta_step=1,
                                  alpha=0.99,
                                  vx_list = None, vy_list = None,
                                  wind_upper_bound = 20):
        """
        Note: For longer simulations than time for the phase screen to cross the screen due to wind, this method is a
         bad choice due to periodicity effects

        Generate a stack of phase screens that are evolved over time according to two major user parameters:
            (1) delta_step - the number of pixels the wind will cross between each update
            (2) vx/y_list - lists giving the wind speed at each layer in the x/y directions

        This code creates wider phase screens than self.N and then crops them to self.N. This is to avoid the
        periodicity issues that come with DFT's.

        :param output_dir_stub: the output directory where the phase screens will be saved to avoid memory issues
        :param timeintervals: the number of time intervals over which you will be evolving the screen
        :param delta_step: the number of pixels the phase screen at the receiver will drift between updates
                          Enter 1 for Nyquist sampling at the receiver plane.
        :param alpha: the memory scalar of the phase screen update. Set to zero for frozen flow
        :param vx: the velocity in the x direction at each phase screen. By default, it will be randomly initialized
        :param vy: the velocity in the x direction. By default, it will be randomly initialized
        :param wind_upper_bound: the upper bound in meters/second for the wind initialization
        :return:
        """

        # Assign values to the variables for the phase screens that will be saved
        self.number_evol_tint = timeintervals
        self.evol_dir_stub = output_dir_stub

        # Assign values for the velocities of the wind at each phase screen
        if vx_list is not None:
            self.vx_list = vx_list
        if vy_list is not None:
            self.vy_list = vy_list

        # If either phase screen list is not initialized, then randomly initialize based on direction of the wind
        if self.vy_list is None or self.vx_list is None:
            self.vy_list = []
            for ix in range(self.NumScr):
                rand_speed = np.random.random() * wind_upper_bound
                rand_direction = np.random.random() * 2*np.pi
                self.vy_list.append(rand_speed * np.sin(rand_direction))
                self.vx_list.append(rand_speed * np.cos(rand_direction))


        # Figure out how much padding will need to be added to the phase screens in order to account for DFT periodicity
        max_velocity = np.max(np.sqrt( np.array(self.vx_list)**2.0 + np.array(self.vy_list)**2.0))
        N_pad = self.N + (delta_step * timeintervals)
        Crop_bottom = int((delta_step * timeintervals)/2.0)
        Crop_top = Crop_bottom + self.N

        # Calculate the update time period based on the receiver sampling plane
        T_period = delta_step * self.Rdx / max_velocity / 2


        # Update a placeholder for the update time and the alpha so you can later output them
        self.update_time = T_period
        self.alpha_memory = alpha


        # Initialize the time-zero phase screen arrays
        phz_0 = np.zeros(shape=(N_pad, N_pad, self.NumScr))
        phz_lo_0 = np.zeros(shape=(N_pad, N_pad, self.NumScr))
        phz_hi_0 = np.zeros(shape=(N_pad, N_pad, self.NumScr))

        for idxscr in range(0, self.NumScr, 1):
            delta = self.PropSampling[idxscr]

            # FFT-based phase screens for the high-frequency components of the power spectrum
            phz_hi_0[:, :, idxscr] = self.PhaseScreen(delta, N_override=N_pad)

            # FFT-based sub-harmonic screens for the low-frequency components of the power spectrum
            phz_lo_0[:, :, idxscr] = self.SubHarmonicComp(self.NumSubHarmonics, delta, N_override=N_pad)

            phz_0[:, :, idxscr] = self.loon * phz_lo_0[:, :, idxscr] + phz_hi_0[:, :, idxscr]
            # subharmonic compensated phase screens

            # Save a cropped version of the screen
            save_screen = phz_0[:, :, idxscr][Crop_bottom:Crop_top, Crop_bottom:Crop_top]

            # Save the time zero screens as images and as .npy files to be loaded in later
            output_screen_npy_fname = output_dir_stub + "_scr" + str(idxscr) + "_t0000.npy"
            np.save(output_screen_npy_fname, save_screen)

            output_screen_png_fname = output_dir_stub + "_scr" + str(idxscr) + "_t0000.PNG"
            matplotlib.image.imsave(output_screen_png_fname, save_screen)

        # Now for each timestep in the specified number of time intervals, compute the phase screen at the next time
        phz_t_minus_1 = phz_0
        for t_int in range(1, timeintervals + 1):
            for idxscr in range(0, self.NumScr, 1):

                # Derive the frequency sampling grid that has no FFT-shift applied
                delta = self.PropSampling[idxscr]
                b = self.aniso
                c = 1.0
                thetar = (pi / 180.0) * self.theta
                del_f = 1.0 / (self.N * delta)  # Frequency grid spacing(1/m)
                na = self.alpha / 6.0  # Normalized alpha value
                a = gamma(na - 1.0) * cos(na * pi / 2.0) / (4.0 * pi ** 2.0)
                nae = 22 / 6.0  # Normalized alpha value
                ae = gamma(nae - 1.0) * cos(nae * pi / 2.0) / (4.0 * pi ** 2.0)
                c_ae = (gamma(0.5 * (5.0 - nae)) * ae * 2.0 * pi / 3.0) ** (1.0 / (nae - 5.0))
                fme = c_ae / self.l0  # Inner scale frequency(1/m)
                fx = np.arange(-N_pad/2.0, N_pad / 2.0) * del_f
                fx, fy = np.meshgrid(fx, -1 * fx)
                tx = fx * cos(thetar) + fy * sin(thetar)
                ty = -1.0 * fx * sin(thetar) + fy * cos(thetar)
                f = np.sqrt((tx ** 2.0) / (b ** 2.0) + (ty ** 2.0) / (c ** 2.0))

                # Get the wind velocity at the current time step
                vx_screen = self.vx_list[idxscr]
                vy_screen = self.vy_list[idxscr]

                # Derive the shift phase component that applies a phase ramp to denote an evolving phase screen
                theta_wind = -2. * np.pi * T_period * (fx * vx_screen + fy * vy_screen)

                # Be sure to apply an FFTSHIFT to line up the zero frequency terms of the alpha coefficient with the
                # phase screens at previous time steps
                alpha_coeff = sf.fftshift(alpha * (np.cos(theta_wind) + 1j * np.sin(theta_wind)))

                # Derive the memory of the wind that is being blown across the screen
                FFT_phz_t = sf.fft2(phz_t_minus_1[:, :, idxscr])
                wind_blown_memory = alpha_coeff * FFT_phz_t

                # Debug:
                # plt.imshow(np.abs(FFT_phz_t).real)
                # plt.show()


                # If the alpha parameter is 1, use frozen flow. Otherwise, randomly draw a high-frequency screen
                if alpha == 1:
                    new_phase = 0
                else:
                    new_phase = np.sqrt(1.0 - np.real(np.abs(alpha_coeff ** 2.0))) * \
                                        sf.fft2( self.PhaseScreen(delta, N_override=N_pad))

                # Derive the phase screen at the current time step by taking an inverse
                ifft2_comp = sf.ifft2(wind_blown_memory + new_phase)

                # Draw the real component for the updates in phase screens
                phz_t_minus_1[:, :, idxscr] = ifft2_comp.real

                # Crop out the center of NxN in order to get the
                save_screen = phz_t_minus_1[:, :, idxscr][Crop_bottom:Crop_top, Crop_bottom:Crop_top ]

                # Save the phase screen as an image and a numpy array to later load up for turbulence simulations and
                # analysis of statistics of simulations
                output_screen_npy_fname = output_dir_stub + "_scr" + str(idxscr) + "_t" + "{:04d}".format(t_int) + ".npy"
                np.save(output_screen_npy_fname, save_screen)
                output_screen_png_fname = output_dir_stub + "_scr" + str(idxscr) + "_t" + "{:04d}".format(t_int) + ".PNG"
                matplotlib.image.imsave(output_screen_png_fname, save_screen)


    def periodicity_corrected_evolve_phase_screen_stack(self,
                                                       output_dir_stub,
                                                       sim_time,
                                                       sampling_interval_hz,
                                                       alpha = 0.99,
                                                       power_retained = 0.01,
                                                       vx_list = None, vy_list = None,
                                                       wind_upper_bound = 20):

        # If the number of desired time intervals exceeds the maximum time intervals, then replace
        sim_timeintervals = int(sim_time * sampling_interval_hz)
        max_timeintervals = int(np.log(power_retained) / np.log(alpha))
        if sim_timeintervals > max_timeintervals:
            sim_timeintervals = max_timeintervals

        print("Number of Time intervals being performed: ", sim_timeintervals)

        # Calculate the sampling time period
        T_period = 1 / sampling_interval_hz

        # Assign values for the velocities of the wind at each phase screen
        if vx_list is not None:
            self.vx_list = vx_list
        if vy_list is not None:
            self.vy_list = vy_list

        # If either phase screen list is not initialized, then randomly initialize based on direction of the wind
        if self.vy_list is None or self.vx_list is None:
            self.vy_list = []
            for ix in range(self.NumScr):
                rand_speed = np.random.random() * wind_upper_bound
                rand_direction = np.random.random() * 2*np.pi
                self.vy_list.append(rand_speed * np.sin(rand_direction))
                self.vx_list.append(rand_speed * np.cos(rand_direction))

        # Figure out how much padding will need to be added to the phase screens in order to account for DFT periodicity
        max_velocity = np.max(np.sqrt( np.array(self.vx_list)**2.0 + np.array(self.vy_list)**2.0))


        # If the time required for a screen to blow across the sample plane is less than the simulation time, use that
        # for your simulation time
        time_blow_across = self.SideLen / max_velocity

        # if time_blow_across <= sim_time:
        #     pad_factor = np.ceil(max_timeintervals / (time_blow_across * sampling_interval_hz))
        # else:
        #     pad_factor = np.ceil(max_timeintervals / sim_timeintervals)

        pad_factor = np.ceil(max_timeintervals / (time_blow_across * sampling_interval_hz))


        # Calculate the padding to achieve the desired power retention to avoid periodicity effects
        N_pad = int(self.N * pad_factor + self.N)

        print(N_pad)

        Crop_bottom = int(N_pad / 2) - int(self.N/2)
        Crop_top = Crop_bottom + self.N


        # Update a placeholder for the update time and the alpha so you can later output them
        self.update_time = T_period
        self.alpha_memory = alpha

        # Assign values to the variables for the phase screens that will be saved
        self.number_evol_tint = sim_timeintervals
        self.evol_dir_stub = output_dir_stub


        # Initialize the time-zero phase screen arrays
        phz_0 = np.zeros(shape=(N_pad, N_pad, self.NumScr))
        phz_lo_0 = np.zeros(shape=(N_pad, N_pad, self.NumScr))
        phz_hi_0 = np.zeros(shape=(N_pad, N_pad, self.NumScr))

        for idxscr in range(0, self.NumScr, 1):
            delta = self.PropSampling[idxscr]

            # FFT-based phase screens for the high-frequency components of the power spectrum
            phz_hi_0[:, :, idxscr] = self.PhaseScreen(delta, N_override=N_pad)

            # FFT-based sub-harmonic screens for the low-frequency components of the power spectrum
            # phz_lo_0[:, :, idxscr] = self.SubHarmonicComp(self.NumSubHarmonics, delta, N_override=N_pad)
            # phz_0[:, :, idxscr] = self.loon * phz_lo_0[:, :, idxscr] + phz_hi_0[:, :, idxscr]

            # Ignore subharmonics because we are clipping the screen
            phz_lo_0 = 0
            phz_0[:, :, idxscr] =  phz_hi_0[:, :, idxscr]


            # Save a cropped version of the screen
            save_screen = phz_0[:, :, idxscr][Crop_bottom:Crop_top, Crop_bottom:Crop_top]

            # Save the time zero screens as images and as .npy files to be loaded in later
            output_screen_npy_fname = output_dir_stub + "_scr" + str(idxscr) + "_t0000.npy"
            np.save(output_screen_npy_fname, save_screen)

            output_screen_png_fname = output_dir_stub + "_scr" + str(idxscr) + "_t0000.PNG"
            matplotlib.image.imsave(output_screen_png_fname, save_screen)

        # Now for each timestep in the specified number of time intervals, compute the phase screen at the next time
        phz_t_minus_1 = phz_0
        for t_int in range(1, sim_timeintervals + 1):
            for idxscr in range(0, self.NumScr, 1):

                # Derive the frequency sampling grid that has no FFT-shift applied
                delta = self.PropSampling[idxscr]
                b = self.aniso
                c = 1.0
                thetar = (pi / 180.0) * self.theta
                del_f = 1.0 / (self.N * delta)  # Frequency grid spacing(1/m)
                na = self.alpha / 6.0  # Normalized alpha value
                a = gamma(na - 1.0) * cos(na * pi / 2.0) / (4.0 * pi ** 2.0)
                nae = 22 / 6.0  # Normalized alpha value
                ae = gamma(nae - 1.0) * cos(nae * pi / 2.0) / (4.0 * pi ** 2.0)
                c_ae = (gamma(0.5 * (5.0 - nae)) * ae * 2.0 * pi / 3.0) ** (1.0 / (nae - 5.0))
                fme = c_ae / self.l0  # Inner scale frequency(1/m)
                fx = np.arange(-N_pad/2.0, N_pad / 2.0) * del_f
                fx, fy = np.meshgrid(fx, -1 * fx)
                tx = fx * cos(thetar) + fy * sin(thetar)
                ty = -1.0 * fx * sin(thetar) + fy * cos(thetar)
                f = np.sqrt((tx ** 2.0) / (b ** 2.0) + (ty ** 2.0) / (c ** 2.0))

                # Get the wind velocity at the current time step
                vx_screen = self.vx_list[idxscr]
                vy_screen = self.vy_list[idxscr]

                # Derive the shift phase component that applies a phase ramp to denote an evolving phase screen
                theta_wind = -2. * np.pi * T_period * (fx * vx_screen + fy * vy_screen)

                # Be sure to apply an FFTSHIFT to line up the zero frequency terms of the alpha coefficient with the
                # phase screens at previous time steps
                alpha_coeff = sf.fftshift(alpha * (np.cos(theta_wind) + 1j * np.sin(theta_wind)))

                # Derive the memory of the wind that is being blown across the screen
                FFT_phz_t = sf.fft2(phz_t_minus_1[:, :, idxscr])
                wind_blown_memory = alpha_coeff * FFT_phz_t

                # Debug:
                # plt.imshow(np.abs(FFT_phz_t).real)
                # plt.show()


                # If the alpha parameter is 1, use frozen flow. Otherwise, randomly draw a high-frequency screen
                if alpha == 1:
                    new_phase = 0
                else:
                    new_phase = np.sqrt(1.0 - np.real(np.abs(alpha_coeff ** 2.0))) * \
                                        sf.fft2( self.PhaseScreen(delta, N_override=N_pad))

                # Derive the phase screen at the current time step by taking an inverse
                ifft2_comp = sf.ifft2(wind_blown_memory + new_phase)

                # Draw the real component for the updates in phase screens
                phz_t_minus_1[:, :, idxscr] = ifft2_comp.real

                # Crop out the center of NxN in order to get the
                save_screen = phz_t_minus_1[:, :, idxscr][Crop_bottom:Crop_top, Crop_bottom:Crop_top ]

                # Save the phase screen as an image and a numpy array to later load up for turbulence simulations and
                # analysis of statistics of simulations
                output_screen_npy_fname = output_dir_stub + "_scr" + str(idxscr) + "_t" + "{:04d}".format(t_int) + ".npy"
                np.save(output_screen_npy_fname, save_screen)
                output_screen_png_fname = output_dir_stub + "_scr" + str(idxscr) + "_t" + "{:04d}".format(t_int) + ".PNG"
                matplotlib.image.imsave(output_screen_png_fname, save_screen)


    def TurbSimEvolvingPhase(self,
                             input_phase_stack_dir = None,
                             input_number_tint = None):

        """
        Using a stack of phase screens over time, calculate the output turbulence simulation at each timestep. Also
        output a file containing the parameters used to run the code and generate the turbulence simulations

        :param input_phase_stack_dir: the directory for the phase screen stack, if there is not already one then
        :param input_number_tint: the number of time intervals in the sequence
        :return:
        """


        # Check if the user has input a directory stub to the phase screen stack
        if input_phase_stack_dir is not None and input_number_tint is not None:

            # Read in the phase screen stack from the user and run a simulation at each time interval
            for t_int in range(0, input_number_tint + 1):

                # Placeholder for the phase screen stack at the current time interval
                phz_t = np.zeros(shape=(self.N, self.N, self.NumScr))

                # Get the phase screens at the current time interval
                for idxscr in range(0, self.NumScr, 1):
                    input_screen_idxscrn = np.load(input_phase_stack_dir + "_scr" + str(idxscr) + "_t" + "{:04d}".format(t_int) + ".npy")
                    phz_t[:, :, idxscr] = input_screen_idxscrn

                # Run the simulation and save the png along with the txt file
                output_sim_png_fname = input_phase_stack_dir + "_turbsim_t" + "{:04d}".format(t_int) + ".PNG"
                turbsim_t = self.SplitStepProp(self.Source, np.exp(1j * phz_t))

                # Save the image
                # matplotlib.image.imsave(output_sim_png_fname, turbsim_t)
                fig, ax = plt.subplots(figsize=(8,6))
                ax.imshow( np.abs(turbsim_t)**2.0, extent=([-self.Rdx*self.N/2., self.Rdx*self.N/2.,
                                              -self.Rdx * self.N / 2., self.Rdx * self.N / 2.]))

                # If testing the beam rider, then create a circle to show the target point for reference
                if self.simOption == 200:
                    circle1 = Circle((self.beam_rider_center[0], self.beam_rider_center[1]),
                                         self.beam_rider_radius,
                                         facecolor = 'none',
                                         edgecolor='white',
                                         fill=False,
                                         linestyle='-')
                    ax.add_patch(circle1)


                fig.savefig(output_sim_png_fname)
                fig.close()

                output_sim_npy_fname =  input_phase_stack_dir + "_turbsim_t" + "{:04d}".format(t_int) + ".npy"
                np.save(output_sim_npy_fname, turbsim_t)

            # Write out the file containing the simulation parameters
            fout = input_phase_stack_dir + "simulation_params.txt"
            fo = open(fout, "w")
            sim_parameters = {   "Cn2": self.Cn2,
                                 "r0" : self.r0,
                                 "beam_waist": self.w0,
                                 "wavelength" : self.wvl,
                                 "N": self.N,
                                 "Source Sidelen" :self.SideLen,
                                 "Source dx": self.dx,
                                 "Receiver Rdx": self.Rdx,
                                 "L0":self.L0,
                                 "l0": self.l0,
                                 "Num Screens":self.NumScr,
                                 "Propagation Dist":self.PropDist,
                                 "Alpha Memory":self.alpha_memory,
                                 "Radius Curvature":self.fcurv,
                                 "Vx wind":self.vx_list,
                                 "Vy wind":self.vy_list,
                                 "Time Interval Sampling": self.update_time,
                                 "Timesteps": self.number_evol_tint
                                 }

            for k, v in sim_parameters.items():
                fo.write(str(k) + ': ' + str(v) + '\n')
            fo.close()


        # Check if the user has input a directory stub to the phase screen stack
        elif self.evol_dir_stub is not None and self.number_evol_tint is not None:

            # Read in the phase screen stack from the user and run a simulation at each time interval
            for t_int in range(0, self.number_evol_tint + 1):

                # Placeholder for the phase screen stack at the current time interval
                phz_t = np.zeros(shape=(self.N, self.N, self.NumScr))

                # Get the phase screens at the current time interval
                for idxscr in range(0, self.NumScr, 1):
                    input_screen_idxscrn = np.load(self.evol_dir_stub + "_scr" + str(idxscr) + "_t" + "{:04d}".format(t_int) + ".npy")
                    phz_t[:, :, idxscr] = input_screen_idxscrn

                # Run the simulation and save the png along with the txt file
                output_sim_png_fname = self.evol_dir_stub + "_turbsim_t" + "{:04d}".format(t_int) + ".PNG"
                turbsim_t = self.SplitStepProp(self.Source, np.exp(1j * phz_t))

                # Save the image
                fig, ax = plt.subplots(figsize=(8,6))
                ax.imshow( np.abs(turbsim_t)**2.0, extent=([-self.Rdx*self.N/2., self.Rdx*self.N/2.,
                                              -self.Rdx * self.N / 2., self.Rdx * self.N / 2.]))

                # If testing the beam rider, then create a circle to show the target point for reference
                if self.simOption == 200:
                    circle1 = Circle((self.beam_rider_center[0], self.beam_rider_center[1]),
                                         self.beam_rider_radius,
                                         facecolor = 'none',
                                         edgecolor='white',
                                         fill=False,
                                         linestyle='-')
                    ax.add_patch(circle1)


                fig.savefig(output_sim_png_fname)
                plt.close(fig)

                # matplotlib.image.imsave(output_sim_png_fname, np.abs(turbsim_t)**2.0)

                output_sim_npy_fname =  self.evol_dir_stub  + "_turbsim_t" + "{:04d}".format(t_int) + ".npy"
                np.save(output_sim_npy_fname, turbsim_t)


            # Write out the file containing the simulation parameters
            fout = self.evol_dir_stub + "simulation_params.txt"
            fo = open(fout, "w")
            sim_parameters = {   "Cn2": self.Cn2,
                                 "r0" : self.r0,
                                 "beam_waist": self.w0,
                                 "wavelength" : self.wvl,
                                 "N": self.N,
                                 "Source Sidelen" :self.SideLen,
                                 "Source dx": self.dx,
                                 "Receiver Rdx": self.Rdx,
                                 "L0":self.L0,
                                 "l0": self.l0,
                                 "Num Screens":self.NumScr,
                                 "Propagation Dist":self.PropDist,
                                 "Alpha Memory":self.alpha_memory,
                                 "Radius Curvature":self.fcurv,
                                 "Vx wind":self.vx_list,
                                 "Vy wind":self.vy_list,
                                 "Time Interval Sampling": self.update_time,
                                 "Timesteps": self.number_evol_tint
                                 }

            for k, v in sim_parameters.items():
                fo.write(str(k) + ': ' + str(v) + '\n')
            fo.close()


        else:

            print("Evolving phase screen generation step has not been performed. Exiting simulation...")