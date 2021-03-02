"""
Infinite Phase Screens
----------------------
An implementation of the "infinite phase screen", as deduced by Francois Assemat and Richard W. Wilson, 2006.
"""

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

import screens as scrs
scrn = scrs.Screen()



class PhaseScreen(object):
    """
    A "Phase Screen" for use in AO simulation.  Can be extruded infinitely.
    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.
    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 
        X = A.Z + B.b
    where X is the new phase vector, Z is some data from the existing screen,
    and b is a random vector with gaussian statistics.
    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:
        A =  Cov_xz . (Cov_zz)^(-1).
    B is less trivial.
        BB^t = Cov_xx - A.Cov_zx
    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 
        B = UL, 
    where U and L are obtained from the svd for BB^t
        U, w, U^t = svd(BB^t)
    L is a diagonal matrix where the diagonal elements are w^(1/2).    
    On initialisation an initial phase screen is calculated using an FFT based method.
    When 'add_row' is called, a new vector of phase is added to the phase screen.
    Existing points to use are defined by a "stencil", than is set to 0 for points not to use
    and 1 for points to use. This makes this a generalised base class that can be used by 
    other infinite phase screen creation schemes, such as for Von Karmon turbulence or 
    Kolmogorov turbulence.
    """
    def set_X_coords(self):
        """
        Sets the coords of X, the new phase vector.
        """
        self.X_coords = np.zeros((self.nx_size, 2))
        self.X_coords[:, 0] = -1
        self.X_coords[:, 1] = np.arange(self.nx_size)
        self.X_positions = self.X_coords * self.pixel_scale

    def set_stencil_coords(self):
        """
        Sets the Z coordinates, sections of the phase screen that will be used to create new phase
        """
        self.stencil = np.zeros((self.stencil_length, self.nx_size))

        max_n = 1
        while True:
            if 2 ** (max_n - 1) + 1 >= self.nx_size:
                max_n -= 1
                break
            max_n += 1

        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1

            coords = np.round(np.linspace(0, self.nx_size - 1, n_points)).astype('int32')
            self.stencil[col - 1][coords] = 1

        # Now fill in tail of stencil
        for n in range(1, self.stencil_length_factor + 1):
            col = n * self.nx_size - 1
            self.stencil[col, self.nx_size // 2] = 1

        self.stencil_coords = np.array(np.where(self.stencil == 1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale

        self.n_stencils = len(self.stencil_coords)

    def calc_seperations(self):
        """
        Calculates the seperations between the phase points in the stencil and the new phase vector
        """
        positions = np.append(self.stencil_positions, self.X_positions, axis=0)
        self.seperations = np.zeros((len(positions), len(positions)))

        if numba:
            calc_seperations_fast(positions, self.seperations)
        else:
            for i, (x1, y1) in enumerate(positions):
                for j, (x2, y2) in enumerate(positions):
                    delta_x = x2 - x1
                    delta_y = y2 - y1

                    delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

                    self.seperations[i, j] = delta_r


    def get_phase_covariance(self, scr=scrn):

        phase_cov = scr.phase_covariance(self.separations)
        return phase_cov

    def get_ft_phase_screen(self, scr=scrn):

        ft_phase_scr = scr.ft_phase_screen()
        return ft_phase_scr

    def make_covmats(self):
        """
        Makes the covariance matrices required for adding new phase
        """ 
        self.cov_mat = self.get_phase_covariance()

        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]

    def makeAMatrix(self):
        """
        Calculates the "A" matrix, that uses the existing data to find a new 
        component of the new phase vector.
        """
        # Cholsky solve can fail - if so do brute force inversion
        try:
            cf = linalg.cho_factor(self.cov_mat_zz)
            inv_cov_zz = linalg.cho_solve(cf, np.identity(self.cov_mat_zz.shape[0]))
        except linalg.LinAlgError:
            # print("Cholesky solve failed. Performing SVD inversion...")
            # inv_cov_zz = np.linalg.pinv(self.cov_mat_zz)
            raise linalg.LinAlgError("Could not invert Covariance Matrix to for A and B Matrices. Try with a larger pixel scale or smaller L0")

        self.A_mat = self.cov_mat_xz.dot(inv_cov_zz)

    def makeBMatrix(self):
        """
        Calculates the "B" matrix, that turns a random vector into a component of the new phase.
        """
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - self.A_mat.dot(self.cov_mat_zx)

        # Then do SVD to get B matrix
        u, W, ut = np.linalg.svd(BBt)

        L_mat = np.zeros((self.nx_size, self.nx_size))
        np.fill_diagonal(L_mat, np.sqrt(W))

        # Now use sqrt(eigenvalues) to get B matrix
        self.B_mat = u.dot(L_mat)

    def make_initial_screen(self):
        """
        Makes the initial screen usign FFT method that can be extended 
        """
        self._scrn = self.get_ft_phase_screen()

        self._scrn = self._scrn[:, :self.nx_size]

    def get_new_row(self):
        random_data = np.random.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]
        new_row = self.A_mat.dot(stencil_data) + self.B_mat.dot(random_data)

        new_row.shape = (1, self.nx_size)
        return new_row

    def add_row(self):
        """
        Adds a new row to the phase screen and removes old ones.
        """

        new_row = self.get_new_row()

        self._scrn = np.append(new_row, self._scrn, axis=0)[:self.stencil_length, :self.nx_size]

        return self.scrn

    @property
    def scrn(self):
        """
        The current phase map held in the PhaseScreen object in radians.
        """
        return self._scrn[:self.requested_nx_size, :self.requested_nx_size]


class PhaseScreenKolmogorov(PhaseScreen):
    """
    A "Phase Screen" for use in AO simulation using the Fried method for Kolmogorov turbulence.
    This represents the phase addition light experiences when passing through atmospheric 
    turbulence. Unlike other phase screen generation techniques that translate a large static 
    screen, this method keeps a small section of phase, and extends it as neccessary for as many 
    steps as required. This can significantly reduce memory consuption at the expense of more 
    processing power required.
    The technique is described in a paper by Assemat and Wilson, 2006 and expanded upon by Fried, 2008.
    It essentially assumes that there are two matrices, "A" and "B",
    that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by 
        X = A.Z + B.b
    where X is the new phase vector, Z is some data from the existing screen,
    and b is a random vector with gaussian statistics.
    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:
        A =  Cov_xz . (Cov_zz)^(-1).
    B is less trivial.
        BB^t = Cov_xx - A.Cov_zx
    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as 
        B = UL, 
    where U and L are obtained from the svd for BB^t
        U, w, U^t = svd(BB^t)
    L is a diagonal matrix where the diagonal elements are w^(1/2).    
    The Z data is taken from points in a "stencil" defined by Fried that samples the entire screen.
    An additional "reference point" is also considered, that is picked from a point separate from teh stencil 
    and applied on each iteration such that the new phase equation becomes:
    
    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen. The phase in the screen data
    is always accessed as ``<phasescreen>.scrn`` and is in radians.
    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        stencil_length_factor (int, optional): How much longer is the stencil than the desired phase? default is 4
    """
    def __init__(self, req_nx_size=256, pixel_scale=0.01, r0=1e-3, L0=1e3, random_seed=None, stencil_length_factor=4):

        self.req_nx_size = req_nx_size
        n = np.asarray([0])
        nx_size = np.asarray([0])
        while (2 ** n + 1) <= int(self.req_nx_size):
            n += 1
            nx_size = 2 ** n + 1

        self.nx_size = nxsize
        #self.nx_size = self.find_allowed_size()
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = stencil_length_factor
        self.stencil_length = stencil_length_factor * self.nx_size

        if random_seed is not None:
            np.random.seed(random_seed)

        # Coordinate of Fried's "reference point" that stops the screen diverging
        self.reference_coord = (1, 1)

        self.set_X_coords()
        self.set_stencil_coords()

        self.calc_seperations()
        self.make_covmats()

        self.makeAMatrix()
        self.makeBMatrix()
        self.make_initial_screen()


    def get_new_row(self):
        random_data = np.random.normal(0, 1, size=self.nx_size)

        stencil_data = self._scrn[(self.stencil_coords[:, 0], self.stencil_coords[:, 1])]

        reference_value = self._scrn[self.reference_coord]

        new_row = self.A_mat.dot(stencil_data - reference_value) + self.B_mat.dot(random_data) + reference_value

        new_row.shape = (1, self.nx_size)
        return new_row


    def __repr__(self):
        return str(self.scrn)
    


@numba.jit(nopython=True, parallel=True)
def calc_seperations_fast(positions, seperations):

    for i in numba.prange(len(positions)):
        x1, y1 = positions[i]
        for j in range(len(positions)):
            x2, y2 = positions[j]
            delta_x = x2 - x1
            delta_y = y2 - y1

            delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

            seperations[i, j] = delta_r



