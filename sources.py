
# Source Types for Wave Optics Simulation (Branched Flow Honors Thesis 2021)
# Author: Jakob Faber

from imports import *
from numba import jit

# Sample params for testing:
# cn2 = 1e-14
# PropDist = 3000.0
# beam_waist = 0.05
# wavelength = 1e-06
# f_curv = -3000.0
# log2Nrange_list = [9, 8]

class Source():

    def __init__(self, N=256, SideLen=1.0, dtx = 0.1, drx=0.1, dx=5e-3, W0 = 5e-2, wavelen=1e-6, PropDist=10e3):

        self.N = N                         # number of grid points per side
        self.SideLen = SideLen             # Length of one side of square phase secreen [m]
        self.wavelen = wavelen             # Wavelength [m]
        self.k = 2 * np.pi / self.wavelen  # Optical wavenumber [rad/m]
        self.PropDist = PropDist           # Propagation distance(Path Length) [m]
        self.drx = drx                     # Diameter of aperture [m]
        self.dx = dx                       # Sampling interval at source plane
        self.dtx = dtx                     # Transmitting aperture size for Gauss [m]
        self.w0 = W0                       # the gaussian width of the beam

        # Place holders for geometry/source variables
        self.Source = None
        self.r1 = None
        self.x1 = None
        self.y1 = None
        self.rR = None
        self.xR = None
        self.yR = None
        self.Uout = None

        x = np.linspace(-self.N/2, (self.N/2)-1, self.N) * self.dx
        y = np.linspace(-self.N/2, (self.N/2)-1, self.N) * self.dx 
        self.x1, self.y1 = np.meshgrid(x, y)
        self.r1 = np.sqrt(self.x1**2 + self.y1**2) 


    def PlaneSource(self):
        #Uniform plane wave

        plane = np.ones([self.N,self.N]) 

        return plane
       
    def PointSource(self):
        #Schmidt Point Source
        
        droi = 4.0 *self.drx                 #Observation plane region [m]
        D1 = self.wavelen * self.PropDist / droi  #Central Lobe width [m]
        R = self.PropDist                         #Radius of curvature at wavefront [m]
        temp = np.exp(-1j*self.k/(2*R) * (self.r1**2)) / (D1**2)
        pt = temp * np.sinc((self.x1/D1)) * np.sinc((self.y1/D1)) * np.exp(-(self.r1/(4.0 * D1))**2)        
        
        return pt
 
      
    def FlattePointSource(self):
    
        fpt = np.exp(-(self.r1**2) / (10*( self.dx**2)) ) \
            * np.cos(-(self.r1**2) / (10*(self.dx**2)) )
        
        return fpt
        
      
    def CollimatedGaussian(self):
       
        source = np.exp(-(self.r1**2 / self.w0**2))
        source = source * self.MakePupil(self.dtx)
     
        return source


    def MakePupil(self,D_eval):
        #Target pupil creation

        boundary1 = -(self.SideLen / 2) #sets negative coord of sidelength
        boundary2 = self.SideLen / 2 #sets positive coord of sidelength
    
        A = np.linspace(boundary1, boundary2, self.N) #creates a series of numbers evenly spaced between
        #positive and negative boundary
        A = np.array([A] * self.N) #horizontal distance map created
        
        base = np.linspace(boundary1, boundary2, self.N) #creates another linspace of numbers
        set_ones = np.ones(self.N) #builds array of length N filled with ones
        B = np.array([set_ones] * self.N) 
        for i in range(0, len(base)):
            B[i] = B[i] * base[i] #vertical distance map created
        
        A = A.reshape(self.N,self.N)
        B = B.reshape(self.N,self.N) #arrays reshaped into matrices
    
        x_coord = A**2
        y_coord = B**2
    
        rad_dist = np.sqrt(x_coord + y_coord) #now radial distance has been defined
    
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
        mask = mask.reshape(self.N,self.N) #mask created and reshaped into a matrix
    
        return mask #returns the pupil mask as the whole function's output


