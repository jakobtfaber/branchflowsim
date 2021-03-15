import numpy as np
from numpy.random import randn

from scipy.special import gamma
from scipy.special import airy
from scipy.special import ai_zeros
from scipy.interpolate import *
from scipy.fftpack import fft2, ifft2, fftshift, fft, ifft
from scipy.ndimage import map_coordinates, filters
from scipy.signal import convolve2d
from scipy.stats import gaussian_kde

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
from IPython.display import Video

from RayTrace3D_Utilities import *

import sympy as sym
from sympy.printing.pycode import NumPyPrinter
from sympy.printing import ccode
from sympy import lambdify, Sum

import multiprocessing
from numba import jit
import builtins as bt
import math

# Define relevant constants
c = 2.998e10 #speed of light
pctocm = 3.0856e18 #1 pc in cm
GHz = 1e9 #central frequency
re = 2.8179e-13 #electron radius
kpc = 1e3 # in units of pc
autocm = 1.4960e13 #1 AU in cm
pi = np.pi

# Define starting coordinates on source plane (x,y) in dimensionless coordinates (u_x, u_y)
# where u_x = (x / a_x) and a_x is the characteristic length scale (same goes for u_y).
# This is done using Sympy
u_x, u_y = sym.symbols('u_x u_y')
A, B = 1.5e-2, 5

#Use Sympy to find derivatives of the potential
N, j, theta, phi, sigma = sym.symbols('N j theta phi sigma') #name variables

# Define various lens geometries

gaussrand = sigma * sym.sqrt(2/N) * sym.Sum(sym.cos(u_y*sym.cos(sym.Indexed(theta, j)) + \
                            u_x*sym.sin(sym.Indexed(theta, j)) + sym.Indexed(phi, j)), (j, 1, N))
#gaussrand = 2 * sym.sqrt(2) * sym.cos(u_y*sym.cos(0.25*np.pi) + u_x*sym.sin(0.25*np.pi) + 2*np.pi)
gauss = sym.exp(-u_x**2-u_y**2) #gaussian
ring = 2.7182*(u_x**2 + u_y**2)*gauss #ring
rectgauss = sym.exp(-u_x**4-u_y**4)
stgauss = gauss*(1. - A*(sym.sin(B*(u_x))+sym.sin(B*(u_y - 2*pi*0.3)))) #rectangular gaussian
asymgauss = sym.exp(-u_x**2-u_y**4) #asymmetrical gaussian
supergauss2 = sym.exp(-(u_x**2+u_y**2)**2) #gaussian squared
supergauss3 = sym.exp(-(u_x**2+u_y**2)**3) #gaussian cubed
superlorentz = 1./((u_x**2 + u_y**2)**2+1.) #lorentzian with width (gamma) of 2

# Define preferred lens geometry (use gauss as test case)
lensfunc = gaussrand

# Differentiate the lens equation to 1st, 2nd, and 3rd order using Sympy
#lensg = np.array([sym.diff(lensf, u_x), sym.diff(lensf, u_y)])
lensg = np.array([sym.diff(lensfunc, u_x), sym.diff(lensfunc, u_y)])
lensh = np.array([sym.diff(lensfunc, u_x, u_x), sym.diff(lensfunc, u_y, u_y), sym.diff(lensfunc, u_x, u_y)])
lensgh = np.array([sym.diff(lensfunc, u_x, u_x, u_x), \
                sym.diff(lensfunc, u_x, u_x, u_y), \
                   sym.diff(lensfunc, u_x, u_y, u_y), \
                       sym.diff(lensfunc, u_y, u_y, u_y)])

# Use Sympy to turn the lens equations into Numpy functions using Sympy
#lensfun = sym.lambdify([u_x, u_y, theta, phi, N, sigma], lensfunc, 'numpy')
#lensg = sym.lambdify([u_x, u_y, theta, phi, N, sigma], lensg, 'numpy')
#lensh = sym.lambdify([u_x, u_y, theta, phi, N, sigma], lensh, 'numpy')
#lensgh = sym.lambdify([u_x, u_y, theta, phi, N, sigma], lensgh, 'numpy')

#Gaussian screen functions & derivatives

scrfun = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.cos(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j]) for j in range(1, N-1))
scrgx = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(-np.sin(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j])*np.sin(theta[j]) for j in range(1, N-1))
scrgy = lambda u_x, u_y, theta, phi, N, sigma : \
        np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(-np.sin(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j])*np.cos(theta[j]) for j in range(1, N-1))
scrgxx = lambda u_x, u_y, theta, phi, N, sigma : \
        -np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(theta[j])**2*np.cos(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j]) for j in range(1, N-1))
scrgyy = lambda u_x, u_y, theta, phi, N, sigma : \
        -np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.cos(u_x*np.sin(theta[j]) + u_y*np.cos(theta[j]) + \
                phi[j])*np.cos(theta[j])**2  for j in range(1, N-1))
scrgxy = lambda u_x, u_y, theta, phi, N, sigma : \
        -np.sqrt(2)*sigma*np.sqrt(1/N)*bt.sum(np.sin(theta[j])*np.cos(u_x*np.sin(theta[j]) + \
                u_y*np.cos(theta[j]) + phi[j])*np.cos(theta[j]) for j in range(1, N-1))

def mapToUprime(uvec, alp, ax, ay, rF2, lc, sigma, theta, phi, N, V=None):
    """ 
    Parameters:
        uvec : vector containing u-plane coordinates
        alp : alpha coefficient
        ax : characteristic length scale in x
        ay : characteristic length scale in y
    Returns:   
        [upx, upy] : array of coordinates in the u-plane that have been
                    mapped to coordiantes in the u'-plane
    """
    ux, uy = uvec
    if V:
        V = np.gradient(V)
    upx = ux + alp*scrgx(ux, uy, theta, phi, N, sigma)/ax**2
    upy = uy + alp*scrgy(ux, uy, theta, phi, N, sigma)/ay**2
    rays = np.array([upx, upy])
    
    # Calculate Amplitude, Field, Phase and Phase Shift
    alp = rF2*lc
    psi20 = scrgxx(upx, upy, theta, phi, N, sigma)
    psi02 = scrgyy(upx, upy, theta, phi, N, sigma)
    psi11 = scrgxy(upx, upy, theta, phi, N, sigma)
    phi20 = ax**2/rF2 + lc*psi20
    phi02 = ay**2/rF2 + lc*psi02
    phi11 = lc*psi11
    H = phi20*phi02 - phi11**2
    sigma = np.sign(phi02)
    delta = np.sign(H)
    amp = (ax*ay/rF2)*np.abs(H)**-0.5
    
    phase = 0.5*rF2*lc**2*((scrgx(upx, upy, theta, phi, N, sigma)/ax)**2 + \
                        (scrgy(upx, upy, theta, phi, N, sigma)/ay)**2) + \
                        lc*scrfun(upx, upy, theta, phi, N, sigma) - 0.5*pi
    
    pshift = pi*(delta + 1)*sigma*0.25

    field = amp*np.exp(1j*(phase + pshift))
    dynspec = np.real(np.multiply(field, np.conj(field)))

    return rays, amp, phase, field, dynspec, pshift


# We are going to solve the lens equation to find the mapping between the
# u' and u plane numerically using a root finding algorithm.
# For now, however, this won't be necessary - but we'll set up the tools for later (3.2.21).
# Compute 100 zeros and values of the Airy function Ai and its derivative
# where index 1 for ai_zeros() returns the first 100 zeros of Ai’(x) and
# index 0 for airy() returns Ai'(x).
# Note for later: A caustic in which two images merge corresponds to a
# fold catastrophe, and close to a fold the field follows an Airy function pattern.
airyzeros = ai_zeros(100)[1]
airyfunc = airy(airyzeros)[0]**2/2.
airsqrenv = interp1d(airyzeros, airyfunc, kind = 'cubic', fill_value = 'extrapolate')

# Define screen paramters (Gaussian & Kolmogorov)
nscreen = 100
N = 10
sigma = 2
iscreen = 1 #which wavefront to plot
dso = 1.*kpc*pctocm #distance from source to observer
dsl = 1.*kpc*pctocm / nscreen #distance from source to screen
dm = -1e-5*pctocm #dispersion measure
#ax, ay = 0.04*autocm, 0.04*autocm #screen width (x,y)
ax, ay = 0.02*autocm, 0.02*autocm #screen width (x,y)
uxmax, uymax = 5., 5. #screen coordinates

# Frequencies
fmin, fmax = 1.4*GHz, 1.41*GHz #min/max frequency
nchan = 30
freqs = np.linspace(fmin, fmax, nchan)
print('Observation Frequency (GHz): ', (fmax+fmin)//2 * 1e-9)

# Construct u plane
npoints = 200
rx = np.linspace(-uxmax, uxmax, npoints)
ry = np.linspace(-uymax, uymax, npoints)
uvec = np.meshgrid(rx, ry)
ux, uy = uvec

raypropsteps = np.zeros((nchan+1, nscreen, 2, npoints, npoints)) #store ray wavefront at each screen
screens = np.zeros((nscreen, npoints, npoints)) #store phases at each screen
phases = np.zeros((nchan+1, nscreen, npoints, npoints))
phaseshift = np.zeros((nchan+1, nscreen, npoints, npoints))
fields = np.zeros((nchan+1, nscreen, npoints, npoints), dtype = np.float64)
dynspecs = np.zeros((nchan+1, nscreen, npoints, npoints))
amps = np.zeros((nchan+1, nscreen, npoints, npoints))
thetas = np.zeros((nscreen, N))
phis = np.zeros((nscreen, N))

for scr in range(nscreen):

    thetas[scr] = np.random.uniform(0, 2*np.pi, N)
    phis[scr] = np.random.uniform(0, 2*np.pi, N)
    screens[scr] = scrfun(ux, uy, thetas[scr], phis[scr], N, sigma)

for n in range(nchan):

    print('Channel Frequency: ', freqs[n])

    # Calculate coefficients for the scr equation
    rF2 = rFsqr(dso, dsl, freqs[n])#freqs[n])
    uF2x, uF2y = rF2*np.array([1./ax**2, 1./ay**2])
    lc = lensc(dm, freqs[n]) #freqs[n]) #calculate phase perturbation due to the scr
    #print('Phase Pertrubation $\phi_{0}$: ', lc)
    alp  = rF2*lc
    coeff = alp*np.array([1./ax**2, 1./ay**2])

    for scr in range(nscreen):

        if scr == 0:
            map_ = mapToUprime(uvec, alp, ax, ay, rF2, lc, sigma, thetas[scr], phis[scr], N)
            raypropsteps[n][scr] = map_[0]
            amps[n][scr] = map_[1]
            phases[n][scr] = map_[2]
            fields[n][scr] = map_[3]
            dynspecs[n][scr] = map_[4]
            phaseshift[n][scr] = map_[5]
            
        else:
            map_ = mapToUprime(raypropsteps[n][scr-1], alp, ax, ay, rF2, lc, sigma, thetas[scr], phis[scr], N)
            raypropsteps[n][scr] = map_[0]
            amps[n][scr] = map_[1]
            phases[n][scr] = map_[2]
            fields[n][scr] = map_[3]
            dynspecs[n][scr] = map_[4]
            phaseshift[n][scr] = map_[5]

chan = 15
print('Plotted Frequency (GHz): ', freqs[chan] * 1e-9)
screen = 24
print('Plotted Screen: ', screen)

# Construct Dynamic Spectrum
dynspec = np.zeros((nscreen, nchan, npoints))

for s in range(nscreen):
    for n in range(nchan):
        dynspec[s][n] = dynspecs[n][s][npoints//2, :]

dynspec_filt = filters.gaussian_filter(dynspec, 1)

# Construct Secondary Spectrum

secfft = np.fft.fftn((dynspec_filt[screen])-np.mean(dynspec_filt[screen]))
secreal = np.absolute(np.fft.fftshift(secfft))**2
secspec = 10*np.log10(secreal/np.max(secreal))

x_dat = raypropsteps[:, :, 0, :, :]
y_dat = raypropsteps[:, :, 1, :, :]

propx = np.zeros((nscreen, (npoints*npoints)))
propy = np.zeros((nscreen, (npoints*npoints)))

for s in range(nscreen):
    propx[s] = np.ndarray.flatten(x_dat[chan][s])
    propy[s] = np.ndarray.flatten(y_dat[chan][s])

fig = plt.figure(figsize = (10, 5))

ax0 = fig.add_subplot(231)
plt.imshow(screens[screen], aspect = 'auto')
plt.title('Screen')

ax1 = fig.add_subplot(232)
plt.imshow(phases[chan][screen], aspect = 'auto')
plt.title('Phases')
#plt.colorbar()

ax3 = fig.add_subplot(233)
xflat = np.ndarray.flatten(np.array(raypropsteps[chan][screen][0]))
yflat = np.ndarray.flatten(np.array(raypropsteps[chan][screen][1]))
plt.scatter(xflat, yflat, c = 'k', s = 0.02)
plt.title('Ray Tracing')

ax4 = fig.add_subplot(234)
plt.hist2d(xflat, yflat, (npoints//2, npoints//2))
plt.title('Ray Density')

ax5 = fig.add_subplot(235)
plt.imshow(amps[chan][screen], aspect = 'auto')
plt.title('Amplitude')

ax6 = fig.add_subplot(236)
plt.imshow(fields[chan][screen], aspect = 'auto')
plt.title('Electric Field')

plt.tight_layout()
fig.savefig('DiagnosticPlot1.png')

fig1 = plt.figure(figsize = (20, 10))

ax2 = fig1.add_subplot(231)
plt.plot(amps[chan][screen][:, npoints//2])
plt.title('Amplitude (Central Slice)')

ax3 = fig1.add_subplot(232)
plt.plot(fields[chan][screen][:, npoints//2])
plt.title('Electric Field (Central Slice)')

ax4 = fig1.add_subplot(233)
plt.imshow(dynspec[screen], aspect = 'auto')
plt.title('Dynamic Spectrum')

ax5 = fig1.add_subplot(234)
plt.imshow(secspec, aspect = 'auto')
plt.title('Secondary Spectrum')

#ax6 = fig1.add_subplot(235)
## get electric field impulse response
#p = np.fft.fft(np.multiply(dynspec[screen], np.blackman(nchan)[:, None]), 2*nchan)
#p = np.real(p*np.conj(p))  # get intensity impulse response
## shift impulse to middle of window
#pulsewin = np.transpose(np.roll(p, nchan))
#
#Freq = freqs/1000
#lpw = np.log10(pulsewin)
#vmax = np.max(lpw)
#vmin = np.median(lpw) - 3
##plt.pcolormesh(np.linspace(0, uxmax, nchan),
##              (np.arange(0, nchan, 1) - nchan/2) /
##               (2*(c/nchan)*Freq),
##               lpw[int(nchan/2):, :], vmin=vmin, vmax=vmax)
#plt.colorbar
#plt.ylabel('Delay (ns)')
#plt.xlabel('$x/r_f$')
#plt.plot(np.linspace(0, uxmax, nchan),
#         -dm/(2*(c/nchan)*Freq), 'k')  # group delay=-phase delay
#
plt.tight_layout()
fig1.savefig('DiagnosticPlot2.png')

