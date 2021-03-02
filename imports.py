
# Imports necessary python packages & what have you (Branched Flow Honors Thesis 2021)
# Author: Jakob Faber

import numpy as np
from numpy import random
from numpy.random import randn
from numpy.fft import fft2, ifft2
from scipy.special import gamma
from scipy.interpolate import griddata
import scipy.constants as sc
import matplotlib.pyplot as plt
from numba import jit