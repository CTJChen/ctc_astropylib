#this function is used to calculate K-correction for a power-law X-ray SED.

from scipy.interpolate import InterpolatedUnivariateSpline as interpol
from scipy.integrate import trapz as tsum
import numpy as np

def int_pl(x1, x2, gam, nstp=None):
    if nstp is None:
        nstp=1000

    x=np.linspace(x1,x2,nstp)
    y=x**(1-gam)
    #whats this? to normalize! when calculating ``intrinsic luminosity'', assuming an intrinsic power-law spectrum of gamma=1.8
    #y0=x**(1-1.8)
    #interp_func0=interpol(y0,x)
    interp_func=interpol(y,x)
    
    #norm=interp_func0(10.)/interp_func(10.)
    #y*=norm
    integral=tsum(y,x)
    return integral


def interp_pl(l2kev, gam, nstp=None):
#Convert an input monochromatic luminosity at 2kev to 2-10 kev
#
#    
    if nstp is None:
        nstp=1000
    x=np.linspace(2.,10.,nstp)
    y=x**(1-gam)
    #whats this? to normalize! when calculating ``intrinsic luminosity'', assuming an intrinsic power-law spectrum of gamma=1.8
    #y0=x**(1-1.8)
    #interp_func0=interpol(y0,x)
    
    norm=(10**l2kev)/y[0]
    y*=norm
    integral=1000.*tsum(y,x)/4.1356675e-15
    return integral


