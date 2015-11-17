##this function is used to calculate K-correction for a power-law X-ray SED.

from scipy.interpolate import InterpolatedUnivariateSpline as interpol
from scipy.integrate import trapz as tsum
import numpy as np
from chen_stat import bayes_ci

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

def hrerr(s,h,err=None):
    '''
    this calculates the error in a hardness ratio from the errors in the
    individual count rates
    '''
    s=s.astype(float)
    h=h.astype(float)
    es = np.sqrt(s+0.75)+1.
    eh = np.sqrt(h+0.75)+1.
    hrout=(h-s)/(h+s)
    ehr = bayes_ci(np.abs(h-s),(h+s))
    #ehr2=((-1.)/(h+s)-(h-s)/(h+s)**2.)**2.*es**2.+((1.)/(h+s)-(h-s)/(h+s)**2.)**2.*eh**2.
    #ehr= np.sqrt(ehr2)

    if not err:
        return hrout
    else:
        return hrout,ehr    


def ledd(mbh, lam_edd, lx=None, bc=None):
    '''
    Get Eddington luminosity, or X-ray luminosity
    for a given MBH, Eddington ratio (lam_edd)
    If lx is set to True, the luminosity is converted to LX(2-10kev)
    using a simple bolometric correction factor of 22.4 (Lbol = 22.4LX)
    '''
    ledd = lam_edd*3.846e33*3.2e4*10**mbh
    if not lx:
        return ledd*u.erg/u.s
    else:
        if not bc:bc=22.4
        lx = ledd/bc
        return lx







