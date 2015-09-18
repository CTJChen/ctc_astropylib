import numpy as np
from astropy.coordinates.distances import Distance
from astropy import units as u
import scipy.constants as c
from astropy.cosmology import FlatLambdaCDM

# takes in wise(AB) and converts to Jy or microJy


def wise_mag_to_jy(wise_mag, corr=None, mujy=None):
    mag_cor = [0.03, 0.04, 0.03, 0.03]
    wise_jy = np.zeros(4, dtype='float64')
    fiso_wise = [309.540, 171.787, 31.674, 8.363]  # in Jy
    if corr is True:
        wise_mag += mag_cor
    wise_jy = 10**(-wise_mag/2.5)*fiso_wise   # convert to Jy
    if mujy is True:
        wise_jy = 1e6*wise_jy
    return wise_jy


# convert a flux to nulnu at a certain wavelength in the unit of erg/s
# wav should be in micron, and flux should be in Jy
def flux_to_nulnu(flux, z, wav, ld=None, lsun=None, cosmo=None,mujy=None):
    if cosmo is None:
        # print 'no preset cosmology,use FlatLCDM w/ h0.7'
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dlum = Distance(z=z, unit=u.cm, cosmology=cosmo).value
    freq = 3e14/wav
    if mujy:jy2erg=1e29
    else:jy2erg=1e23
    nulnu = np.log10(flux*4*np.pi*dlum**2*freq/((1.+z)*jy2erg))
    if lsun is True:
        nulnu -= 33.5827
    return nulnu


# convert magnitude err to flux err
# when the flux is known
def magerr_to_ferr(flux, magerr):
    fluxerr = flux*(10**(magerr/2.5)-1)
    return fluxerr


def ab_to_jy(inp, tomag=None, mujy=None):
    if tomag is None:
        out = 3631*10**(-inp/2.5)
    else:
        out = 2.5*(23.-np.log10(inp))-48.6
    if mujy is True:
        out = 1e6*out
    return out


def jhk_mag_to_jy(inp, band=None, mujy=None):
    #
    # in Jy
    if band == 'j':
        fiso_jhk = 1594.
    elif band == 'h':
        fiso_jhk = 1024.
    elif band == 'ks':
        fiso_jhk = 666.7
    elif band is None:
        fiso_jhk = [1594., 1024., 666.7]
    inp = np.array(inp)
    jhk_jy = 10**(-1*inp/2.5)*fiso_jhk
    if mujy is True:
        jhk_jy = 1e6*jhk_jy
    return jhk_jy


def sdss_mag_to_jy(inp, band=None, mujy=None, inv=None):
    #
    # in Jy
    if not inv:
        if band == 'u':
            fiso_sdss = 3767.266
        elif band == 'g':
            fiso_sdss = 3631.
        elif band == 'r':
            fiso_sdss = 3631.
        if band == 'i':
            fiso_sdss = 3631.
        if band == 'z':
            fiso_sdss = 3564.727
        elif band is None:
            fiso_sdss = [3767.266, 3631., 3631., 3631., 3564.727]
        inp = np.asarray(inp)
        sdss_jy = 10**(-1*inp/2.5)*fiso_sdss
        if mujy is True:
            sdss_jy = 1e6*sdss_jy
        return sdss_jy
    else:
        if mujy:inp/=1e6
        if band == 'u':
            fiso_sdss = 3767.266
        elif band == 'g':
            fiso_sdss = 3631.
        elif band == 'r':
            fiso_sdss = 3631.
        if band == 'i':
            fiso_sdss = 3631.
        if band == 'z':
            fiso_sdss = 3564.727
        elif band is None:
            fiso_sdss = [3767.266, 3631., 3631., 3631., 3564.727]
        inp = np.asarray(inp)
        sdss_mag = -2.5*np.log10(inp/fiso_sdss)
        return sdss_mag
