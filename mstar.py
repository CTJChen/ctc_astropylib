
import numpy as np
from astropy.coordinates.distances import Distance
from astropy import units as u
import scipy.constants as c
from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(H0=70,Om0=0.3)
#Flat cosmology, might want to update this
import pandas as pd
from os.path import expanduser
home = expanduser("~")
store=pd.HDFStore(home+'/work_scripts/dwarf/zibetti2009.h5')

from chen_observ import sdss_mag_to_jy
from chen_observ import flux_to_nulnu
from chen_observ import ab_to_jy
from chen_stat import dmod
dict_wav={'u':0.3543,'g':0.4770,'r':0.6231,'i':0.7625,'z':0.9134,
'U':0.36,'B':0.44,'V':0.55,'R':0.64,'I':0.79}
mabs_sun = {'u':6.41, 'g':5.15, 'r':4.67, 'i':4.56, 'z':4.53}

def z09_mstar(mags,band,color,color_str,redshift,ubv=None,ld=None,close=None):
    '''
    Use the Zibetti 2009 table B1 values
    to estimate stellar mass quickly
    input:
    1. magnitude(s) : array of magnitudes
    if it's ubvri, the mags should be in AB
    2. band(s) : array of strings, in the order of blue->red, don't use u/U band
    3. colors for sdss:u-g~z, g-r~z,r-i,r-z
    4. redshift : 
    set ld if luminosity distance is passed instaed of redshift
    '''
    wav=np.zeros(len(band),dtype=float)
    mag_dmod = dmod(redshift)
    for i in range(len(band)):
        wav[i]=dict_wav[band[i]]
        lband = (mabs_sun[band[i]] - (mags[i]-mag_dmod))/2.5
    def get_z09pars(band,color_str,ubv=None):
        if not ubv:
            z09=store['z09_sdss']
            pars=z09.loc[color_str,band].values
        else:
            z09=store['z09_ubv']
            pars=z09.loc[color_str,band].values
        return pars
    mass_band=np.zeros((len(band),len(color_str)),dtype=np.float64)
    for i in range(len(band)):
        for j in range(len(color_str)):
            pars=get_z09pars(band[i],color_str[j],ubv=ubv)
            mass_band[i,j]=lband[i]+pars[0]+0.2+pars[1]*color[j]
    if close:store.close()
    return mass_band


def sdss_z09(sdssmags, z=None,color = None, band=None):
    '''
    call z09_mstar to calculate the stellar mass
    using the input sdss magnitudes
    default: band = 'i', color = 'g-i'
    '''
    if color is None:
        color = 'g-i'
    if band is None:
        band = 'i'
    if z is None:
        z=0.000000001
    umag, gmag, rmag, imag, zmag = sdssmags
    color_str = ['u-g','u-r','u-i','u-z','g-r','g-i','g-z','r-i','r-z']
    sdsscolor = [umag-gmag, umag-rmag,umag-imag,umag-zmag,gmag-rmag,gmag-imag,gmag-zmag,rmag-imag,rmag-zmag]
    colors=pd.DataFrame({'bands':color_str,'color':sdsscolor})
    mags = [gmag, rmag, imag, zmag]
    bands = ['g','r','i','z']
    zmstar = z09_mstar(mags,bands,sdsscolor,color_str,z)
    mstar = pd.DataFrame(zmstar,index=bands,columns=color_str)
    return mstar.loc[band,color]


def sdss_z09_izi(abs_i,col_gi):
    '''
    calculate the stellar mass using i-band and g-i
    and solar abs. magnitude 4.56
    input : rest-frame g-i and rest-frame i-band abs mag
    log M = 1.032(col_gi) - 0.963 + log L
    '''
    abs_i_sun = 4.56
    loglsun = (abs_i_sun-abs_i)/2.5
    mstar = 1.032*col_gi-0.963+loglsun
    return mstar


def mstar_k(kmag,z):
    '''
    Use redshift and observed ks-band luminosity to estimate stellar mass
    '''
    dm = dmod(z)
    kmag_abs = kmag-dm
    lk_sun = (3.28-kmag_abs)/2.5
    mass_k = np.log10(0.31)+lk_sun
    return mass_k

