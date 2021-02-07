#!/usr/bin/env python
# -*- coding: utf-8 -*-
__doc__ = """
if you have a set of SDSS magnitudes at a given redshift, you can get 
the stellar mass by doing the following:

from mass_to_light import *
sdss_z09(sdssmags,z=redshift,color = 'r-z') 

If you want to change the cosmology usead you should modify the source code below. 
Also, the software calls for a table from Zibetti et al. 2009, you can change the path as you wish.
"""

import numpy as np
from astropy.coordinates.distances import Distance
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(H0=70,Om0=0.3)
import pandas as pd
from os.path import expanduser
home = expanduser("~")
storez09=pd.HDFStore(home+'/zibetti2009.h5')    
store=storez09
dict_wav={'u':0.3543,'g':0.4770,'r':0.6231,'i':0.7625,'z':0.9134,
'U':0.36,'B':0.44,'V':0.55,'R':0.64,'I':0.79}
mabs_sun = {'u':6.41, 'g':5.15, 'r':4.67, 'i':4.56, 'z':4.53}


def flux_to_nulnu(flux, z, wav, ld=None, lsun=None, cosmo=None,mujy=None):
	# convert a flux to nulnu at a certain wavelength in the unit of erg/s
	# wav should be in micron, and flux should be in Jy
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

def sdss_mag_to_jy(inp, band=None, mujy=None, inv=None):
    #
    # in Jy
    inp = np.asarray(inp)
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
            fiso_sdss = np.asarray([3767.266, 3631., 3631., 3631., 3564.727])
        sdss_jy = 10**(-1*inp/2.5)*fiso_sdss
        if mujy is True:
            sdss_jy = 1e6*sdss_jy
        return sdss_jy
    else:
        if mujy:inp=inp/1e6
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
            fiso_sdss = np.asarray([3767.266, 3631., 3631., 3631., 3564.727])
        sdss_mag = np.log10(inp/fiso_sdss)*(-2.5)
        return sdss_mag


def mtl_mstar(mags,band,color,color_str,redshift,ld=None,close=None, method='z09'):
    '''
    Use the Zibetti 2009 table B1 values (MTL = mass-to-light ratio)
    or Bell 2003, default is z09
    to estimate stellar mass quickly
    input:
    1. magnitude(s) : array of magnitudes
    if it's ubvri, the mags should be in AB
    2. band(s) : array of strings, in the order of blue->red, don't use u/U band
    3. colors for sdss:u-g~z, g-r~z,r-i,r-z
    4. redshift : 
    set ld if luminosity distance is passed instaed of redshift
    '''
    def get_lum(mag,redshift,ld=ld):
        #convert mags to fluxes
        flux=[]
        if len(band)==5:
            flux=sdss_mag_to_jy(mag)
        else:
            for i in range(len(band)):
                flux.append(sdss_mag_to_jy(mag[i],band=band[i]))
        #now calculate luminosity distances using z and flux
        flux=np.asarray(flux)
        lband=[]
        lband=flux_to_nulnu(flux,redshift,wav,lsun=True,cosmo=cosmo,ld=ld)
        #in log lsun unit
        return lband

    def get_pars(band,color_str):
        if method is 'z09':
            mtl_sdss=store['z09_sdss']
            #mtl_bri=store['z09_bri']
        elif method is 'b03':
            mtl_bri=store['b03_bri']
            mtl_sdss = store['b03_sdss']
        #if ubv is True:
        #    pars=mtl_bri.loc[color_str,band].values
        else:
            print('method could only be z09 or b03')
        pars = mtl_sdss.loc[color_str,band].values
        return pars    

    wav=np.zeros(len(band),dtype=np.float64)
    for i in range(len(band)):
        wav[i]=dict_wav[band[i]]                
    #Using lband, pars and mag_band to calculate mass_band
    lband=get_lum(mags,redshift)
    #print lband
    mass_band=np.zeros((len(band),len(color_str)),dtype=np.float64)
    for i in range(len(band)):
        for j in range(len(color_str)):
            pars = get_pars(band[i],color_str[j])
            mass_band[i,j]=lband[i]+pars[0]+pars[1]*color[j]
    if close:store.close()
    return mass_band

    


def sdss_mass(sdssmags, z=None,color = None, band=None, method='z09'):
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
    if method is 'z09':
        zmstar = mtl_mstar(mags,bands,sdsscolor,color_str,z,method=method)
        mstar = pd.DataFrame(zmstar,index=bands,columns=color_str)
    elif method is 'b03':
        bmstar = mtl_mstar(mags,bands,sdsscolor,color_str,z,method=method)
        mstar = pd.DataFrame(zmstar,index=bands,columns=color_str)
    else:
        print('method can only be z09 or b03')
    return mstar.loc[band,color]

