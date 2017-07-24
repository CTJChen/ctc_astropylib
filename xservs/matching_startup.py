#!/Users/ctchen/anaconda3/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

__doc__ = """
Import these functions to read data and make catalogs?
import pickle
with open(home+'/work_scripts/xservs_lss/data/hsc_cats.pickle', 'wb') as f:
    pickle.dump([hsc, hsc22, hsc23, cathsc, cathsc22, cathsc23], f)

"""

import os
from astropy.io import fits, ascii
from astropy.table import Table as tab
from scipy.optimize import leastsq as lstsqr


def READSDSS():
    qso = pd.read_csv(home+'/work_scripts/sdssquasar_lss.csv',comment='#')
    catqso = makecd(qso.ra,qso.dec)
    qso['rmag_psf'] = qso['r']
    return qso, catqso


def READXRAY(band='Full'):
    '''
    Convert count rates to fluxes using
    6.59*1e-12 for 0.5-10 keV band
    3.25*1e-12 for 0.5-2 keV band
    1.40*1e-11 for 2-10 keV band
    #sumdf = pd.read_csv(home+'/Dropbox/DB_XSERVS/catalogs/xmmcat_0503.csv',dtype={'EMLID':str,'cellid':str})
    #sumdf = pd.read_csv(home+'/Dropbox/DB_XSERVS/catalogs/xmmcat_0524_nc.csv',dtype={'EMLID':str,'cellid':str})
    sumdf = pd.read_csv(home+'/Dropbox/DB_XSERVS/catalogs/xmmcat_0601_nc.csv',
    dtype={'EMLID':str,'cellid':str,'RA':np.float64,'DEC':np.float64})
    #sumdf = pd.read_csv(home+'/Dropbox/DB_XSERVS/catalogs/xmmcat_edt0531_nc.csv',dtype={'EMLID':str,'cellid':str})
    #sumdf = sumdf[sumdf.EXT == 0.] #excluding extended sources
    #sumdf.reset_index(inplace=True)
    #sumdf.drop('index',axis=1,inplace=True)
    flux = sumdf.RATE.copy()*6.59e-12
    frac = np.mean(flux/sumdf.FLUX)
    sumdf['FLUX'] = flux
    sumdf['FLUX_ERR'] = sumdf.FLUX_ERR.copy()*frac
    sumdf['SASERR'] = np.sqrt(sumdf.RADEC_ERR**2 + 0.5**2)
    sumdf['r68'] = pd.Series(f_xerr(sumdf.SCTS.values.copy()),index=sumdf.index)
    sumdf['xposerr'] = pd.Series(sumdf.r68.values.copy()/1.51517,index=sumdf.index)
    '''
    if (band == 'Full') | (band == 'full') | (band == 'f'):
        sumdf = pd.read_csv(home+'/Dropbox/DB_XSERVS/catalogs/xfull_0602nc.csv',dtype={'EMLID':str,'cellid':str})
    elif (band == 'Soft') | (band == 'soft') | (band == 's'):
        sumdf = pd.read_csv(home+'/Dropbox/DB_XSERVS/catalogs/xsoft_0602nc.csv',dtype={'EMLID':str,'cellid':str})
    elif (band == 'Hard') | (band == 'hard') | (band == 'h'):
        sumdf = pd.read_csv(home+'/Dropbox/DB_XSERVS/catalogs/xhard_0602nc.csv',dtype={'EMLID':str,'cellid':str})
    xcat = makecd(sumdf.RA,sumdf.DEC)
    return sumdf,xcat

def READHSC(maglim=None,xdf=None):
    #HSC DEEP mag limits (grizy, Aihara et al. 2017):  26.8 26.6 26.5 25.6 24.8
    '''
        hsc = pd.read_csv(home + '/work_scripts/wide.csv')
        print(hsc.ra.count())
        hsc = pd.read_csv(home + '/Dropbox/DB_XSERVS/catalogs/hscwide_clean.csv.gz')
        hsc = pd.read_csv(home + '/Dropbox/DB_XSERVS/catalogs/hscwide_5band.csv.gz')
        print(hsc.ra.count())
    hsc = hsc[np.invert(hsc['yflags_pixel_edge'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_interpolated_any'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_saturated_any'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_cr_any'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_bad'])]
    hsc = hsc[np.invert(hsc['rflags_pixel_edge'])]
    hsc = hsc[np.invert(hsc['rflags_pixel_interpolated_any'])]
    hsc = hsc[np.invert(hsc['rflags_pixel_saturated_any'])]
    hsc = hsc[np.invert(hsc['rflags_pixel_cr_any'])]
    hsc = hsc[np.invert(hsc['rflags_pixel_bad'])]

    '''
    #'/Users/ctchen/Dropbox/DB_XSERVS/catalogs/hscwide_5band.csv.gz'
    #'hscwide_clean.csv.gz'
    #only require i-band photometry to be ok.
    #hsc = readfits(home + '/surveys/xxl/multiwavelength/hsc/hsc_wide_iband.fits')
    #hsc = pd.read_csv(home + '/Dropbox/DB_XSERVS/catalogs/hscwide_5band.csv.gz',dtype = {'# object_id':str,'specz_original_flag':str})
    hsc = pd.read_csv(home + '/Dropbox/DB_XSERVS/catalogs/hsc_wide_iband_pz.csv.gz',dtype = {'# object_id':str,'specz_original_flag':str})
    #
    #hsc = pd.read_csv(home + '/Dropbox/DB_XSERVS/catalogs/hscwide_clean.csv.gz')
    hsc = hsc[np.invert(hsc['iflags_pixel_edge'])]
    hsc = hsc[np.invert(hsc['iflags_pixel_interpolated_any'])]
    hsc = hsc[np.invert(hsc['iflags_pixel_saturated_any'])]
    hsc = hsc[np.invert(hsc['iflags_pixel_cr_any'])]
    hsc = hsc[np.invert(hsc['iflags_pixel_bad'])]
    hsc = hsc[np.invert(hsc['zflags_pixel_edge'])]
    hsc = hsc[np.invert(hsc['zflags_pixel_interpolated_any'])]
    hsc = hsc[np.invert(hsc['zflags_pixel_saturated_any'])]
    hsc = hsc[np.invert(hsc['zflags_pixel_cr_any'])]
    hsc = hsc[np.invert(hsc['zflags_pixel_bad'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_edge'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_interpolated_any'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_saturated_any'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_cr_any'])]
    hsc = hsc[np.invert(hsc['yflags_pixel_bad'])]
    hsc.rename(columns={'# object_id':'hscid'},inplace=True)
    hsc.drop_duplicates('hscid',inplace=True)
    hsc = hsc[hsc.pz_mizuki > 0.0]#trying photoz-limited sample
    hsc = hsc[(hsc.imag_psf <= 26.5) &
              (hsc.zmag_psf <= 25.6) &
              (hsc.ymag_psf <= 24.8)]
    hsc.set_index('hscid',inplace=True)
    if xdf is not None:#limit hsc catalog to the region probed by x-ray cat.
        hsc = hsc[(hsc.ra < max(xdf.RA) + 1./60.) & (hsc.ra > min(xdf.RA) - 1./60) &
              (hsc.dec < max(xdf.DEC) + 1./60) & (hsc.dec > min(xdf.DEC) - 1./60)]
    if maglim is not None:
        hsc = hsc[ (hsc.imag_psf <= maglim[0]) & (hsc.imag_psf >= maglim[1])]
    hsc['numid'] = range(len(hsc))
    cathsc = makecd(hsc.ra,hsc.dec)
    return hsc,cathsc

import pickle
with open(home+'/work_scripts/xservs_lss/data/hsc_cats.pickle', 'wb') as f:
    pickle.dump([hsc, hsc22, hsc23, cathsc, cathsc22, cathsc23], f)



if READIZY:
    hscriz = pd.read_csv(home + '/Dropbox/DB_XSERVS/catalogs/hscwide_clean.csv.gz')
    hscriz = hscriz[(hscriz.imag_psf <= 26.5) & (hscriz.rmag_psf <= 26.6)]
    hscriz.rename(columns={'# object_id':'hscid'},inplace=True)
    hscriz.set_index('hscid',inplace=True)
    cathscriz = makecd(hscriz.ra,hscriz.dec)
    hscriz['numid'] = range(len(hscriz))
    len(cathscriz)

def READVIDEO(maglim=None,xdf=None):
    fn = home+ '/Dropbox/DB_XSERVS/catalogs/videocat.csv.tar.gz'
    video = pd.read_csv(fn)
    #k-band selected?
    video = video[video.KSAUTOMAG.notnull() & video.KSAPERMAG3.notnull() &
    (video.PRIMARY_SOURCE == 1.0) & (video.KSAUTOMAG < 23.8)].reset_index()
    video.rename(columns={'RA2000':'ra','DEC2000':'dec'},inplace=True)
    video.loc[:,'SOURCEID'] = video.SOURCEID.astype(int).astype(str)
    if xdf is not None:#limit video catalog to the region probed by x-ray cat.
        video = video[(video.ra < max(xdf.RA) + 1./60.) & (video.ra > min(xdf.RA) - 1./60) &
              (video.dec < max(xdf.DEC) + 1./60) & (video.dec > min(xdf.DEC) - 1./60)]
    video['numid'] = range(len(video))
    catvideo = makecd(video.ra,video.dec)
    return video, catvideo

def READLSS(deep=False,xdf=None,maglim=None):
    #
    if deep:
        lss = readfits('/Users/ctchen/surveys/xxl/multiwavelength/cfhtls_deep.fits')
        lss = readfits('/Users/ctchen/surveys/xxl/multiwavelength/cfhtls_deep.fits')
    else:
        lss = pd.read_csv('/cuc36/surveys/xxl/multiwavelength/cfhtls_wide.tsv',sep='\t',comment='#')

    lss.rename(columns={'RAJ2000':'ra','DEJ2000':'dec'},inplace=True)
    lss.replace(['      '],np.nan,inplace=True)
    lss['imag'] = lss.imag.astype(float64)
    lss = lss[lss.imag <=24.8] #80% completeness
    if xdf is not None:
        lss = lss[(lss.ra < max(xdf.RA) + 1./60.) & (lss.ra > min(xdf.RA) - 1./60) &
              (lss.dec < max(xdf.DEC) + 1./60) & (lss.dec > min(xdf.DEC) - 1./60)]
    if maglim is not None:
        lss = lss[ (lss.imag <= maglim[0]) & (lss.imag >= maglim[1])]
    lss.set_index('CFHTLS',inplace=True)
    catlss = makecd(lss.ra,lss.dec)
    lss['numid'] = pd.Series(range(len(lss)),index=lss.index)
    return lss, catlss
def READSERVS():
    fname = '/cuc36/xxl/multiwavelength/vacc/xmm/121212/xmm-irac12-sextractor.fits.gz'
    #fname = '/cuc36/xxl/multiwavelength/vacc/xmm/121212/servs-xmm-data-fusion-sextractor.fits.gz'
    servs = tab(fits.getdata(fname,1)).to_pandas()
    servs = servs[(servs.FLUX_APER_3_1 > 1.0) & (servs.FLUX_APER_3_2 > 1.0)]
    servs['ID_SERVS'] = 'IRAC1_' + servs.ID_1.astype(str).str.zfill(6)
    servs['mag1'] = ab_to_jy(servs.FLUX_APER_3_1.values.copy()/1e6,tomag=True)
    servs['mag2'] = ab_to_jy(servs.FLUX_APER_3_2.values.copy()/1e6,tomag=True)
    servs.set_index('ID_12',inplace=True)
    servs['numid'] = pd.Series(range(len(servs)), index=servs.index)
    servs.rename(columns={'RA_12':'ra','DEC_12':'dec'},inplace=True)
    catservs = makecd(servs.ra,servs.dec)
    return servs,catservs#Check Data Fusion
