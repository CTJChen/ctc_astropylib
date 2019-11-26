# this function is used to calculate K-correction for a power-law X-ray SED.
import pandas as pd
from scipy.interpolate import interp1d as interpol
from scipy.integrate import trapz as tsum
import numpy as np
from ctc_stat import bayes_ci
import sys
from astropy.io import fits
from astropy.table import Table as tab
from scipy.stats import norm
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import subprocess
import math
from os.path import expanduser
import sys
home = expanduser("~")
'''
import progressbar
pbar = progressbar.ProgressBar(widgets=[
    progressbar.Percentage(), '|', progressbar.Counter('%6d'),
    progressbar.Bar(), progressbar.ETA()])
'''

def int_pl(x1, x2, gam, nstp=None):
    if nstp is None:
        nstp = 1000

    x = np.linspace(x1, x2, nstp)
    y = x**(1 - gam)
    # whats this? to normalize! when calculating ``intrinsic luminosity'', assuming an intrinsic power-law spectrum of gamma=1.8
    # y0=x**(1-1.8)
    # interp_func0=interpol(y0,x)
    interp_func = interpol(y, x)

    # norm=interp_func0(10.)/interp_func(10.)
    # y*=norm
    integral = tsum(y, x)
    return integral


def interp_pl(l2kev, gam, nstp=None):
    # Convert an input monochromatic luminosity at 2kev to 2-10 kev
    #
    #
    if nstp is None:
        nstp = 1000
    x = np.linspace(2., 10., nstp)
    y = x**(1 - gam)
    # whats this? to normalize! when calculating ``intrinsic luminosity'', assuming an intrinsic power-law spectrum of gamma=1.8
    # y0=x**(1-1.8)
    # interp_func0=interpol(y0,x)

    norm = (10**l2kev) / y[0]
    y *= norm
    integral = 1000. * tsum(y, x) / 4.1356675e-15
    return integral


def hrerr(s, h, err=None):
    '''
    this calculates the error in a hardness ratio from the errors in the
    individual count rates
    '''
    s = s.astype(float)
    h = h.astype(float)
    es = np.sqrt(s + 0.75) + 1.
    eh = np.sqrt(h + 0.75) + 1.
    hrout = (h - s) / (h + s)
    ehr = bayes_ci(np.abs(h - s), (h + s))
    # ehr2=((-1.)/(h+s)-(h-s)/(h+s)**2.)**2.*es**2.+((1.)/(h+s)-(h-s)/(h+s)**2.)**2.*eh**2.
    #ehr= np.sqrt(ehr2)

    if not err:
        return hrout
    else:
        return hrout, ehr


def ledd(mbh, lam_edd, lx=None, bc=None):
    '''
    Get Eddington luminosity, or X-ray luminosity
    for a given MBH, Eddington ratio (lam_edd)
    If lx is set to True, the luminosity is converted to LX(2-10kev)
    using a simple bolometric correction factor of 22.4 (Lbol = 22.4LX)
    '''
    ledd = lam_edd * 3.846e33 * 3.2e4 * 10**mbh
    if not lx:
        return ledd * u.erg / u.s
    else:
        if not bc:
            bc = 22.4
        lx = ledd / bc
        return lx


def behr_wrap(softsrc, hardsrc, softbkg, hardbkg, softarea, hardarea, softeff, hardeff,verbose=False, invertBR=False, prog = False):
    wd_behr = home + '/lib/BEHR/ &&'
    if invertBR:
        str_ssrc = ' softsrc=' + str(math.ceil(hardsrc))
        str_hsrc = ' hardsrc=' + str(math.ceil(softsrc))
        str_sbkg = ' softbkg=' + str(math.ceil(hardbkg))
        str_hbkg = ' hardbkg=' + str(math.ceil(softbkg))
        str_sarea = ' softarea=' + str(hardarea)
        str_harea = ' hardarea=' + str(softarea)
        str_seff = ' softeff=' + str(hardeff)
        str_heff = ' hardeff=' + str(softeff)
    else:
        str_ssrc = ' softsrc=' + str(math.ceil(softsrc))
        str_hsrc = ' hardsrc=' + str(math.ceil(hardsrc))
        str_sbkg = ' softbkg=' + str(math.ceil(softbkg))
        str_hbkg = ' hardbkg=' + str(math.ceil(hardbkg))
        str_sarea = ' softarea=' + str(softarea)
        str_harea = ' hardarea=' + str(hardarea)
        str_seff = ' softeff=' + str(hardeff)
        str_heff = ' hardeff=' + str(softeff)
    behr_run = 'cd ' + wd_behr + ' ./BEHR' + str_ssrc + str_hsrc + \
        str_sbkg + str_hbkg + str_sarea + str_harea + str_seff + str_heff + ' && cd -'
    if verbose:
        print(behr_run)
    p = subprocess.Popen(behr_run, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    td = StringIO(p.communicate()[0].decode('ascii'))
    df = pd.DataFrame(columns=['Out', 'Mode', 'Mean', 'Median', 'LowerB', 'UpperB'],index=range(3))
    for line in td:
        if '#(S/H)' in line:
            df.loc[0,:] = str.split(line)
        elif '(H-S)/(H+S)' in line:
            df.loc[1,:] = str.split(line)
        elif 'log10(S/H)' in line:
            df.loc[2,:] = str.split(line)
    df.ix[:,1] = df.ix[:,1].astype(float)
    df.ix[:,2] = df.ix[:,2].astype(float)
    df.ix[:,3] = df.ix[:,3].astype(float)
    return df


def behrhug(df_input, verbose=False, invertBR=False, prog=False,nprog=20):
    '''
    Python wrapper of BEHR
    set invertBR=Tru for band ratio H/S
    otherwise, the default output is S/H
    Note that the hardness ratio would be wrong if invertBR is set.
    '''
    cols = ['BR', 'BR_LB', 'BR_UB', 'HR', 'HR_LB',
            'HR_UB', 'logBR', 'logBR_LB', 'logBR_UB',
            'mBR','mHR','mlogBR']
    df_out = pd.DataFrame(index=df_input.index,
                          columns=cols)
    if prog:
        fracid = np.linspace(0,len(df_input),nprog,dtype=int)
    for index, row in df_input.iterrows():
        df = behr_wrap(row.softsrc, row.hardsrc, row.softbkg, row.hardbkg, row.softarea, row.hardarea,
                       row.softeff, row.hardeff, verbose=verbose, invertBR=invertBR)
        df_out.loc[index, ['BR', 'BR_LB', 'BR_UB','mBR']] = df.loc[
            0, ['Median', 'LowerB', 'UpperB','Mode']].values.astype(float)
        df_out.loc[index, ['HR', 'HR_LB', 'HR_UB','mHR']] = df.loc[
            1, ['Median', 'LowerB', 'UpperB','Mode']].values.astype(float)
        df_out.loc[index, ['logBR', 'logBR_LB', 'logBR_UB','mlogBR']] = df.loc[
            2, ['Median', 'LowerB', 'UpperB','Mode']].values.astype(float)
        if prog:
            if index in fracid:
                print(int(100*index/len(df_out)),'%')
    return df_out

def xmm_bkgd(filename, df=False, fit=False, sig=None):
    '''
    The input file should be the output of the step 1 in:
    http://www.cosmos.esa.int/web/xmm-newton/sas-thread-epic-filterbackground
    which bins the photons in 100s time intervals
    '''
    if sig is None:
        sig = 3
    inp = fits.getdata(filename)
    inp = tab(inp).to_pandas()
    # Fit a gaussian model to the rate distribution
    if df is True:
        return inp
    elif fit is True:
        mu,std = norm.fit(inp.RATE.values)
        r = [mu,std]
        return r
    else:
        mu,std = norm.fit(inp.RATE.values)
        r = [mu,std*sig]
        return r

def xmm_gti(filename, df=False):
    '''
    Read GTI fits file and return the total GTI time in ks.
    '''
    inp = tab(fits.getdata(filename)).to_pandas()
    gti = 0.
    for index, row in inp.iterrows():
        gti = gti + row.STOP-row.START
    return gti/1000.


def nherr(nh, nhuerr, nhlerr, unit=None, output=False):
    '''
    convert XSPEC NH output (in unit of 10^22 unless specified)
    intou log NH and uncertainties in log
    example : NH = 3.3 (-0.5, +1.0) * 10^22
    convert to log using nherr(3.3,-0.5,1.0)
    '''
    #make sure if nhlerr is not negative
    nhlerr = np.abs(nhlerr)
    if nhuerr < 0:print('first argument nhuerr should be a positive number')
    if unit is None:
        unit = 22
    lnh = np.log10(nh * 10**unit)
    if nhlerr >= nh:
        lnhlerr = np.nan
    else:
        lnhlerr = lnh - np.log10((nh + nhlerr) * 10**unit)
    lnhuerr = np.log10((nh + nhuerr) * 10**unit) - lnh
    print('NH=' + str(np.round(lnh, decimals=2)) + '+' +
          str(np.round(lnhuerr, decimals=2)) + str(np.round(lnhlerr, decimals=2)))
    if output:
        return [lnh, lnhlerr, lnhuerr]
    else:
        return


def emllist(df,mosaic=False):
    '''
    Takes a dataframe, made by reading emldetect products
    Makes a summary dataframe, calculate average off-axis angle of different detectors
    '''
    if not mosaic:
        df_out = df.copy()
        dfpn = df_out[df_out.ID_INST == 1.0]
        dfm1 = df_out[df_out.ID_INST == 2.0]
        dfm2 = df_out[df_out.ID_INST == 3.0]
        df_out = df_out[df_out.ID_INST == 0.0]
        dfm1.is_copy = False
        dfm2.is_copy = False
        if (len(dfpn) == len(dfm2)) and (len(dfm1) == len(dfpn)):
            #PN, M1, M2 have the same length
            dfpn.is_copy = False
            dfm1.is_copy = False
            dfm2.is_copy = False
            df_out['OFFAXm1'] = dfm1.OFFAX.values
            df_out['OFFAXm2'] = dfm2.OFFAX.values
            df_out['OFFAXpn'] = dfpn.OFFAX.values
            df_out['OFFAX'] = (df_out.OFFAXm1+df_out.OFFAXm2+df_out.OFFAXpn)/3.
        elif len(dfm1) == len(dfm2):
            #usually it's PN that's problematic, so consider only M1 and M2
            dfm1.is_copy = False
            dfm2.is_copy = False
            df_out['OFFAXm1'] = dfm1.OFFAX.values
            df_out['OFFAXm2'] = dfm2.OFFAX.values
            df_out['OFFAX'] = (df_out.OFFAXm1+df_out.OFFAXm2)/2.
        elif (len(dfpn) == len(dfm2)):
            #unless it's m1 that's problematic
            dfpn.is_copy = False
            dfm2.is_copy = False
            df_out['OFFAXpn'] = dfpn.OFFAX.values
            df_out['OFFAXm2'] = dfm2.OFFAX.values
            df_out['OFFAX'] = (df_out.OFFAXpn+df_out.OFFAXm2)/2.
        elif (len(dfpn) == len(dfm2)):
            #it's also possible m2 is problematic
            dfpn.is_copy = False
            dfm1.is_copy = False
            df_out['OFFAXpn'] = dfpn.OFFAX.values
            df_out['OFFAXm1'] = dfm1.OFFAX.values
            df_out['OFFAX'] = (df_out.OFFAXpn+df_out.OFFAXm1)/2.
        elif len(dfm1) == len(df_out):
            print('using M1 off-axis angle')
            dfm1.is_copy = False
            df_out['OFFAXm1'] = dfm1.OFFAX.values
            df_out['OFFAX'] = df_out.OFFAXm1
        else:
            print('more than two cameras are problematic, returning the original with OFFAX=np.nan')
            df_out['OFFAX'] = np.nan

            #
        df_out.reset_index(inplace=True)
        return df_out
    else:
        df_out = df.copy()
        df_out = df_out[df_out.ID_INST == 0.0]
        df_out.reset_index(inplace=True)
        return df_out
