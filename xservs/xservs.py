#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
#%matplotlib inline
import numpy as np
import pandas as pd
from ctc_observ import *
from ctc_arrays import *
from scipy.integrate import quad
import os
from scipy import stats
#simid = os.environ['simid']
#home = '/Users/cuc36'
__doc__ = " "

def get_photoz_stat(zs, zp):
    '''
    Calculate sigma_nmad, median(det_z / (1 + z_spec)), 
    outlier ids, outlier num and fraction, and print on the screen
    
    Inputs:
        zs, z_spec array
        zp, z_phot array
    Output:
        out, outlier id array
    '''

    det_z = zp - zs     # Delta_z and its median value
    med_det_z = np.median(det_z)
    sigm = 1.48 * np.median(abs(det_z - med_det_z) / (1 + zs))  # Sigma_nmad

    err = det_z / (1 + zs)  # Relative errors
    out = np.arange(len(zs))[abs(err) > 0.15]   # Outlier ids, using relative error > 0.15 criterion 
    out_num = len(out)  # outlier num and fraction
    out_fra = float(len(out)) / len(zs)
    
    med = np.median(det_z / (1 + zs))   # Median(det_z / (1 + z_spec)) 

    print('Source Num. %d' %len(zs))    # Output to screen
    print('Sigma_nmad: %.4f' %sigm)
    print('Outlier: num=%d, frac=%.3f' %(out_num, out_fra))
    print('Median_error: %.5f' %med)

    return out  # Return outlier ids

def ratetoflux(row,band='soft',err=False,ind=1.):
    '''
    To be applied to XMM EMLDETECT product
    compute error-weighted fluxes (among three EPIC detectors)

    '''
    if band == 'soft':
        ecf = [1e-11/6.23, 1e-11/1.77]
    elif band == 'hard':
        ecf = [1e-11/1.15, 1e-11/0.43]
    elif band == 'full':
        ecf = [1e-11/2.84, 1e-11/0.88]
    elif band == 'med':
        ecf = [1e-11/1.9496978, 1e-11/0.76394]
    elif band == 'uhd':
        ecf = [1e-11/0.66357, 1e-11/0.18695]
    else:
        print('Wrong Band Name -- only full/soft/hard/uhd/med. Assuming soft band')
        ecf = [1e-11/7.066, 1e-11/1.989]
    totalerr = 0.
    total = 0.
    rates = np.array([row.ratem1,row.ratem2,row.ratepn])
    rateerr = np.array([row.ratem1_err,row.ratem2_err,row.ratepn_err])
    ecfs = np.array([ecf[1],ecf[1],ecf[0]])
    exps = np.array([row.expm1,row.expm2,row.exppn])
    if (min(rates) == 0.) & (max(rates[rates > min(rates)])/min(rates[rates > min(rates)]) > 5):
        #Avoiding underflow
        rates[rates == min(rates[rates > min(rates)])] = 0.

    if not err:
        for index, i in enumerate(rates):
            if i*exps[index] > 1.:
                totalerr += 1 / (rateerr[index]**ind)
                total += ecfs[index] * i / (rateerr[index]**ind)
        return total/totalerr
    else:
        for index, i in enumerate(rates):
            if i*exps[index] > 1.:
                instflux = i*ecfs[index]
                instfluxerr = instflux*rateerr[index]/i
                totalerr += 1./(instfluxerr**2)
        return np.sqrt(1/totalerr)

def simlike(dfsim, dfdet, radius = 15.):
    '''
    compute rsq = dra / raerr**2 + ddec / decerr**2 + dflux / fluxerr**2
    forevery input source within 15":
    '''
    catin = makecd(dfsim.ra,dfsim.dec)
    catout = makecd(dfdet.RA,dfdet.DEC)
    idin, idout, d2d, d3d = catout.search_around_sky(catin,15*u.arcsec)
    dist = np.zeros(len(dfsim)) + np.nan
    simidout = np.zeros(len(dfsim)) + np.nan
    simidin = np.zeros(len(dfsim)) + np.nan
    simidout[idin] = idout
    simidin[idin] = 1
    dist[idin] = d2d.arcsec
    radecerr = dfdet.RADEC_ERR.values.copy()[idout] + 0.5
    fluxerr = dfdet.FLUX_ERR.values.copy()[idout]

    dra = ((catin.ra[idin] - catout.ra[idout]).arcsec / radecerr )**2
    ddec = ((catin.dec[idin] - catout.dec[idout]).arcsec / radecerr )**2
    dflux = ((dfsim.flux.values[idin] - dfdet.flux.values[idout]) / fluxerr)**2
    rsq = np.zeros(len(dfsim)) + np.nan
    rsq[idin] = dra + ddec + dflux
    return simidin,simidout,dist, pd.Series(rsq,index=dfsim.index)


#Define logN-logS

def logNlogS(flux, dnds = False, ref='XCOS-Soft', K = None, g1 = None, g2 = None, sb = None):
    '''
    Based on the ChaMP logN logS relation presented by Kim et al. 2007
    gamma=1.7
    flux should be in 1e-15 ergs/s/cm2
    '''
    if ref == 'XCOS-soft':
        A = 140.
        g1 = 1.60
        g2 = 2.40
        sb = 1.
        K = A / (sb**(g2-g1))
    elif ref == 'ChaMP-full':
        K = 1407.
        g1 = 1.64
        g2 = 2.48
        sb = 19.2
    elif ref == 'XCOS-hard':
        A = 4130.
        g1 = 1.55
        g2 = 2.46
        sb = 10.5
        K = A / (sb**(g2-g1))
    else:
        print('ref not available, should manually set K, g1, g2, sb')
        print('or use :XCOS-hard, XCOS-soft, ChaMP-full')
    if np.isscalar(flux):
        if flux < sb:
            if dnds:
                ns = K * flux**(-g1)
            else:
                ns = K * ( (1/(1-g1)) - ( 1/(1-g2) ) ) * sb**(1-g1) + \
                K * (1/(g1-1)) * flux**(1-g1)
        else:
            if dnds:
                ns = K * sb**(g2-g1) *flux**(-g2)
            else:
                ns = K*(1/(g2-1)) * (sb**(g2-g1)) * (flux**(1-g2))
        return ns
    else:
        ns = np.zeros_like(flux)
        if dnds:
            ns[flux < sb] = K * flux[flux < sb]**(-g1)
            ns[flux >= sb] = ns = K * sb**(g2-g1) *flux[flux >= sb]**(-g2)
        else:
            ns[flux < sb] = K * ( (1/(1-g1)) - ( 1/(1-g2) ) ) * sb**(1-g1) + \
            K * (1/(g1-1)) * flux[flux < sb]**(1-g1)
            ns[flux >= sb] = K*(1/(g2-1)) * (sb**(g2-g1)) * (flux[flux >= sb]**(1-g2))
        return ns

def NS_soft(flux,dnds = False):
	'''
	Based on Cappelluti+2009 XMM-COSMOS logN-logS
	essentially, K = B, alpha1 = gamma2, alpha2 = gamma1
	flux should be in 1e-15 ergs/s/cm2
	#A = 141, so B = 141/10.**(gamma2-gamma1)
	'''
	A = 1400.
	g1 = 1.60
	g2 = 2.40
	sb = 10.
	K = A/(sb**(g2-g1))
	if np.isscalar(flux):
		if flux < sb:
			if dnds:
				ns = K * flux**(-g1)
			else:
				ns = K * ( (1/(1-g1)) - ( 1/(1-g2) ) ) * sb**(1-g1) + \
				K * (1/(g1-1)) * flux**(1-g1)
		else:
			if dnds:
				ns = K * sb**(g2-g1) *flux**(-g2)
			else:
				ns = K*(1/(g2-1)) * (sb**(g2-g1)) * (flux**(1-g2))
		return ns
	else:
		ns = np.zeros_like(flux)
		if dnds:
			ns[flux < sb] = K * flux[flux < sb]**(-g1)
			ns[flux >= sb] = ns = K * sb**(g2-g1) *flux[flux >= sb]**(-g2)
		else:
			ns[flux < sb] = K * ( (1/(1-g1)) - ( 1/(1-g2) ) ) * sb**(1-g1) + \
			K * (1/(g1-1)) * flux[flux < sb]**(1-g1)
			ns[flux >= sb] = K*(1/(g2-1)) * (sb**(g2-g1)) * (flux[flux >= sb]**(1-g2))
		return ns



def NS_hard(flux,dnds = False):
	'''
	Based on Cappelluti+2009 XMM-COSMOS logN-logS
	essentially, K = B, alpha1 = gamma2, alpha2 = gamma1
	flux should be in 1e-15 ergs/s/cm2
	#A = 141, so B = 141/10.**(gamma2-gamma1)
	'''
	A = 4130.
	g1 = 1.55
	g2 = 2.46
	sb = 10.5
	K = A/(sb**(g2-g1))
	if np.isscalar(flux):
		if flux < sb:
			if dnds:
				ns = K * flux**(-g1)
			else:
				ns = K * ( (1/(1-g1)) - ( 1/(1-g2) ) ) * sb**(1-g1) + \
				K * (1/(g1-1)) * flux**(1-g1)
		else:
			if dnds:
				ns = K * sb**(g2-g1) *flux**(-g2)
			else:
				ns = K*(1/(g2-1)) * (sb**(g2-g1)) * (flux**(1-g2))
		return ns
	else:
		ns = np.zeros_like(flux)
		if dnds:
			ns[flux < sb] = K * flux[flux < sb]**(-g1)
			ns[flux >= sb] = ns = K * sb**(g2-g1) *flux[flux >= sb]**(-g2)
		else:
			ns[flux < sb] = K * ( (1/(1-g1)) - ( 1/(1-g2) ) ) * sb**(1-g1) + \
			K * (1/(g1-1)) * flux[flux < sb]**(1-g1)
			ns[flux >= sb] = K*(1/(g2-1)) * (sb**(g2-g1)) * (flux[flux >= sb]**(1-g2))
		return ns


@np.vectorize
def expected_sky_count(mu_n, k):
    k = int(k)
    mu_s_list = np.zeros(k+1)
    weights_list = np.zeros(k+1)
    #mu_s_bar_0 = (-1. * mu_n + np.sqrt(mu_n**2 + 4.) )/2.
    mu_s_bar_0 = k
    # zero-count prob
    #prob_0 = (1. - mu_n/np.sqrt(mu_n**2 + 4.) )**k
    prob_0 = stats.poisson.pmf(k, mu=mu_s_bar_0) * stats.poisson.pmf(0, mu=mu_n)
    mu_s_list[0] = mu_s_bar_0
    weights_list[0] = prob_0
    for k_n in np.arange(1,k+1):
        k_n = float(k_n)
        mu_s_bar = ((k -k_n) * mu_n)/ k_n 
        #prob = (float(factorial(k)) / (factorial(k_n) * factorial(k - k_n))) * ((k-k_n)**(k-k_n)*(k_n)**(k_n)/(k**k))
        prob = stats.poisson.pmf(k-k_n, mu = mu_s_bar) * stats.poisson.pmf(k_n, mu= mu_n)
        #poisson(k-k_n, mu_s_bar) * poisson(k_n, mu_n)
        #prob = (k_n**k_n * (k - k_n)**(k - k_n))/(k**k)
        mu_s_list[int(k_n)] = mu_s_bar
        weights_list[int(k_n)] = prob
    mu_s_star = np.average(mu_s_list,weights=weights_list)
    err_mu = np.sqrt(np.cov(mu_s_list,aweights = weights_list))
    return mu_s_star, err_mu

def bkgsub(img,bkgrate,exp=None):
    # expected background count = bkg count/s * exp
    # if bkg is in counts already, exp is not needed.
    assert img.shape == bkgrate.shape
    if exp is not None:
        assert img.shape == exp.shape
    else:
        exp = np.ones_like(img)
    bkg = bkgrate * exp
    bkgflat = bkg.ravel()
    imgflat = img.ravel()
    mu_s_flat, mu_s_err_flat = expected_sky_count(bkgflat, imgflat)
    mu_s = mu_s_flat.reshape(img.shape)
    return mu_s
