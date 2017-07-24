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
#simid = os.environ['simid']
#home = '/Users/cuc36'
__doc__ = " "


def ratetoflux(row,band='soft',err=False,ind=1.):
    '''
    To be applied to XMM EMLDETECT product
    compute error-weighted fluxes (among three EPIC detectors)

    '''
    if band == 'soft':
        ecf = [1e-11/7.45, 1e-11/2.0]
    elif band == 'hard':
        ecf = [1e-11/1.22, 1e-11/0.45]
    elif band == 'full':
        ecf = [1e-11/3.26, 1e-11/0.97]
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
