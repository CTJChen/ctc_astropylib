
##work in progress
from __future__ import division, print_function
import sys
import scipy.stats.distributions as dist
import numpy as np
from astropy.cosmology import comoving_volume as vcomov
def irlf_r10(z,loglir):
   k=0.0089*(1+z)**1.1
   sigma=0.7
   lstar=1.77e9*((1+z)**2.7)
   lf=np.zeros((len(z),len(loglir)),dtype='float32')
   for i in np.arange(len(z)):
      lf[i,:]=k[i]*(((10**loglir)/lstar[i])**(1-1.2))*np.exp((-1/(2*sigma**2))*(np.log10(1+(10**loglir)/lstar[i]))**2)
   return lf

def count_lf_r10(z,loglir,area=0,zrange=[0.,0.]):
   '''This is to count how many objects are above loglir
   area and zrange must be set simultaneously
   '''
   if area > 0:
      skyfrac=float(area)/41253.
   k=0.0089*(1+z)**1.1
   sigma=0.7
   lstar=1.77e9*((1+z)**2.7)
   lf=np.zeros((len(z),len(loglir)),dtype='float32')
   for i in np.arange(len(z)):
      lf[i,:]=k[i]*(((10**loglir)/lstar[i])**(1-1.2))*np.exp((-1/(2*sigma**2))*(np.log10(1+(10**loglir)/lstar[i]))**2)
   return lf
