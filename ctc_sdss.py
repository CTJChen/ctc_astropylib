import numpy as np
import pandas as pd

def maggies2mag(maggies, ivarmaggies=None, nsigma=None, magerr=False, lomagerr=False, himagerr=False, magnsigmag=False, nano=True):
  '''
  Modified from J. Moustakas MAGGIES2MAG idl routine at 
  https://github.com/moustakas/impro
; OPTIONAL INPUTS: 
;   ivarmaggies - corresponding inverse variance maggies [NBAND,NOBJ] 
;   nsigma - NSIGMA upper limit for measurements measured at <NSIGMA
;     significance (default 2)
;
; OUTPUTS: 
;   mag - output AB magnitudes [NBAND,NOBJ]
;
; OPTIONAL OUTPUTS:
;   magerr - approximate 1-sigma error (assumes the error is symmetric
;     in magnitude) [NBAND,NOBJ] 
;   lomagerr - "lower" magnitude error [NBAND,NOBJ] 
;   himagerr - "upper" magnitude error [NBAND,NOBJ] 
;   magnsigmag - NSIGMA magnitude limit

  '''

  mag = np.zeros(len(maggies),dtype=float)
  if ivarmaggies != None:
    if nsigma == None: nsigma = 2.0
    magerr = np.zeros(len(ivarmaggies),dtype=float)
    lomagerr0 = np.zeros(len(ivarmaggies),dtype=float)
    himagerr0 = np.zeros(len(ivarmaggies),dtype=float)
    magnsigmag0 = np.zeros(len(ivarmaggies),dtype=float)

    snr = 1e9*maggies*np.sqrt(ivarmaggies*1.) # fractional error
    if not nano: snr = maggies*np.sqrt(ivarmaggies*1.)
    good = snr >= nsigma
    ngood = np.sum(good)
    print ngood, 'good sources'
    upper = (ivarmaggies != 0.0) & (snr < nsigma)
    nupper = np.sum(upper)
    nodata = ivarmaggies == 0.0
    nnodata = np.sum(nodata)
#      upper = where((maggies ne 0.0) and (ivarmaggies ne 0.0) and snr lt nsigma,nupper)
#      nodata = where((maggies eq 0.0) and (ivarmaggies eq 0.0),nnodata)

# significant detections
    if (ngood != 0):
      mag[good] = -2.5*np.log10(maggies[good])
      mag[np.invert(good)]=np.nan
      fracerr = 1.0/snr[good]
      magerr[good] = 2.5*np.log10(np.exp(1))*fracerr # Symmetric in magnitude (approximation)
      magerr[np.invert(good)] = np.nan
      lomagerr0[good] = 2.5*np.log10(1+fracerr)    # Bright end (flux upper limit)
      lomagerr0[np.invert(good)] = np.nan
      himagerr0[good] = 2.5*np.log10(1-fracerr)    # Faint end  (flux lower limit)
      himagerr0[np.invert(good)] = np.nan
# NSIGMA upper limits
    if (nupper != 0):
      magnsigmag0[upper] = 2.5*np.log10(np.sqrt(ivarmaggies[upper])/nsigma) # note "+" instead of 1/ivar
      magnsigmag0[np.invert(upper)] = np.nan
    mag = pd.DataFrame({'mag':mag})
    mag['lomagerr'] = lomagerr0
    mag['himagerr'] = himagerr0
    mag['magnsigmag'] = magnsigmag0
  else:
    good = maggies > 0.0
    ngood = np.sum(good)
    if (ngood != 0): 
      mag[good] = 2.5*(-1)*np.log10(maggies[good])
      mag[np.invert(good)] = np.nan
    else:mag[:]=np.nan

#   if (ngood ne 0L) then magerr[good] = 1.0/(0.4*np.sqrt(ivarmaggies[good])*maggies[good]*alog(10.0))

  return mag
 