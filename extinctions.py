
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpol
from scipy.io import readsav

def ext_smc_mky(wav,ebv):
    path='/Users/ctchen/analysis/SED/templates/'
    wav_ext,kappa_ext=np.loadtxt(path+'ext_assef_kappa.dat',unpack=True)
    interpfunc=interpol(wav_ext,kappa_ext)
    kappa=interpfunc(wav)
    kappa[np.where(kappa < 0)] = 0.
    ext=10.**(-kappa*ebv)
    ext[np.where(ext < 1e-10)]==0
    return ext


def ext_d03(wav,ebv):
    ext=readsav('/Users/ctchen/idl/decompqso/draine03_alt.idl')
    RV=3.1
    AL=RV*ebv*1.87e21*1.086*ext['dd_ext']
    ext=10.**(-0.4*AL)
    return ext
