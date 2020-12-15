from os.path import expanduser
home = expanduser("~")
import sys
sys.path.append(home+'/lib/ctc_astropylib')
print("loading numpy, pandas, plt, units as u")
print("setting cos_flat as flat LCDM, H=70, Om=0.3 cosmology")
print("usually takes a few seconds")
from ctc_observ import *
from ctc_arrays import *
#from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table as tab
from astropy.io import fits, ascii
from time import gmtime, strftime
global today
today = strftime("%Y%m%d", gmtime())


import log_progress
cos_flat = FlatLambdaCDM(H0=70, Om0=0.3)
'''
dict_wav={'u':0.3543,'g':0.4770,'r':0.6231,'i':0.7625,'z':0.9134,
'U':0.36,'B':0.44,'V':0.55,'R':0.64,'I':0.79,
'W1':3.368,'W2':4.618,'W3':12.082,'W4':22.194,
'j':1.235,'h':1.662,'ks':2.159,
'SDSSu':0.3543,'SDSSg':0.4770,'SDSSr':0.6231,'SDSSi':0.7625,'SDSSz':0.9134,
'2MASSJ':1.235,'2MASSH':1.662,'2MASSKs':2.159} 
'''
print('matplotlib/seaborn environment settings : default init_plotting(fsize=14, lw_axis=1.0)')
import seaborn.apionly as sns
sns.set_style("ticks",{"xtick.direction": "in","ytick.direction": "in"})
from pylab import *
import matplotlib.pyplot as plt
from ctc_plots import *
init_plotting(fsize=14, lw_axis=1.0)

        