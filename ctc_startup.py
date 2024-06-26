from os.path import expanduser
home = expanduser("~")
import sys
sys.path.append(home+'/lib/ctc_astropylib')
from tqdm.notebook import tqdm as log_progress
from time import gmtime, strftime
global today
today = strftime("%Y%m%d", gmtime())
print("importing gmtime and strftime, setting today="+today)
print("\nloading numpy, pandas as np and pd, set pandas display default values")
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print("\nloading astropy functions: units as u, io.fits, io.ascii, table.Table as tab")
from astropy import units as u
from astropy.table import Table as tab
from astropy.io import fits, ascii
print("setting cos_flat as flat LCDM, H=70, Om=0.3 cosmology")
print("usually takes a few seconds")
from astropy.cosmology import FlatLambdaCDM
cos_flat = FlatLambdaCDM(H0=70, Om0=0.3)
print("\nimporting custom python scripts in ctc_astropylib...")
from ctc_observ import *
from ctc_arrays import *


'''
dict_wav={'u':0.3543,'g':0.4770,'r':0.6231,'i':0.7625,'z':0.9134,
'U':0.36,'B':0.44,'V':0.55,'R':0.64,'I':0.79,
'W1':3.368,'W2':4.618,'W3':12.082,'W4':22.194,
'j':1.235,'h':1.662,'ks':2.159,
'SDSSu':0.3543,'SDSSg':0.4770,'SDSSr':0.6231,'SDSSi':0.7625,'SDSSz':0.9134,
'2MASSJ':1.235,'2MASSH':1.662,'2MASSKs':2.159} 
import seaborn as sns
sns.set_style("ticks",{"xtick.direction": "in","ytick.direction": "in"})
from pylab import *
'''
print('\nimporting matplotlib and other plotting scripts,\
ctc_plots default init_plotting(fsize=14, lw_axis=1.0)')
import matplotlib.pyplot as plt
from ctc_plots import *
init_plotting(fsize=14, lw_axis=1.0)
from matplotlib.colors import LogNorm