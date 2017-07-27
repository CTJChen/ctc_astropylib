print("loading numpy, pandas, plt, units as u")
print("setting cos_flat as flat LCDM, H=70, Om=0.3 cosmology")
print("usually takes a few seconds")
from ctc_observ import *
from ctc_arrays import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table as tab
from astropy.cosmology import FlatLambdaCDM
cos_flat = FlatLambdaCDM(H0=70, Om0=0.3)
dict_wav={'u':0.3543,'g':0.4770,'r':0.6231,'i':0.7625,'z':0.9134,'U':0.36,'B':0.44,
'V':0.55,'R':0.64,'I':0.79,'W1':3.368,'W2':4.618,'W3':12.082,'W4':22.194,'j':1.235,
'h':1.662,'ks':2.159,'SDSSu':0.3543,'SDSSg':0.4770,'SDSSr':0.6231,'SDSSi':0.7625,
'SDSSz':0.9134,'2MASSJ':1.235,'2MASSH':1.662,'2MASSKs':2.159}

import seaborn.apionly as sns
sns.set_style("ticks",{"xtick.direction": "in","ytick.direction": "in"})

def init_plotting(fsize=None,lw_axis=None):
    #plt.rcParams['figure.figsize'] = (8, 6)
    if fsize is None: fsize=14
    if lw_axis is None: lw_axis=1.0
    plt.rcParams['font.size'] = fsize
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.weight'] = 'medium'#'bold','medium'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 0.6*plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 0.9*plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 0.9*plt.rcParams['font.size']
    #plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = lw_axis
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.interactive(False)
