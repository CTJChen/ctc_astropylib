print("loading numpy, pandas, plt, units as u")
print("setting cos_flat as flat LCDM, H=70, Om=0.3 cosmology")
print("usually takes a few seconds")

from chen_observ import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cos_flat = FlatLambdaCDM(H0=70, Om0=0.3)

