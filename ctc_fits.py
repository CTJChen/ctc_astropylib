import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table as tab
from astropy.nddata import Cutout2D
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from ctc_observ import makecd
import copy


def readfits(fname):
	return tab(fits.getdata(fname)).to_pandas()

def makecutout(ra, dec, size, image=None, header=None, filename=None, savefig=None):
    '''
    make image cutouts centered at ra, dec (in degrees), with a box of size.
    Variables: 
    --ra : scalar, should be in degree
    --dec : scalar, should be in degree
    --size : scalar tuple --> pixles; can also be a tuple with astropy angular units
    --filename : file name of the fits file to be cropped
    --image and header : if the fits file is not a single extension image, 
    it might be necessary to manually input image and header from the fits file.
    --savefig : should be a string of the cropped fits filename to be saved
    '''
    if filename is not None:
        hdu = fits.open(filename)
        cutoutimage = hdu[0].data
        cutoutheader = hdu['PRIMARY'].header
    else:
        cutoutimage = copy.deepcopy(image)
        cutoutheader = copy.deepcopy(header)
    fitswcs = wcs.WCS(cutoutheader)
    position = makecd(ra, dec)#, frame='icrs')
    try:
        cutout2 = Cutout2D(cutoutimage, position, size, wcs=fitswcs)
        wcs_cropped = cutout2.wcs
        cutoutheader.update(wcs_cropped.to_header())
        if savefig is not None:
            hdu = fits.PrimaryHDU(cutout2.data, header=cutoutheader)
            hdu.writeto(savefig, overwrite=True)
    except ValueError:
        print('Out of bound')
        cutout2 = 0
    return cutout2
