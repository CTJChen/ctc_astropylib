# -*- coding: utf-8 -*-
"""
Created on 11.2016
@author: Chien-Ting J. Chen
Based on the suggestions made by Piero Ranalli
(https://github.com/piero-ranalli)
----------------------------------
Reproject XMM event files and attitude files
onto a shifted RA/DEC
For event files,
the following values will be updated
using the new RA/DEC input:
TCRVLi, TCRVLj, REFXCRVL, REFYCRVL
(i and j are the numbers which identify the RA and DEC groups of keywords)
RA_NOM, DEC_NOM, RA_PNT and DEC_PNT
For attitude files:
1. the AAHFRA, MAHFRA, AAHFDEC and MHFDEC keywords in the primary image (extension #0)
2. the AHFRA and AHFDEC *columns* in extension #1.
"""
import numpy as np
import sys
import yaml
from astropy.io import fits
import astropy.coordinates as cd

documents = '''
obsids : 'obsids.txt'
path_lv1  : '/Volumes/razorback3/cuc36/'
path_reproj : '/Volumes/razorback3/cuc36/reproj'
'''
parms = yaml.load(documents)

## Or define the parameteres in a file, and load them with
# stream = file('document.yaml', 'r')
# params = yaml.load(stream)

def makeProjection(hdr):
    """Return a wcs projection object based on the XMM event header keys."""
    wcskeys = {
        'NAXIS': 2,
        'CTYPE1': hdr['REFXCTYP'],
        'CRPIX1': hdr['REFXCRPX'],
        'CRVAL1': hdr['REFXCRVL'],
        'CDELT1': hdr['REFXCDLT'],
        'CUNIT1': hdr['REFXCUNI'],
        'CTYPE2': hdr['REFYCTYP'],
        'CRPIX2': hdr['REFYCRPX'],
        'CRVAL2': hdr['REFYCRVL'],
        'CDELT2': hdr['REFYCDLT'],
        'CUNIT2': hdr['REFYCUNI'],
    }

    return wcs.Projection(wcskeys)

def getColumnNames(hdu):
    """Get names of columns in HDU."""
    return [d.name for d in hdu.get_coldefs()]

def reproject(inevt, matchevt, outevt):
    """Reproject input event file (filename inevt) to match event file
    matchevt. Writes output file outevt."""

    # read inputs
    fmatch = pyfits.open(matchevt)
    hdrmatch = fmatch['events'].header
    projmatch = makeProjection(hdrmatch)

    fin = pyfits.open(inevt)
    hdrin = fin['events'].header
    projin = makeProjection(hdrin)

    # turn coordinates into an array of (x,y) point values
    xin = fin['events'].data.field('x')
    yin = fin['events'].data.field('y')
    coord = N.hstack((xin.reshape(-1,1), yin.reshape(-1,1)))

    # turn pixel coordinates into world coordinates
    inworld = projin.toworld(coord)
    # turn back into pixel coordinates for new wcs system
    matchpixel = projmatch.topixel(inworld)
    intpixel = matchpixel.round().astype('int')

    # put new coordinates back into file
    newxvals = intpixel[:,0]
    fin['events'].data.field('x')[:] = newxvals
    newyvals = intpixel[:,1]
    fin['events'].data.field('y')[:] = newyvals

    # copy headers from match file
    for h in ('REFXCRPX', 'REFXCRVL', 'REFXCDLT', 'REFXCUNI', 'REFXCTYP',
              'REFYCRPX', 'REFYCRVL', 'REFYCDLT', 'REFYCUNI', 'REFYCTYP'):
        hdrin.update(h, hdrmatch[h])

    # copy column information for X and Y - seems to be duplicated here
    columns = getColumnNames(fin['events'])
    xcol, ycol = columns.index('X')+1, columns.index('Y')+1
    for col in xcol, ycol:
        for h in ('TCRPX%i' % col, 'TCRVL%i' % col, 'TCDLT%i' % col,
                  'TCUNI%i' % col, 'TCTYP%i' % col):
            hdrin.update(h, hdrmatch[h])

    # update data ranges (is this required?)
    hdrin.update('TDMAX%i' % xcol, newxvals.max())
    hdrin.update('TDMIN%i' % xcol, newxvals.min())
    hdrin.update('TDMAX%i' % ycol, newyvals.max())
    hdrin.update('TDMIN%i' % ycol, newyvals.min())

    # write as output file
    fin.writeto(outevt)
