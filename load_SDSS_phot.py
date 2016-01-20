
import numpy as np
import urllib2
import urllib
import os
import os.path
import argparse
import logging
import pandas as pd
from astropy.table import Table as tab
from astropy import coordinates as coord
from astropy import units as u

def load_SDSS_phot_dr7(ra,dec,search_radius,pandas=None,ver=None):
    '''
    ra in degrees, dec in degrees, search_radius in arcmin.
    '''
    def gen_SDSS_sql(ra, dec, search_radius):
        query_out  = ' p.objID, p.type, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.Err_u, p.Err_g, p.Err_r, p.Err_i, p.Err_z, p.extinction_u, p.extinction_i, p.extinction_r, p.extinction_g, p.extinction_z, p.petroMag_u, p.petroMag_g, p.petroMag_r, p.petroMag_i, p.petroMag_z, petroMagErr_u, petroMagErr_g, petroMagErr_r, petroMagErr_i, petroMagErr_z'
        query_from = ' FROM fGetNearbyObjEq({ra},{dec},{search_radius}) n, PhotoPrimary p WHERE n.objID=p.objID'.format(ra=str(ra),dec=str(dec),search_radius=str(search_radius))
        query_str='SELECT'+query_out+query_from
        return query_str 

    def query_SDSS(sSQL_query):
    	sURL = 'http://cas.sdss.org/dr7/en/tools/search/x_sql.asp'
    	# for POST request
    	values = {'cmd': sSQL_query,
    						'format': 'csv'}
    	data = urllib.urlencode(values)
    	request = urllib2.Request(sURL, data)
    	response = urllib2.urlopen(request)
    	return response.read()
    	
    sql_str=gen_SDSS_sql(ra,dec,search_radius)
    sdss_ds=query_SDSS(sql_str)
    lines=sdss_ds.split('\n')
    nobj=len(lines)-2
    if ver:print(str(nobj)+' SDSS objects found')
    if nobj >0:
        cols=lines[0].split(',')
        #pop columnes and the EOF line
        lines.pop(0)
        lines.pop(-1)
        rows=[]
        for i in lines:
            tt=i.split(',')
            tt=map(float,tt)
            rows.append(tt)
        tab_out=tab(rows=rows,names=cols)
        if pandas:
            tab_out=pd.DataFrame.from_records(tab_out._data)
            radec=coord.SkyCoord(ra,dec,unit=(u.degree,u.degree),frame='icrs')
            sdss=coord.SkyCoord(tab_out.ra, tab_out.dec, unit=(u.degree,u.degree),frame='icrs')
            tab_out['dist_arcsec'] = radec.separation(sdss).arcsec
        #should fix objID to string
        #should find out what p.type means
        return tab_out
    else: 
        return []


def load_SDSS_phot_dr12(ra,dec,search_radius,pandas=None, limit=None):
    '''
    ra in degrees, dec in degrees, search_radius in arcmin.
    '''
    #creat url
    sURL='http://skyserver.sdss.org/dr12/en/tools/search/x_radial.aspx?'
    ra_str='ra='+str(ra)+'&'
    dec_str='dec='+str(dec)+'&'
    radius_str='radius='+str(search_radius)+'&'
    if not limit: limit=20
    limit_str='format=csv&limit='+str(limit)
    # for POST request
    urlquery=sURL+ra_str+dec_str+radius_str+limit_str
    response=urllib2.urlopen(urlquery)
    if pandas:
        tab_out=pd.read_csv(response,comment='#')
    else:
        tab_out=ascii.read(response, delimiter=',')
    #should fix objID to string
    #should find out what p.type means
    return tab_out
