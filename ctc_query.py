
import numpy as np
import os
import os.path
import argparse
import logging
import pandas as pd
from astropy.table import Table as tab
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import votable
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
import requests


def load_SDSS_phot_dr7(ra,dec,search_radius,pandas=True,ver=None,columns=None):
    '''
    ra in degrees, dec in degrees, search_radius in arcmin.
    '''
    def gen_SDSS_sql(ra, dec, search_radius,columns=columns):
        if columns is None:
            query_out  = ' p.*'
        else:
            query_out = columns+',p.ra,p.dec'
        query_from = ' FROM fGetNearbyObjEq({ra},{dec},{search_radius}) n, PhotoPrimary p WHERE n.objID=p.objID'.format(ra=str(ra),dec=str(dec),search_radius=str(search_radius))
        query_str='SELECT'+query_out+query_from
        return query_str 

    def query_SDSS(sSQL_query):
        sURL = 'http://cas.sdss.org/dr7/en/tools/search/x_sql.asp'
        # for POST request
        values = {'cmd': sSQL_query,'format': 'csv'}
        r = requests.get(sURL,params=values)
        return r.text
    
    sql_str=gen_SDSS_sql(ra,dec,search_radius)
    sdss_ds=query_SDSS(sql_str)
    csvtxt = StringIO(sdss_ds)
    out = pd.DataFrame.from_csv(csvtxt)
    if len(out) == 0:
        print('No objects have been found for RA='+str(ra)+', DEC='+str(dec)+' within '+str(search_radius)+' arcmin')
        return out
    elif not pandas:
        radec=coord.SkyCoord(ra,dec,unit=(u.degree,u.degree))
        sdss=coord.SkyCoord(out.ra, out.dec, unit=(u.degree,u.degree))    
        out['dist_arcsec'] = radec.separation(sdss).arcsec
        out = tab.from_pandas(out)
        return out
    else:
        radec=coord.SkyCoord(ra,dec,unit=(u.degree,u.degree))
        sdss=coord.SkyCoord(out.ra, out.dec, unit=(u.degree,u.degree))    
        out['dist_arcsec'] = radec.separation(sdss).arcsec        
        return out


def load_SDSS_phot_dr12(ra,dec,search_radius,pandas=True,ver=None,columns=None):
    '''
    ra in degrees, dec in degrees, search_radius in arcmin.
    '''
    def gen_SDSS_sql(ra, dec, search_radius,columns=columns):
        if columns is None:
            query_out  = ' p.*'
        else:
            query_out = columns+',p.ra,p.dec'
        query_from = ' FROM fGetNearbyObjEq({ra},{dec},{search_radius}) n, PhotoPrimary p WHERE n.objID=p.objID'.format(ra=str(ra),dec=str(dec),search_radius=str(search_radius))
        query_str='SELECT'+query_out+query_from
        return query_str 

    def query_SDSS(sSQL_query):
        sURL = 'http://skyserver.sdss.org/dr12/en/tools/search/x_sql.aspx'
        # for POST request
        values = {'cmd': sSQL_query,'format': 'csv'}
        r = requests.get(sURL,params=values)
        return r.text
    
    sql_str=gen_SDSS_sql(ra,dec,search_radius)
    sdss_ds=query_SDSS(sql_str)
    csvtxt = StringIO(sdss_ds)
    out = pd.read_csv(csvtxt,comment='#')
    if len(out) == 0:
        print('No objects have been found for RA='+str(ra)+', DEC='+str(dec)+' within '+str(search_radius)+' arcmin')
        return out
    elif not pandas:
        radec=coord.SkyCoord(ra,dec,unit=(u.degree,u.degree))
        sdss=coord.SkyCoord(out.ra, out.dec, unit=(u.degree,u.degree))    
        out['dist_arcsec'] = radec.separation(sdss).arcsec        
        out = tab.from_pandas(out)
        return out
    else:
        radec=coord.SkyCoord(ra,dec,unit=(u.degree,u.degree))
        sdss=coord.SkyCoord(out.ra, out.dec, unit=(u.degree,u.degree))    
        out['dist_arcsec'] = radec.separation(sdss).arcsec        
        return out


def IRSA_TAP(ra,dec,search_radius,irsa_table,pandas=True,sql_SELECT=None,sql_WHERE=None,verbose=False):
    '''
    Querying the designated IRSA table using the native IRSA API
    This would result in tables with all of the columns available.
    Use astroquery if you're only interested in typical queries.
    input: ra, dec(J2000) & search_radius (deg)
    Some of the most commonly used tables are:
    wise_allwise_p3as_psd (ALL WISE)
    fp_xsc (2MASS extended source catalog)
    fp_psc (2MASS point source catalog)
    '''
    j2000 ="'"+"J2000"+"'"
    sURL = 'http://irsa.ipac.caltech.edu/TAP/sync'
    str_from = 'FROM+'+irsa_table+'+'
    if sql_SELECT == None :
        str_select = 'SELECT+*+' #select all
    else:
        str_select = 'SELECT+'+sql_SELECT+'+'
    if sql_WHERE == None :
        str_where = 'WHERE+'
    else:
        str_where = 'WHERE+'+sql_WHERE+'+'
    str_coord1 = 'CONTAINS(POINT('+j2000+',ra,dec),'
    str_coord2 = 'CIRCLE('+j2000+','+str(ra)+','+str(dec)+','+str(search_radius)+'))=1'
    str_coord=str_coord1+str_coord2
    str_sql = 'QUERY='+str_select+str_from+str_where+str_coord+'&FORMAT=CSV'
    r = requests.get(sURL,params=str_sql)
    if verbose:print(r.url)
    csvtxt = StringIO(r.text)
    out = pd.read_csv(csvtxt,comment='#')
    if len(out) == 0:
        print('No objects have been found for RA='+str(ra)+', DEC='+str(dec)+' within '+str(search_radius)+' arcmin')
        return out
    elif not pandas:
        out = tab.from_pandas(out)
        return out
    else:
        return out
    if verbose:
        print(urlquery)
    response=urllib2.urlopen(urlquery)
    if pandas:
        tab_out = votable.parse_single_table(response).to_table().to_pandas()
    else:
        tab_out = votable.parse_single_table(response).to_table()
    return tab_out
