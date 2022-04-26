
import numpy as np
import urllib.request, urllib.error, urllib.parse
import urllib.request, urllib.parse, urllib.error
import os
import os.path
import argparse
import logging
import pandas as pd
from astropy.table import Table as tab
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import votable
def load_SDSS_phot_dr7(ra,dec,search_radius,pandas=None,ver=None,columns=None):
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
        values = {'cmd': sSQL_query,
                            'format': 'csv'}
        data = urllib.parse.urlencode(values)
        data = data.encode('utf-8')

        request = urllib.request.Request(sURL, data)
        response = urllib.request.urlopen(request)
        return response.read()
        
    sql_str=gen_SDSS_sql(ra,dec,search_radius)
    sdss_ds=query_SDSS(sql_str)
    sdss_ds = sdss_ds.decode('utf-8')
    
    lines=sdss_ds.split('\n')
    nobj=len(lines)-2
    if ver:print((str(nobj)+' SDSS objects found'))
    if nobj >0:
        cols=lines[0].split(',')
        #pop columnes and the EOF line
        lines.pop(0)
        lines.pop(-1)
        rows=[]
        for i in lines:
            tt=i.split(',')
            tt=list(map(float,tt))
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


def load_SDSS_phot_dr12(ra,dec,search_radius,pandas=None,ver=None,columns=None):
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
        sURL = 'http://skyserver.sdss.org/dr12/en/tools/search/x_sql.aspx?'
        # for POST request
        values = {'cmd': sSQL_query,
                            'format': 'csv'}
        data = urllib.parse.urlencode(values)
        data = data.encode('utf-8')
        request = urllib.request.Request(sURL, data)
        response = urllib.request.urlopen(request)
        return response.read()
        
    sql_str=gen_SDSS_sql(ra,dec,search_radius)
    sdss_ds=query_SDSS(sql_str)
    sdss_ds = sdss_ds.decode('utf-8')
    lines=sdss_ds.split('\n')
    nobj=len(lines)-2
    if ver:print((str(nobj)+' SDSS objects found'))
    if nobj >0:
        #pop table name and the EOF line
        lines.pop(0)
        lines.pop(-1)    
        cols=lines[0].split(',')
        #pop column
        lines.pop(0)
        rows=[]
        for i in lines:
            tt=i.split(',')
            tt=list(map(float,tt))
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


def IRSA_TAP(ra,dec,search_radius,irsa_table,pandas=False,sql_SELECT=None,sql_WHERE=None,verbose=False):
    '''
    '''
    j2000 ="'"+"J2000"+"'"
    sURL = 'http://irsa.ipac.caltech.edu/TAP/sync?QUERY='
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
    urlquery = sURL+str_select+str_from+str_where+str_coord
    if verbose:
        print(urlquery)
    response=urllib.request.urlopen(urlquery)
    if pandas:
        tab_out = votble.parse_single_table(response).to_pandas()
    else:
        tab_out = votable.parse_single_table(response)
    return tab_out
