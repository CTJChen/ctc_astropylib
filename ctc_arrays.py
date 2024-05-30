'''
For two 1-d arrays
find the arr2 indicies for the corresponding arr1 input
'''
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table as tab
import astropy.coordinates as cd

#import statsmodels.nonparametric.api as smnp

def linspacearr(arr,nbins):
	'''
	Return a np.linspace array with min/max of arr and nbins
	'''
	return np.linspace(np.min(arr),np.max(arr),nbins)

def readfits(fname):
	return tab(fits.getdata(fname)).to_pandas()

def awhere(arr1,arr2):
	'''
	This only works if arr1 is a subset of arr2
	'''
	#sort arr1
	id_sort = np.argsort(arr2)
	sorted_arr2 = arr2[id_sort]
	sorted_id = np.searchsorted(sorted_arr2, arr1)
	sorted_id[arr2[sorted_id] != arr1] = -1
	return sorted_id

def match_unique(id_cat1,id_cat2,d2d,d3d=None,cols=None):
	'''
	Takes the output of astropy coordinates.Skycoord.search_around_sky
	Find if there's any duplicated match
	And only return the object that's matched within the minimal distance
	Default : d3d = None
	cols = None
	The astropy's search algorithm
	will contain no duplicates for id_cat2, so that should be used as the index
	'''
	#use 3-d distance when possible
	if d3d is not None:
		dist = d3d
	else:
		dist = d2d
	#use user defined column names when applicable
	if cols is None:
		cols = ['id_cat1','id_cat2','dist']
	elif len(cols)>2:
		print('only takes the names of the two catalogs, no need to name dist arrays')
		cols = cols[0:2]
		cols.append('dist')
	else:
		cols.append('dist')
	df = pd.DataFrame({cols[0]:id_cat1,cols[1]:id_cat2,cols[2]:dist})
	grp = df.groupby(cols[0])
	df_out = grp.apply(lambda t:t[t.dist == t.dist.min()] if sum(t.dist==t.dist.min()) == 1 else t[t[cols[1]] == t[cols[1]].min()])
	df_out['nmatch'] = grp.count().dist.values
	return df_out


def binvalue(bins):
	binv = [0.5*(i[0]+i[1]) for i in zip(bins,bins[1:])]
	return binv

'''
def kde1d(arr,kernel='gau', bw=0.2):
'''
#use statsmodels.nonparametric.api to do density estimation
'''
	kde = smnp.KDEUnivariate(arr)
	kde.fit(bw=bw,kernel=kernel)
	xgrid = np.linspace(min(arr),max(arr),100)
	dist_kde = kde.evaluate(xgrid)
	return xgrid, dist_kde
'''

def sampdist(df1, df2, sampcol,bins=10):
    '''
    Return a DF that's randomly sampled from df2, according
    to the given distribution of sampcol in df1
    bins could be the number of bins (int), or the bins (array/list/tuple) to be used
    '''
    try:
        out, sbin = pd.cut(df1[sampcol],bins = bins,retbins=True)
        grp = df1.groupby(out)
        nsrcs = grp[sampcol].count().values
        arr_id2 = np.array([])
        for idx in range(len(sbin) - 1):
            s0 = sbin[idx]
            s1 = sbin[idx+1]
            nbin = nsrcs[idx]
            if (nbin > 0) & (sum((df2[sampcol] >= s0) & (df2[sampcol] < s1)) > 0):
                id2 = df2[(df2[sampcol] >= s0) & (df2[sampcol] < s1)].sample(nbin,replace=True).index.values
                arr_id2 = np.hstack((arr_id2,id2))
        return arr_id2
    except (~sampcol in df1.columns) | (~sampcol in df2.columns):
        print(sampcol, ' must be in both input dataframes')

def newcol(df, colname):
	return pd.Series(np.zeros(len(xdfall))+np.nan, index=df.index)

@np.vectorize
def distnn(ra, dec):
    '''
    input RA/DECunits = deg, deg
    '''
    from sklearn.neighbors import NearestNeighbors
    catc = cd.SkyCoord(ra, dec,unit=(u.deg,u.deg))
    X = np.vstack((catc.ra.deg,catc.dec.deg)).transpose()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    distarcsec = distances[:,1]*3600. #return distance in arcsec
    nnidx = indices[:,1]
    return distarcsec, nnidx

