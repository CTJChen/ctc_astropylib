'''
For two 1-d arrays
find the arr2 indicies for the corresponding arr1 input
'''
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table as tab

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
		cols = ['id_cat1','id_cagt2','dist']
	elif len(cols)>2:
		print('only takes a list of the 2 column names, no need to name dist array')
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
