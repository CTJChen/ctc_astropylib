import numpy as np
import pandas as pd
from os.path import expanduser
home = expanduser("~")

def sdss_resp(wav, extended = False, z= None, nowav = False):
	'''
	Interpolate the SDSS onto the wavlengths given
	and return a pandas DF
	'''
	bands=['u','g','r','i','z']
	output = pd.DataFrame({'wav':wav})
	sdsspath = home+'/idl/cal/sdss/'
	for i in bands:
		fname = sdsspath+i+'.dat'
		band = pd.read_csv(fname, sep='   ', comment = '#',\
			   names = ['wav_ang','resp_ps','resp_es','resp_es_na','atm_ext'])
		wav_micron = band.wav_ang/1e4
		resp = band.resp_ps
		if extended: resp=band.resp_es
		output[i] = np.interp(wav, wav_micron, resp)
	if nowav: output.drop('wav', axis=1, inplace=True)
	return output


def wise_resp(wav, z= None, nowav = False):
	'''
	Interpolate the WISE onto the wavlengths given
	and return a pandas DF
	'''
	bands=['W1','W2','W3','W4']
	output = pd.DataFrame({'wav':wav})
	wisepath = home+'/idl/cal/WISE/RSR-'
	for i in bands:
		fname = wisepath+i+'.EE.txt'
		band = pd.read_fwf(fname, comment = '#', colspecs = [(0,6),(8,20)],\
			   names = ['wav_micron','resp'])
		wav_micron = band.wav_micron
		resp = band.resp
		output[i] = np.interp(wav, wav_micron, resp)
	if nowav: output.drop('wav', axis=1, inplace=True)
	return output

def nir_2mass_resp(wav, z = None, nowav = False):
	'''
	Interpolate the 2MASS onto the wavlengths given
	and return a pandas DF
	'''
	bands=['J','H','Ks']
	output = pd.DataFrame({'wav':wav})
	wisepath = home+'/idl/cal/2MASS/resp_2mass_'
	for i in bands:
		fname = wisepath+i+'.txt'
		band = pd.read_csv(fname, comment = '#', sep='   ',\
			   names = ['wav_micron','resp'])
		wav_micron = band.wav_micron
		resp = band.resp
		output[i] = np.interp(wav, wav_micron, resp)
	if nowav: output.drop('wav', axis=1, inplace=True)
	return output

def bess_resp(wav, z = None, nowav = False):
	'''
	Convolve the input wavelengths with the Bessell's UBVRI filters
	obtained from http://spiff.rit.edu/classes/phys440/lectures/filters/filters.html
	return : a pandas DF with response curves
	'''
	bands=['bess_u','bess_b','bess_v','bess_r','bess_i']
	output = pd.DataFrame({'wav':wav})
	besspath = home+'/idl/cal/bessell/'
	for i in bands:
		fname = besspath+i+'.pass'
		band = pd.read_csv(fname, comment = '#', sep='\t',\
			   names = ['wav_ang','resp'])
		wav_micron = band.wav_ang.values/1e4
		resp = band.resp
		output[i] = np.interp(wav, wav_micron, resp)
	if nowav: output.drop('wav', axis=1, inplace=True)
	return output



def galex_resp(wav, z = None, nowav = False):
	'''
	Convolve the input wavelengths with the Bessell's UBVRI filters
	obtained from http://spiff.rit.edu/classes/phys440/lectures/filters/filters.html
	return : a pandas DF with response curves
	'''
	bands=['galex_fuv','galex_nuv']
	output = pd.DataFrame({'wav':wav})
	galexpath = home+'/idl/cal/galex/'
	for i in bands:
		fname = galexpath+i+'.dat'
		band = pd.read_csv(fname, comment = '#', sep=' ',\
			   names = ['wav_ang','resp'])
		wav_micron = band.wav_ang.values/1e4
		resp = band.resp
		output[i] = np.interp(wav, wav_micron, resp)
	if nowav: output.drop('wav', axis=1, inplace=True)
	return output

