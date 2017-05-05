import pandas as pd
from astropy.io import fits
import astropy.coordinates as cd
from ctc_observ import *
from ctc_arrays import *
from scipy.interpolate import interp1d

#load HSC catalog first
#hsc = pd.read_csv('/cuc36/xxl/multiwavelength/HSC/wide.csv')

def pdf_sep_gen(sep_arcsec,xposerr,opterr):
	'''
	PDF of angular separation between an X-ray object and the other input catalog
	with positional error poserr
	'''
	poserr=2*(opterr**2+xposerr**2)# this is 2*sigma^2
	return np.exp(-sep_arcsec**2/poserr)/(np.pi*poserr)

def getbkgcat(xcat,catopt,optdf,r_in = 7., r_out=35.,\
		  nmagbin=15, magname = 'rmag_psf', corr_glob=True):
    '''
    Takes in xcat and catopt,
    find optical sources with separation from any x-ray sources
    between r_in and r_out (in arcsec),
    and derive the magnitude dependence of these background sources
    optdf = optdf_in.copy()
    optdf.reset_index(inplace=True)
    if len(catopt) != len(optdf):
    	print("catopt should be the astropy coordinate object computed from optdf!")
    	sys.exit(1)
    '''
    idhsc,idxmm,d2d,d3d=xcat.search_around_sky(catopt,r_in*u.arcsec)
    #Excluding each optical source with an x-ray source within r_in
    itmp=np.arange(len(catopt))
    itmp[np.unique(idhsc)]=-1
    #indicies for optical sources with **NO** X-ray counterparts within r_in
    idhsc_ext=np.where(np.equal(optdf.index.values, itmp))[0]

    #Now search for X-ray and optical matches within r_out
    idhsc_in,idxmm,d2d,d3d=xcat.search_around_sky(catopt,r_out*u.arcsec)
    idhsc_in = np.unique(idhsc_in)
    #Cross-correlated the ``no r_in list'', and the ``r_out list''
    #This will create a list of ``background optical sources''
    idhsc_bkgd=np.intersect1d(idhsc_ext,idhsc_in)
    hsc_bkgd=optdf.loc[idhsc_bkgd].copy()
    hsc_bkgd.reset_index(inplace=True)
    #
    out,rmagbin=pd.cut(hsc_bkgd[magname].values,bins=nmagbin,retbins=True)
    groups=hsc_bkgd.groupby(out)
    #number density = total number of sources divided by the area of annulus
    N_xmm=len(xcat) #number of unique XMM sources
    N_bkgd=len(hsc_bkgd)
    nm=groups.ra.count().values/(np.pi*(r_out**2-r_in**2)*N_xmm)
    if corr_glob:
        #According to Brusa et al. 2007, at faint magnitudes
        #nm is not correct and should use a global one.
        out,rmagbin_global=pd.cut(optdf[magname].values,bins=nmagbin,retbins=True)
        groups=optdf.groupby(out)
        rmag_global = binvalue(rmagbin_global)
        area = \
        (optdf.ra.max() - optdf.ra.min())*(optdf.dec.max() - optdf.dec.min())*3600**2
        nm_global = groups.ra.count().values/area
        iglobal = np.where(rmagbin > 23.)[0][:-1]
        nm[iglobal] = nm_global[iglobal]
    return nm,rmagbin

#def getqm(match,rmagbin, Q, NX, nm, r0=2.5):
def getqm(match,rmagbin, Q):
	'''
    Estimate q(m) -- the expected optical counterpart magnitude
    distribution of at magintude m
	'''
	grp=match.groupby(pd.cut(match['rmag'].values,bins=rmagbin))
	real_m=grp.rax.count().values# - np.pi*r0**2*NX*nm
	qm = real_m*(Q/np.sum(real_m))
	return qm, Q


def calc_RC(match, quntarr, Q,lxcat,LRfrac=0.2):
	'''
	If quntarr is an array with length > 1 (and values between 0 to 1)
	This subroutine finds the LRth value that maximize C and R.
	If quntarr is a single value,
	a array with correctly matched sources would be returned
	'''
	if type(lxcat) != float:
		lxcat = float(lxcat)
	if np.isscalar(quntarr):
		LRth = quntarr
		tmp = match[match.LR > LRth].copy()
		grp = tmp.groupby('xid')
		#select sources with only one match
		onematch = grp.filter(lambda x: len(x) == 1).copy()
		onematch.reset_index(inplace=True)
		onematch['Rc'] = onematch.LR.values/(onematch.LR.values + 1 - Q)
		#these are sources with multiple matches
		multimatch = grp.filter(lambda x: len(x) > 1).copy()
		#regroup, and for each group only keep sources with LR larger than 0.2*max(LR)
		grp = multimatch.groupby('xid')
		igood = grp.apply(lambda df:df.LR/df.LR.max() >= LRfrac).values
		multimatch = multimatch[igood]
		multimatch.reset_index(inplace=True)
		grp = multimatch.groupby('xid')
		multiRc = grp.apply(lambda df: df.LR/(df.LR.sum()+(1-Q))).values
		multimatch['Rc'] = multiRc
		allmatch = pd.concat([onematch,multimatch])
		R = allmatch.Rc.mean()
		C = allmatch.Rc.sum()/lxcat
		return allmatch, R, C
	else:
		R = np.zeros(len(quntarr))
		C = np.zeros(len(quntarr))
		LRth = np.zeros(len(quntarr))
		for index, i in enumerate(quntarr):
			LRth[index] = match.LR.quantile(i)
			tmp = match[match.LR > LRth[index]].copy()
			grp = tmp.groupby('xid')
			#select sources with only one match
			onematch = grp.filter(lambda x: len(x) == 1).copy()
			onematch.reset_index(inplace=True)
			onematch['Rc'] = onematch.LR.values/(onematch.LR.values + 1 - Q)
			#these are sources with multiple matches
			multimatch = grp.filter(lambda x: len(x) > 1).copy()
			#regroup, and for each group only keep sources with LR larger than 0.2*max(LR)
			grp = multimatch.groupby('xid')
			igood = grp.apply(lambda df:df.LR/df.LR.max() >= LRfrac).values
			if np.sum(igood) > 0:
				multimatch = multimatch[igood]
				if multimatch.xid.nunique() > 1:
					grp = multimatch.groupby('xid')
					multiRc = grp.apply(lambda df: df.LR/(df.LR.sum()+(1-Q))).values
					multimatch['Rc'] = multiRc
					multimatch.reset_index(inplace=True)
					allmatch = pd.concat([onematch,multimatch])
					R[index] = allmatch.Rc.mean()
					C[index] = allmatch.Rc.sum()/lxcat
				else:
					multimatch = multimatch[igood]
					multimatch['Rc'] = multimatch.LR/(multimatch.LR.sum()+(1-Q))
					allmatch = pd.concat([onematch,multimatch])
			else:
				allmatch = onematch
				R[index] = -1.
				C[index] = -1.
		return R, C, LRth
		'''
		func = interp1d(quntarr, T, bounds_error=False,fill_value='extrapolate')
		lthmax = np.linspace(0.,1.,1000)[np.where(func(np.linspace(0.,1.,1000)) ==
		max(func(np.linspace(0.,1.,1000))))]
		return R, C, lthmax
		'''

def calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, rsearch=5.0,\
			lth = np.linspace(0.05,0.9,10), LRfrac=0.2,lrmax=None,\
			magname = 'rmag_psf',xerrname='xposerr',
            xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',
            opticalid = 'hscid'):
	'''
	input variables:
	xdf, xcat, optdf,catopt,optdf,nm, qm, Q, rmag, rsearch=5.0,\
	magname = 'rmag_psf',xerrname='xposerr',
	xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',
    opticalid = 'hscid'
	For computing LR for every optical source within rsearch:
	'''
	idxmm, idhsc, d2d , d3d=catopt.search_around_sky(xcat,rsearch*u.arcsec)
	match = pd.DataFrame({'xid':idxmm,'optid':idhsc,'dist':d2d.arcsec,\
	'rmag':optdf.loc[idhsc,magname].values,'xposerr':xdf.loc[idxmm,xerrname],\
	'raopt':optdf.loc[idhsc,ora].values,'decopt':optdf.loc[idhsc,odec].values,\
	'rax':xdf.loc[idxmm,xra].values,'decx':xdf.loc[idxmm,xdec].values,\
	'optname':optdf.loc[idhsc,opticalid].values})

	fr = pdf_sep_gen(match.dist.values,match.xposerr.values,0.05)
	n_m = interp1d(rmag, nm, bounds_error=False,fill_value='extrapolate')
	q_m = interp1d(rmag, qm, bounds_error=False,fill_value='extrapolate')
	fnm = n_m(match.rmag.values)
	fqm = q_m(match.rmag.values)
	fqm[np.where(fqm < 0.)] = 1e-8
	LR = fr*fqm/fnm
	match['LR'] = LR
	if lrmax is None:
		R, C, LRth = calc_RC(match, lth, Q, len(xcat),LRfrac=LRfrac)
		func = interp1d(LRth, R+C, bounds_error=False,fill_value='extrapolate')
		arr = match.LR.values
		farr = func(arr)
		lthmax = arr[np.where(farr == max(farr))]
		if not np.isscalar(lthmax):
			if len(lthmax) >= 1:
				lthmax = lthmax[0]
		allmatch, R, C  = calc_RC(match,lthmax, Q, len(xcat),LRfrac=LRfrac)
		return allmatch, R, C, lthmax
	else:
		allmatch, R, C  = calc_RC(match,lrmax, Q, len(xcat),LRfrac=LRfrac)
		return allmatch, R, C, lrmax

def likmatch(xdf, xcat, optdf_in, catopt, radecerr = False, r0=2.5,rsearch=5.0, \
	r_in = 7., r_out=35., lth = np.linspace(0.05,0.9,10),LRfrac=0.2,lrmax=None,\
	nmagbin=15, niter=10,magname = 'rmag_psf',xerrname='xposerr',\
	xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',\
	opticalid = 'hscid'):
	'''
	Likelihood ratio based source matching.
	Currently is based on HSC public data release 1
	(wide survey) in the XMM-LSS region.
	Input: source list data frame or fits filename of the source lists.
	See the input parameters for default column names
	***Note that ``opticalid''
	should be provided for each unique optical source
	Default : xdf is in XMM SRCLIST format
	optdf is for HSC.

	Input parameters:
	r0 - radius used for defining q(m)
	r_in and r_out - radius used for selecting background sources
	(X-ray sources with distance from optical counterparts that's larger than
	r_in and smaller than r_out are defined as background sources.)
	if (len(catopt) != len(optdf)) or (len(xcat) != len(xdf)) :
		print("x/opt catalogs should be the astropy coordinate objects computed from the dataframes!!")
		sys.exit(1)
	'''
	NX = float(len(xcat))
	optdf = optdf_in.copy(deep=True)
	optdf.reset_index(inplace=True)
	#making a copy for output
	dfout = xdf.copy(deep=True)
	dfout.reset_index(inplace=True)
	#Background number surface density
	print('Calculating background mag. distribution, nm')
	nm, rmagbin = getbkgcat(xcat,catopt,optdf,r_in = r_in, r_out=r_out,
	nmagbin=nmagbin, magname = 'rmag_psf')

	#Calculating qm for the first time using r0
	print('Calculating initial counterpart mag. dist., qm')
	idopt_r0,idxmm,d2d,d3d=xcat.search_around_sky(catopt,r0*u.arcsec)
	N1 = float(len(np.unique(idopt_r0)))
	Q = N1/NX
	if (N1 != float(len(idopt_r0))):
		print('duplicated optical sources in qm calculation')
	opt_qm = optdf.loc[idopt_r0,:]
	grp=opt_qm.groupby(pd.cut(opt_qm[magname].values,bins=rmagbin))
	total_m=grp.ra.count().values
	real_m=total_m-np.pi*r0**2*NX*nm
	qm = real_m*(Q/np.sum(real_m))
	qm[np.where(qm < 0.)] = 0.

	rmag = binvalue(rmagbin)
	#With qm, nm, and Q, calculate the first match
	print('First LR matching')
	allmatch, R, C, lthmax = \
	calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, rsearch=rsearch,LRfrac=LRfrac,\
			lth = lth,lrmax=lrmax, magname = 'rmag_psf',xerrname=xerrname,
	        xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',
	        opticalid = 'hscid')


	#With the new ``matched sources'', recalculate qm again until C and R converges
	if lrmax is None:
		for i in range(niter):
			print('Iterative LR matching')
			qm, Q = getqm(allmatch,rmagbin, C)#, NX, nm)
			allmatch, R, C, lthmax = \
			calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, rsearch=rsearch,LRfrac=LRfrac,\
					lth = lth, lrmax=lrmax , magname = 'rmag_psf',xerrname=xerrname,\
			        xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',\
			        opticalid = 'hscid')
			print(R, C, len(allmatch),lthmax)
		return allmatch, R, C
	else:
		return allmatch, R, C
