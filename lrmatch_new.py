import pandas as pd
import astropy.coordinates as cd
from ctc_observ import *
from ctc_arrays import *
#from scipy.interpolate import interp1d
from scipy.interpolate import pchip
from statsmodels.nonparametric.smoothers_lowess import lowess

#load HSC catalog first
#hsc = pd.read_csv('/cuc36/xxl/multiwavelength/HSC/wide.csv')
def pdf_sep_gen(sep_arcsec,xposerr,opterr,pdf='Rayleigh'):
	'''
	PDF of angular separation between an X-ray object and the other input catalog
	with positional error poserr
	'''
	if pdf == 'Gaussian':
		#that was 2d-normal
		poserr=2*(opterr**2+xposerr**2)# this is 2*sigma^2
		return np.exp(-sep_arcsec**2/poserr)/(np.pi*poserr)
	else:
		poserr = (opterr**2+xposerr**2)
		return (sep_arcsec/poserr)*np.exp((-sep_arcsec**2)/poserr)


def MLE(match):
    '''
    R and C for a single LRthreshold value
    '''
    if 'level_0' in match.columns:
    	match = match.drop('level_0',axis=1)
    if 'index' in match.columns:
    	match = match.drop('index',axis=1)
    tmp = match.set_index('matchid')
    grp = tmp.groupby('xid')
    #select sources with only one match
    onematch = grp.filter(lambda x: len(x) == 1).copy()
    onematch['Rc'] =  1.0
    #these are sources with multiple matches
    multimatch = tmp.loc[np.delete(tmp.index.values, onematch.index.values),:]
    #onematch.reset_index(inplace=True)
    nmx = match.xid.nunique() - onematch.xid.nunique()
    if 'level_0' in onematch.columns:
    	onematch = onematch.drop('level_0',axis=1)
    if 'index' in onematch.columns:
    	onematch = onematch.drop('index',axis=1)
    if 'level_0' in multimatch.columns:
    	multimatch = multimatch.drop('level_0',axis=1)
    if 'index' in multimatch.columns:
    	multimatch = multimatch.drop('index',axis=1)

    if nmx == 0:
        allmatch = onematch
    elif nmx == 1:
        multimatch['Rc'] = multimatch.LR / multimatch.LR.sum()
        allmatch = pd.concat([onematch,multimatch],ignore_index=False)
    else:
    #regroup, and for each group only keep sources with LR larger than LRfrac*max(LR)
        grp = multimatch.groupby('xid')
        multiRc = grp.apply(lambda df: df.LR / df.LR.sum()).values
        multimatch['Rc'] = multiRc
        allmatch = pd.concat([onematch,multimatch],ignore_index=False)
    return allmatch.reset_index().sort_values(by='matchid').reset_index().drop('index',axis=1)


def getbkgcat(xcat,catopt,optdf,r_in = 7., r_out=35.,magonly=False,\
		  nmagbin=15, magname = 'imag_psf', ora='ra',odec='dec',corr_glob=False,globonly=False):
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
	if magonly:
		return hsc_bkgd[magname].values
	else:
		out,rmagbin=pd.cut(hsc_bkgd[magname].values,bins=nmagbin,retbins=True)
		groups=hsc_bkgd.groupby(out)
		#number density = total number of sources divided by the area of annulus
		N_xmm=len(xcat) #number of unique XMM sources
		N_bkgd=len(hsc_bkgd)
		nm=groups[ora].count().values/(np.pi*(r_out**2-r_in**2)*N_xmm)
		if corr_glob | globonly:
		    #According to Brusa et al. 2007, at faint magnitudes
		    #nm is not correct and should use a global one.
		    out,rmagbin_global=pd.cut(optdf[magname].values,bins=nmagbin,retbins=True)
		    groups=optdf.groupby(out)
		    rmag_global = binvalue(rmagbin_global)
		    area = \
		    (optdf[ora].max() - optdf[ora].min())*(optdf[odec].max() - optdf[odec].min())*3600**2
		    nm_global = groups[ora].count().values/area
		    iglobal = np.where(rmagbin > 23.)[0][:-1]
		if corr_glob:
			nm[iglobal] = nm_global[iglobal]
		elif globonly:
			return nm_global, rmagbin
		return nm,rmagbin


#def getqm(match,rmagbin, Q, NX, nm, r0=2.5):
def getqm(match,rmagbin, Q, nm, NX, r0=3.0):
	'''
    Estimate q(m) -- the expected optical counterpart magnitude
    distribution of at magintude m
	'''
	grp=match.groupby(pd.cut(match['rmag'].values,bins=rmagbin))
	real_m=grp.rax.count().values# - np.pi*r0**2*NX*nm
	real_m[np.where(real_m < 0.)] = \
	0.1*nm[np.where(real_m < 0.)]*np.pi*NX*r0**2
	qm = real_m*Q/np.sum(real_m)
	rmagarr = np.array([])
	qmarr = np.array([])
	nmarr = np.array([])
	for index, i in enumerate(rmagbin[:-1]):
	    rmagarr = np.hstack((rmagarr,np.linspace(i, rmagbin[index+1], 5)))
	    qmarr = np.hstack((qmarr, np.zeros(5) + qm[index]))
	result = lowess(qmarr,rmagarr,frac=0.2)
	x_smooth = result[:,0]
	y_smooth = result[:,1]
	return x_smooth, y_smooth, Q, qm#, real_m



def calc_RCMAX(match, quntarr, Q,NX,LRfrac=0.2,first=False):
	'''
	R and C for a single LRthreshold value
	'''
	if type(NX) != float:
		NX = float(NX)	
	LRth = quntarr
	tmp = match[match.LR > LRth].copy().reset_index().drop('index',axis=1)
	grp = tmp.groupby('xid')
	#select sources with only one match
	onematch = grp.filter(lambda x: len(x) == 1).copy()
	onematch['Rc'] =  onematch.LR.values/(onematch.LR.values + 1 - Q)
	#these are sources with multiple matches
	multimatch = tmp.loc[np.delete(tmp.index.values, onematch.index.values),:].reset_index().drop('index',axis=1)
	onematch.reset_index(inplace=True)
	nmx = tmp.xid.nunique() - onematch.xid.nunique()
	if nmx == 0:
		allmatch = onematch
	elif nmx == 1:
		multimatch['Rc'] = multimatch.LR/(multimatch.LR.sum() + (1-Q))
		allmatch = pd.concat([onematch,multimatch],ignore_index=False)
	else:
	#regroup, and for each group only keep sources with LR larger than LRfrac*max(LR)
		grp = multimatch.groupby('xid')
		igood = grp.apply(lambda df:df.LR/df.LR.max() >= LRfrac).values
		multimatch = multimatch[igood].reset_index().drop('index',axis=1)
		grp = multimatch.groupby('xid')
		multiRc = grp.apply(lambda df: df.LR/(df.LR.sum()+(1-Q))).values
		multimatch['Rc'] = multiRc
		allmatch = pd.concat([onematch,multimatch],ignore_index=False)
	R = allmatch.Rc.mean()
	C = allmatch.Rc.sum()/NX
	return allmatch, R, C, LRth


def calc_RC(match, quntarr, Q,NX,LRfrac=0.2,first=False):
	'''
	R and C for an array of LRthreshold values
	if first==True,
	quantarr should be between 0 to 1 
	and the LRthreshold values to be looped through would be
	match.LR.quantile(quntarr)

	If quntarr is an array with length > 1 (and values between 0 to 1)
	This subroutine finds the LRth value that maximize C and R.
	'''
	if type(NX) != float:
		NX = float(NX)
	if first:
		#if it's the first time, loop through the LR values in quantile arrays
		#return R, C, LRth
		LRth = match.LR.quantile(quntarr).values
		print('first -- ', 'min/max LRth are ', np.min(LRth), np.max(LRth))
	else:
		LRth = quntarr
	R = np.zeros(len(quntarr))
	C = np.zeros(len(quntarr))
	for index, lrthiter in enumerate(LRth):
		tmp = match[match.LR > lrthiter].copy().reset_index().drop('index',axis=1)
		grp = tmp.groupby('xid')
		onematch = grp.filter(lambda x: len(x) == 1).copy() #select sources with only one match
		#onematch.reset_index(inplace=True)
		onematch['Rc'] = onematch.LR.values/(onematch.LR.values + 1 - Q)			
		#these are sources with multiple matches
		multimatch = tmp.loc[np.delete(tmp.index.values, onematch.index.values),:].reset_index().drop('index',axis=1)
		onematch.reset_index(inplace=True)
		nmx = tmp.xid.nunique() - onematch.xid.nunique()
		if nmx == 0:
			#no x-ray sources have multiple good counterparts
			allmatch = onematch
		elif nmx == 1:
			#only one x-ray sources have multiple good counterparts
			multimatch['Rc'] = multimatch.LR/(multimatch.LR.sum() + (1-Q))
			allmatch = pd.concat([onematch,multimatch],ignore_index=True)
		else:				
			grp = multimatch.groupby('xid')
			igood = grp.apply(lambda df:df.LR/df.LR.max() >= LRfrac).values
			#dropping sources with LR < LRfrac*LRmax
			multimatch = multimatch[igood].reset_index().drop('index',axis=1)
			#regroup
			grp = multimatch.groupby('xid')			
			multiRc = grp.apply(lambda df: df.LR/(df.LR.sum()+(1-Q))).values
			multimatch['Rc'] = multiRc
			allmatch = pd.concat([onematch,multimatch],ignore_index=False)
		R[index] = allmatch.Rc.mean()
		C[index] = allmatch.Rc.sum()/NX
	return R, C, LRth


def calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, NX,rsearch=5.0,\
			lth = None, LRfrac=0.2,lrmax=None,\
			magname = 'imag_psf',xerrname='xposerr',
            xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',
            opticalid = 'hscid',opterr = 0.1,pdf='Rayleigh',first=False):
	'''
	input variables:
	xdf, xcat, optdf,catopt,optdf,nm, qm, Q, rmag, rsearch=5.0,\
	magname = 'rmag_psf',xerrname='xposerr',
	xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',
    opticalid = 'hscid'
	For computing LR for every optical source within rsearch:
	'''
	if first:
		print('first calc_LR')
	idxmm, idhsc, d2d , d3d=catopt.search_around_sky(xcat,rsearch*u.arcsec)
	match = pd.DataFrame({'xid':idxmm,'optid':idhsc,'dist':d2d.arcsec,\
	'rmag':optdf.loc[idhsc,magname].values,'xposerr':xdf.loc[idxmm,xerrname],\
	'raopt':optdf.loc[idhsc,ora].values,'decopt':optdf.loc[idhsc,odec].values,\
	'rax':xdf.loc[idxmm,xra].values,'decx':xdf.loc[idxmm,xdec].values,\
	'optname':optdf.loc[idhsc,opticalid].values})

	#print('match len = ',len(match), 'xid nunique = ', match.xid.nunique())
	fr = pdf_sep_gen(match.dist.values,match.xposerr.values,opterr,pdf=pdf)
	n_m = pchip(rmag, nm)#, bounds_error=False,fill_value='extrapolate')
	q_m = pchip(rmag, qm)#, bounds_error=False,fill_value='extrapolate')
	fnm = n_m(match.rmag.values)
	fqm = q_m(match.rmag.values)
	fqm[np.where(fqm < 0.)] = 1e-8
	fnm[np.where(fnm < 0.)] = 1e-8
	LR = fr*fqm/fnm
	match['LR'] = pd.Series(LR, index=match.index)
	match['matchid'] = pd.Series(range(len(match)),index=match.index)
	match['raoff'] = pd.Series((match.rax - match.raopt)*3600., index=match.index)
	match['decoff'] = pd.Series((match.decx - match.decopt)*3600., index=match.index)
	#several situations : 
	#1. all matches are unique, no further action is required.
	if match.xid.nunique() - len(match) == 0:
		return match, match, 1.0, 1.0, match.LR.min()
	else:
		if lth is None:
			#If the array of lth values is not provided, 
			#guess it by assuming that only NX sources would be reliable, 
			#so loop through the LR values around that LR quantile
			#qcenter = match.LR.quantile(float(NX)/len(match))
			qcenter = 1. - 1.5*float(NX)/len(match)
			if qcenter < 0.:
				qcenter = 0.1
			lth = np.linspace(0.5*qcenter, 
				min([2.0*qcenter, 0.95]), 30.)
			#print(lth)
		if lrmax is None:
			#first
			R, C, LRth = calc_RC(match, lth, Q, NX,LRfrac=LRfrac,first=first)
			lthmax = LRth[np.argmax((R+C))]
			if not np.isscalar(lthmax):
				if len(lthmax) >= 1:
					lthmax = lthmax[0]
			goodmatch, R, C, LRth  = calc_RCMAX(match,lthmax, Q, len(xcat),LRfrac=LRfrac)
			return match, goodmatch, R, C, lthmax, LRth
		else:
			goodmatch, R, C, LRth  = calc_RCMAX(match,lrmax, Q, len(xcat),LRfrac=LRfrac)
			return match, goodmatch, R, C, lrmax, LRth

def likmatch(xdf, xcat, optdf_in, catopt, radecerr = False, r0=2.5,rsearch=5.0, \
	r_in = 7., r_out=35., lth = None,LRfrac=0.5,lrmax=None,\
	nmagbin=15, niter=10,numid='numid',magname = 'imag_psf',xerrname='xposerr',\
	xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',\
	opticalid = 'hscid',opterr=0.1,pdf='Rayleigh',verbose=True):
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
	optdf = optdf_in.copy(deep=True)	
	optdf.set_index(numid,inplace=True)
	#making a copy for output
	dfout = xdf.copy(deep=True).reset_index()
	#Background number surface density
	nm, rmagbin = getbkgcat(xcat,catopt,optdf,r_in = r_in, r_out=r_out,
	nmagbin=nmagbin, magname = magname,ora=ora,odec=odec)
	if verbose:print('Calculating background mag. distribution, nm')
	#nm = nm/np.sum(nm)
	#find the number of X-ray sources at least one matching withn 1' (sample completeness)
	idopt_r0,d2d,d3d=xcat.match_to_catalog_sky(catopt)#,1.0*u.arcmin)
	NX = sum(d2d.arcmin <= 1.)
	idopt_r0,idxmm,d2d,d3d=xcat.search_around_sky(catopt,r0*u.arcsec)
	N1 = float(len(np.unique(idopt_r0)))
	Q = N1/NX
	print('Q = ', Q, ', N1 = ',N1, ' NX = ', NX)
	if (N1 != float(len(idopt_r0))):
		print('duplicated optical sources in qm calculation')
	opt_qm = optdf.loc[idopt_r0,:]
	grp=opt_qm.groupby(pd.cut(opt_qm[magname].values,bins=rmagbin))
	total_m=grp[ora].count().values
	real_m0=total_m-np.pi*r0**2*NX*nm
	real_m0[np.where(real_m0 < 0.)] = 0.1*nm[np.where(real_m0 < 0.)]*np.pi*NX*r0**2
	qm0 = real_m0*(Q/np.sum(real_m0))

	rmagarr = np.array([])
	qmarr = np.array([])
	nmarr = np.array([])
	for index, i in enumerate(rmagbin[:-1]):
	    rmagarr = np.hstack((rmagarr,np.linspace(i, rmagbin[index+1], 5)))
	    qmarr = np.hstack((qmarr, np.zeros(5) + qm0[index]))
	    nmarr = np.hstack((nmarr, np.zeros(5) + nm[index]))    

	result = lowess(qmarr,rmagarr,frac=0.2)
	rmagsmooth = result[:,0]
	qmsmooth = result[:,1]

	result = lowess(nmarr,rmagarr,frac=0.2)
	#rmagsmooth = result[:,0]
	nmsmooth = result[:,1]

	#for unrealistical qm values (<0), assuming the real counterpart distribution is the same
	#as the background
	#qm0[np.where(qm0 < 0.)] = nm[np.where(qm0 < 0.)]

	rmag = rmagsmooth#binvalue(rmagbin)
	if verbose:print('Calculating initial counterpart mag. dist., qm')	
	if verbose:print('Calculating background mag. distribution, rmag')
	density_raw = pd.DataFrame({
		'rmag':binvalue(rmagbin),
		'qm0':qm0,
		'nm':nm
		}
		)
	density = pd.DataFrame({'rmag':rmag,'qm0':qmsmooth,'qms'+str(np.round(Q,2)):qmsmooth,'nm':nmsmooth})#,'real_ms':real_m0})
	#With qm, nm, and Q, calculate the first match
	if verbose:print('First LR matching')
	match, goodmatch, R, C, lthmax, LRth = \
	calc_LR(xdf, xcat, optdf,catopt,nmsmooth, qmsmooth, Q, rmag, NX, rsearch=rsearch,LRfrac=LRfrac,\
			lth = lth,lrmax=lrmax, magname = magname,xerrname=xerrname,
	        xra = xra, xdec = xdec, ora = ora, odec = odec,
	        opticalid = opticalid,opterr=opterr,pdf=pdf,first=True)
	if verbose:print('Q0='+str(Q), 'R0='+str(R),'C0='+str(C), len(goodmatch), lthmax)
	#With the new ``matched sources'', recalculate qm again until C and R converges
	if lrmax is None:
		for i in range(niter):
			if len(goodmatch) == 0:
				print('No goodmatches (LRthreshold = ',lthmax,'), resetting to 0.4')
				lthmax = 0.4
			lth = np.sort(np.hstack((match.LR.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values, \
				np.linspace(lthmax*0.5,lthmax*1.5,5))))
			lthmax0 = lthmax * 1.
			x_smooth, qm, Q, qmraw = getqm(goodmatch,rmagbin, C, nm, NX, r0 = r0)#, NX, nm)
			match, goodmatch, R, C, lthmax, LRth = \
			calc_LR(xdf, xcat, optdf,catopt,nmsmooth, qm, Q, rmag, NX, rsearch=rsearch,LRfrac=LRfrac,\
					lth = lth, lrmax=lrmax , magname = magname,xerrname=xerrname,\
			        xra = xra, xdec = xdec, ora = ora, odec = odec,\
			        opticalid = opticalid,opterr=opterr,pdf=pdf, first=False)
			density['qm'+str(i)+'_'+str(np.round(Q,2))] = pd.Series(qm,index=density.index)
			density_raw['qm'+str(i)+'_'+str(np.round(Q,2))] = pd.Series(qmraw,index=density_raw.index)
			#density['real_m'+str(i)] = pd.Series(real_m,index=density.index)
			if verbose:print('R, C, len(goodmatch), LRth:' ,R, C, len(goodmatch),lthmax)
			if verbose:print('Iter',i, 'new LRth = ', lthmax, 'old LRth =', lthmax0 )
			if (np.abs(lthmax0 - lthmax) < 0.01) & (lthmax > 0.1) & (i >= 4):
				if verbose:print('LR threshold converges, breaking now')
				density['qmfinal'] = pd.Series(qm,index=density.index)
				density_raw['qmfinal'] = pd.Series(qmraw,index=density_raw.index)
				break
			elif i == max(range(niter)):
				density['qmfinal'] = pd.Series(qm,index=density.index)
				density_raw['qmfinal'] = pd.Series(qmraw,index=density_raw.index)

		return match,goodmatch, R, C, density, density_raw, lthmax, rmagbin
	else:
		match, goodmatch, R, C, lthmax, LRth = \
		calc_LR(xdf, xcat, optdf,catopt,nmsmooth, qmsmooth, Q, rmag, NX, rsearch=rsearch,LRfrac=LRfrac,\
				lth = lth,lrmax=lrmax, magname = magname,xerrname=xerrname,
		        xra = xra, xdec = xdec, ora = ora, odec = odec,
		        opticalid = opticalid,opterr=opterr,pdf=pdf)
		x_smooth, qm, Q, qmraw = getqm(goodmatch,rmagbin, C, nm, NX, r0 = r0)
		density['qmfinal'] = pd.Series(qm,index=density.index)
		density_raw['qmfinal'] = pd.Series(qmraw,index=density_raw.index)
		return match,goodmatch, R, C, density, density_raw, lthmax, rmagbin


def likmatch_rerun(xdf, xcat, optdf_in, catopt, density, radecerr = False, r0=2.5,rsearch=5.0, \
	r_in = 7., r_out=35., lth = np.linspace(0.05,0.9,10),LRfrac=0.2,lrmax=None,\
	nmagbin=15, niter=10,numid='numid',magname = 'imag_psf',xerrname='xposerr',\
	xra = 'RA', xdec = 'DEC', ora = 'ra', odec = 'dec',\
	opticalid = 'hscid',opterr=0.1,pdf='Rayleigh',verbose=True,rc=False):
	'''
	similar to likmatch, but requires the density output from likmatch
	useful for shift-and-rematch simulations
	'''
	optdf = optdf_in.copy(deep=True)
	optdf.set_index(numid,inplace=True)	
	NX = float(len(xcat))
	idopt_r0,idxmm,d2d,d3d=xcat.search_around_sky(catopt,r0*u.arcsec)
	N1 = float(len(np.unique(idopt_r0)))
	Q = N1/NX
	nm = density.nm.values
	qm = density.qmfinal.values
	rmag = density.rmag.values
	match, goodmatch, R, C, lthmax, LRth = \
	calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, NX, rsearch=rsearch,LRfrac=LRfrac,\
			lth = lth,lrmax=lrmax, magname = magname,xerrname=xerrname,
	        xra = xra, xdec = xdec, ora = ora, odec = odec,
	        opticalid = opticalid,opterr=opterr,pdf=pdf)
	if rc:
		return match,goodmatch, R, C
	else:
		return match,goodmatch



def likmatch_ext(
	xdf, xcat, optdf_in, catopt, density, r0=3.0, rsearch=10.0, \
	r_in = 10., r_out=50., \
	lth = None, LRfrac=0.5, lrmax=None, \
	nmagbin=15, niter=10, numid='numid', magname = 'imag_psf', 
	xerrname='xposerr', xra = 'RA', xdec = 'DEC', \
	ora = 'ra', odec = 'dec', opticalid = 'hscid',opterr=0.1, \
	pdf='Rayleigh',verbose=True):
	'''
	Likelihood ratio based source matching.
	different from the original likmatch function,
	this one requires an input array true-counterpart mag, which will be used
	to calculate q(m) using kernel density estimation
	The background mag. distribution nm is optional
	'''
	optdf = optdf_in.copy(deep=True)
	optdf.set_index(numid,inplace=True)	
	NX = float(len(xcat))
	idopt_r0,idxmm,d2d,d3d=xcat.search_around_sky(catopt,r0*u.arcsec)
	N1 = float(len(np.unique(idopt_r0)))
	Q = N1/NX
	nm = density.nm.values
	qm = density.qmfinal.values
	rmag = density.rmag.values

	match, goodmatch, R, C, lthmax, LRth = \
	calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, NX, rsearch=rsearch,LRfrac=LRfrac,\
			lth = lth,lrmax=lrmax, magname = magname,xerrname=xerrname,
	        xra = xra, xdec = xdec, ora = ora, odec = odec,
	        opticalid = opticalid,opterr=opterr,pdf=pdf,first=True)
	if verbose:print('Q0='+str(Q), 'R0='+str(R),'C0='+str(C), len(goodmatch), lthmax)
	return match,goodmatch, R, C, lthmax
	'''
	#With the new ``matched sources'', recalculate qm again until C and R converges
	if lrmax is None:
		for i in range(niter):
			if len(goodmatch) == 0:
				print('No goodmatches (LRthreshold = ',lthmax,'), resetting to 0.4')
				lthmax = 0.4
			lth = np.sort(np.hstack((match.LR.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values, \
				np.linspace(lthmax*0.5,lthmax*1.5,5))))
			lthmax0 = lthmax * 1. 			
			#qm, Q, real_m = getqm(goodmatch,rmagbin, C, nm, NX, r0 = r0)#, NX, nm)
			match, goodmatch, R, C, lthmax, LRth = \
			calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, rsearch=rsearch,LRfrac=LRfrac,\
					lth = lth, lrmax=lrmax , magname = magname,xerrname=xerrname,\
			        xra = xra, xdec = xdec, ora = ora, odec = odec,\
			        opticalid = opticalid,opterr=opterr,pdf=pdf, first=False)
			#density['qm'+str(i)+'_'+str(np.round(Q,2))] = pd.Series(qm,index=density.index)
			#density['real_m'+str(i)] = pd.Series(real_m,index=density.index)
			if verbose:print(R, C, len(goodmatch),lthmax)
			if verbose:print('Iter',i, 'new LRth = ', lthmax, 'old LRth =', lthmax0 )
			if (np.abs(lthmax0 - lthmax) < 0.01) & (lthmax > 0.1) & (i >= 4):
				if verbose:print('LR threshold converges, breaking now')
				#density['qmfinal'] = pd.Series(qm,index=density.index)
				break
			elif i == max(range(niter)):
				print('max niter reached, should check convergence')
				#density['qmfinal'] = pd.Series(qm,index=density.index)
		return match,goodmatch, R, C, lthmax
	else:
		match, goodmatch, R, C, lthmax, LRth = \
		calc_LR(xdf, xcat, optdf,catopt,nm, qm, Q, rmag, rsearch=rsearch,LRfrac=LRfrac,\
				lth = lth,lrmax=lrmax, magname = magname,xerrname=xerrname,
		        xra = xra, xdec = xdec, ora = ora, odec = odec,
		        opticalid = opticalid,opterr=opterr,pdf=pdf)
		#qm, Q, real_m = getqm(goodmatch,rmagbin, C, nm, NX, r0 = r0)
		#density['qmfinal'] = pd.Series(qm,index=density.index)
		return match,goodmatch, R, C, lthmax
	'''

'''
def finalmatch(match,goodmatch):
	match.set_index(match.matchid.values,inplace=True)
	mid_all = np.arange(len(match))
	mid_all[goodmatch.matchid.values] = -1
	badmatch = match.loc[mid_all[mid_all > 0],:]
	#if an xid alread has a counterpart in goodmatch, drop it.
	badmatch = badmatch[np.in1d(badmatch.xid.values, goodmatch.xid.unique(),invert=True)].copy()
	badmatch.reset_index(inplace=True)
	bad_ok = badmatch.drop_duplicates('xid',keep=False)
	ibad = np.arange(len(badmatch))
	ibad[bad_ok.index.values] = -1
	bad_bad = badmatch.loc[np.where(ibad > -1)[0],:]
	bad_bad.drop('index',axis=1,inplace=True)
	okmatch = pd.concat([goodmatch, bad_ok])
	return okmatch, bad_bad
'''

def finalmatch(match,lrth,LRfrac=0.5, r99=True):
	'''
	turn the match DF into the ``good matches''
	the ``ok matches''
	and the ``bad matches''
	'''
	if 'level_0' in match.columns:
		match = match.drop('level_0',axis=1)
	if 'index' in match.columns:
		match = match.drop('index',axis=1)
	tmp = match.set_index('matchid')
	tmp['r99'] = tmp.xposerr * 3.5
	grp = tmp.groupby('xid')
	onematch = grp.filter(lambda x: len(x) == 1).reset_index()
	multimatch = tmp.loc[np.delete(tmp.index.values, 
		onematch.matchid.values),:].reset_index()
	#now work on good matches
	goodones = onematch[onematch.LR >= lrth]
	goodmul = multimatch[multimatch.LR >= lrth]
	grp = goodmul.groupby('xid')
	igood = grp.apply(lambda df:df.LR/df.LR.max() >= LRfrac).values
	goodmul = goodmul[igood].reset_index()
	allgood = pd.concat([goodones, goodmul], ignore_index=False)
	#now work on the ok matches
	nogood = tmp.drop(allgood.matchid.values)
	if r99:
		ng_99 = nogood[nogood.dist <= nogood.r99].drop_duplicates('xid',keep=False).reset_index()
	else:
		ng_99 = nogood.drop_duplicates('xid',keep=False).reset_index()
	#ng_ok = ng_99[ng_99.LR >= lrth].reset_index()
	bad = nogood.drop(ng_99.matchid.values)
	allok = pd.concat([allgood, ng_99], ignore_index=False)
	return allgood, allok, bad

