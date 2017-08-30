def getbkg_iter(qm,rmagbin,NX,catopt,optdf,r_in = 10., r_out=50.,\
		  nmagbin=15, magname = 'imag_psf', ora='ra',odec='dec'):
	'''
	Use external qm, rmagbin to draw random sample from optdf
	then use those positions to search for background sources in catopt
	'''
	bkgdf = pd.DataFrame({'ra':optdf[ora].values.copy(),'dec':optdf[odec].values.copy(),
		'mag':optdf[magname].values.copy(),'rand':np.random.uniform(0,1,len(optdf))
		})
	out=pd.cut(bkgdf['mag'].values,bins=rmagbin)
	grp=bkgdf.groupby(out)
	nsrc = qm*NX
	cnt = 0
	xra = np.array([],dtype=float)
	xdec = np.array([],dtype=float)
	imag = np.array([],dtype=float)
	for ks in grp.groups.keys():
		ns = nsrc[cnt]
		tdf = grp.get_group(ks)
		xra = np.hstack((xra,
		tdf[tdf['rand'].values <= ns/len(tdf)]['ra'].values
		))
		xdec = np.hstack((xdec,
		tdf[tdf['rand'].values <= ns/len(tdf)]['dec'].values))
		imag = np.hstack((imag,
		tdf[tdf['rand'].values <= ns/len(tdf)]['icmodel_mag'].values))
		cnt += 1
	fcat = 


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
		else:
			return nm,rmagbin


