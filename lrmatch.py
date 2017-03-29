import pandas as pd
from astropy.io import fits
import astropy.coordinates as cd
from ctc_observ import *

#load HSC catalog first
hsc = pd.read_csv('/cuc36/xxl/multiwavelength/HSC/wide.csv')


def lrmatch(srclist,filename = False,radecerr = False):
    '''
    Likelihood ratio based source matching.
    Currently is based on HSC public data release 1
    (wide survey) in the XMM-LSS region.
    Input: source list data frame or fits filename of the source lists.
    ***The input catalog must have RA, DEC in degrees ***
    If there is no RADECERR in the input source list
    radecerr (in arcsec) must be specified
    '''
    if filename:
        df = tab(fits.getdata(srclist,1)).to_pandas()
    else:
        df = srclist
    #make a astropy catalog object
    xcat = makecd(df.RA,df.DEC)
    #only focuse on a fraction of the HSC catalog to save time
    hsc = hsc[(hsc.ra > min(df.RA)-0.01) & 
              (hsc.ra < max(df.RA)+0.01) &
              (hsc.dec > min(df.DEC)-0.01) &
              (hsc.dec < max(df.DEC)+0.01)]
    hsc.reset_index(inplace=True)
    cathsc = makecd(hsc.ra,hsc.dec)


    #magnitude distribution for background sources within 5--30"
    r_in, r_out = 5., 30.
    #For each HSC source, search the input source list (presumably XMM) matches within 5"
    idhsc,idxmm,d2d,d3d=xcat.search_around_sky(cathsc,r_in*u.arcsec)
    len(np.unique(idhsc))
    #Excluding each HSC with an XMM source within 5"
    itmp=np.arange(len(cathsc))
    itmp[np.unique(idhsc)]=-1
    #sanity check? probably redundant
    idhsc_ext5=np.where(np.equal(hsc.index.values, itmp))[0]

#For each HSC source, search a 30" radius for XMM counterparts?
idhsc_in30,idxmm,d2d,d3d=xcat.search_around_sky(cathsc,r_out*u.arcsec)
#Cross-correlated the ``no 5 arcsec list'', and the ``30 arcsec list''
idhsc_bkgd=np.intersect1d(idhsc_ext5,idhsc_in30)
print(len(idhsc_bkgd))
hsc_bkgd=hsc.loc[idhsc_bkgd]
hsc_bkgd.reset_index(inplace=True)

out,rmagbin=pd.cut(hsc_bkgd.rmag_psf,bins=10,retbins=True)
groups=hsc_bkgd.groupby(out)
#number density = total number of sources divided by the area of annulus
N_xmm=len(lss) #number of unique NuSTAR sources
N_bkgd=len(hsc_bkgd)
nm=groups.index.count().values/(np.pi*(r_out**2-r_in**2))


#Now estimate q(m) near XMM -- the expected sources magnitude 
#distribution of counterparts
#1. search r0 
r0=1.
idhsc_r0,idxmm,d2d,d3d=xcat.search_around_sky(cathsc,r0*u.arcsec)
idhsc_r0q=np.unique(idhsc_r0)
hsc_qm=hsc.loc[idhsc_r0q]

group_qm=hsc_qm.groupby(pd.cut(hsc_qm.rmag_psf.values,bins=rmagbin))
total_m=group_qm.index.count().values

real_m=total_m-np.pi*r0**2*N_xmm*nm
qm = real_m/np.sum(real_m)

def pdf_sep_gen(sep_arcsec,xposerr):
	'''
	PDF of angular separation between a NuSTAR object and the other input catalog
	with positional error poserr
	'''
	nor=2*np.sqrt(0.1**2+xposerr**2)#caution
	return np.exp(-sep_arcsec**2/nor)/np.pi*nor

idxmm, idhsc,d2d,d3d=cathsc.search_around_sky(xcat,5.0*u.arcsec)
hsc.loc[idhsc,'rmag_psf']
match_5sec = pd.DataFrame({'idxmm':idxmm,'idhsc':idhsc,'d2d':d2d.arcsec,\
'rmag':hsc.loc[idhsc,'rmag_psf'].values,'xposerr':lss.loc[idxmm,'RADEC_ERR'],\
'ra_hsc':hsc.loc[idhsc,'ra'].values,'dec_hsc':hsc.loc[idhsc,'dec'].values})

grp = match_5sec.groupby('idxmm')
ixm = grp.groups.keys()
ihscm = np.zeros_like(ixm)
lss['hsc_objid'] = np.zeros_like(lss.ML_ID_SRC)
lss['ra_hsc'] = np.zeros_like(lss.RA)
lss['dec_hsc'] = np.zeros_like(lss.RA)
lss['sep_arcsec'] = np.zeros_like(lss.RA)
from scipy.interpolate import interp1d
rmagbinv = rmagbin[0:-1] + ((rmagbin[-1] - rmagbin[0])/len(rmagbin))/2.
fnm = interp1d(rmagbinv, nm, bounds_error=False,fill_value='extrapolate')
fqm = interp1d(rmagbinv, qm, bounds_error=False,fill_value='extrapolate')
df = grp.get_group(ixm[0])
df[df.d2d == max(df.d2d)].loc[:,['idhsc','d2d','ra_hsc','dec_hsc']].values[0]
for i in ixm:
    df = grp.get_group(i)
    if len(df) > 1:
        df.loc[:,'LR'] = np.zeros(len(df))
        for index, row in df.iterrows():
            n = fnm(row.rmag)
            q = fqm(row.rmag)
            r = pdf_sep_gen(row.d2d,row.xposerr)
            df.loc[index,'LR'] = q*r/n
        if np.sum(df.LR == max(df.LR)) == 1:
            lss.loc[row.idxmm,['hsc_objid','sep_arcsec','ra_hsc','dec_hsc']] = \
            df[df.LR == max(df.LR)].loc[:,['idhsc','d2d','ra_hsc','dec_hsc']].values[0]
        else:
            print(index,'more than one with equal LR')
            lss.loc[row.idxmm,['hsc_objid','sep_arcsec','ra_hsc','dec_hsc']] = \
            df[(df.LR == max(df.LR)) & (df.d2d == min(df.d2d))].loc[:,['idhsc','d2d','ra_hsc','dec_hsc']].values[0]
    else:
        print(index,'only one source within 5"')
        lss.loc[row.idxmm,['hsc_objid','sep_arcsec','ra_hsc','dec_hsc']] = \
        df.loc[:,['idhsc','d2d','ra_hsc','dec_hsc']].values[0]

np.sum(lss.hsc_objid > 0)
len(np.unique(idxmm))
len(ixm)
lss.loc[lss[lss.hsc_objid > 0].index, 'specz'] = \
hsc.loc[lss[lss.hsc_objid > 0].hsc_objid.values,'specz_redshift'].values

lss.loc[lss[lss.hsc_objid > 0].index, 'photoz0'] = \
hsc.loc[lss[lss.hsc_objid > 0].hsc_objid.values,'pz_demp'].values

lss.loc[lss[lss.hsc_objid > 0].index, 'photoz1'] = \
hsc.loc[lss[lss.hsc_objid > 0].hsc_objid.values,'pz_mizuki'].values


lsshsc = lss[lss.hsc_objid > 0.]

sns.distplot(lsshsc[lsshsc.specz > 0.].specz,kde=False)
plt.xlim(-0.5,8.)

sns.distplot(lsshsc.photoz0,kde=False)
sns.distplot(lsshsc.photoz1,kde=False)

#LSS-herschel
lsshsc.reset_index(inplace=True)

hlss = tab(fits.getdata('/Users/ctchen/surveys/xxl/multiwavelength/XMM-LSS_xID250_DR4.fits',1)).to_pandas()
hlss.columns
hlss = hlss[hlss.f250 > 0.]
hlss.reset_index(inplace=True)
cathlss = makecd(hlss.ra,hlss.dec)
catxmm = makecd(lsshsc.ra_hsc,lsshsc.dec_hsc)
idh, idx, d2d, d3d = catxmm.search_around_sky(cathlss,5.0*u.arcsec)
len(np.unique(idx))
np.sum(lsshsc.loc[np.unique(idx),'specz'] > 0.)



