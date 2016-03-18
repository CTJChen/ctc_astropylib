from scipy.optimize import nnls
import multiprocessing as mp
from ctc_filters import *
from ggplot import *
from astropy.coordinates import Distance
from astropy import units as u
import seaborn as sns

'''
Perform non negative least square SED fitting using
the Assef et al. 2010 templates and extinction laws
#self.built_in_inst = ['SDSS','WISE','2MASS','SPIRE',
'PACS','KPNO_MOSAIC','KPNO_NEWFIRM','GALEX','JOHNSON']
'''
dict_wav={'u':0.3543,'g':0.4770,'r':0.6231,'i':0.7625,'z':0.9134,
'U':0.36,'B':0.44,'V':0.55,'R':0.64,'I':0.79,
'W1':3.368,'W2':4.618,'W3':12.082,'W4':22.194,
'j':1.235,'h':1.662,'ks':2.159,
'J':1.235,'H':1.662,'Ks':2.159,
'SDSSu':0.3543,'SDSSg':0.4770,'SDSSr':0.6231,'SDSSi':0.7625,'SDSSz':0.9134,
'2MASSJ':1.235,'2MASSH':1.662,'2MASSKs':2.159,
'galex_fuv':0.1542,'galex_nuv':0.2274,'bess_b':0.44, 'bess_r':0.64,  'bess_i':0.79} 


class filter_resp:
    '''
    class for filter response curves
    could be used to obtain filter response curves
    for any input instruments
    or manually input arrays
    Currently availabe instruments :
    'SDSS' : u, g, r, i, z
    'WISE' : W1, W2, W3, W4 (3.4, 4.6, 12, 22 microns)
    '2MASS': J, H, Ks
    'SPIRE': 250, 350, 500 (microns)
    'PACS' : 70, 100, 160 (microns)
    'KPNO-MOSAIC' : Bw, R, I
    'KPNO-NEWFIRM': J, H, Ks
    'GALEX' :
    'BESSELL': U, B, V, R, I
    'IRAC'
    'MIPS'
    set all = True to get all available response functions
    otherwise, set instrumnet
    '''
    def __init__(self, instrument = None, wav = None, resp_input = None, all = False):
        self.df = pd.DataFrame()
        self.bulit_in_inst = ['SDSS','2MASS','WISE','GALEX','BESSELL','AKARI-IRC']
        if wav is None: 
            wav = np.hstack((np.linspace(0.03,3.0,500),np.linspace(3.0001,100,500)))
        self.df['wav'] = wav
        if all:#read in response functions
            instrument = None
            for i in self.bulit_in_inst:
                if i == 'GALEX':
                    galex = galex_resp(wav, nowav = True)
                    self.df = pd.concat([self.df,galex],axis=1)
                elif i == 'SDSS':
                    sdss = sdss_resp(wav, nowav = True)
                    self.df = pd.concat([self.df,sdss], axis=1)
                elif i == 'BESSELL':
                    bess = bess_resp(wav, nowav = True)
                    self.df = pd.concat([self.df,bess], axis=1)
                elif i == '2MASS':
                    nir_2mass = nir_2mass_resp(wav, nowav = True)
                    self.df = pd.concat([self.df,nir_2mass], axis=1)                              
                elif i == 'WISE':
                    wise = wise_resp(wav, nowav = True)
                    self.df = pd.concat([self.df,wise], axis=1)
                elif i == 'AKARI-IRC':
                    akari = akari_resp(wav, nowav = True)
                    self.df = pd.concat([self.df,akari], axis=1)
        if instrument:
            for i in instrument:
                if i not in self.bulit_in_inst:
                    print('I do not recognize '+i+', please use resp_input')
                else:
                    if i == 'SDSS':
                        sdss = sdss_resp(wav, nowav = True)
                        self.df = pd.concat([self.df,sdss], axis=1)
                    elif i == '2MASS':
                        nir_2mass = nir_2mass_resp(wav, nowav = True)
                        self.df = pd.concat([self.df,nir_2mass], axis=1)                        
                    elif i == 'WISE':
                        wise = wise_resp(wav, nowav = True)
                        self.df = pd.concat([self.df,wise], axis=1)

        self.df[self.df<1e-4] = 0.

def ext_smc_mky(wav,ebv):
    path_tmp = home+'/lib/sedtemplates/'
    ext = pd.read_csv(path_tmp+'assef_kappa.raw.dat',sep=' ',names=['wav','kappa'])
    kappa = np.interp(wav,ext.wav,ext.kappa)
    kappa[kappa < 0]=0.
    ext['ext']=10**(-kappa*ebv)
    ext[ext.ext < 1e-10] = 0.
    ext[ext.ext > 1.] = 1.
    return ext.ext.values

def int_over_resp(wav, flux, resp, z=None):
    '''
    wav, flux for the input templates
    resp could either be a pandas structure or
    a 1d array
    '''
    if z is not None:
        wav = wav*(1+z)
    if type(resp) is pd.DataFrame:
        if 'wav' in resp.columns:resp.drop('wav',axis=1,inplace=True)
        output=pd.DataFrame(columns = resp.columns)
        for i in range(len(resp.columns)):
            resp_band = resp.ix[:,i]
            int_flux = np.trapz(resp_band*flux, 3e14/wav)
            norm = np.trapz(resp_band, 3e14/wav)
            output[resp.columns[i]] = [int_flux/norm]
        return output
    else:
        int_flux = np.trapz(resp*flux, 3e14/wav)
        norm = np.trapz(resp, 3e14/wav)
        return int_flux/norm


#read in Roberto Assef's low resolution templates 
#and the extinction law
def lrt_sed(data,sedtmp = None,verbose = False):
    '''
    input:
    data: {'index':object names,'band_name': micro jansky in the band, 'redshift':redshift}
    sedtmp: should be a sed_templates object
    1. get some templates
    2. convolve it with the response curves (redshifted),
    3. get only the photometry of the observed filters
    3. x = np.vstack(model1, model2, model3 ...).transpose()
    4. y = [observed photometry]
    5. a, chi = nnls(x,y) #non lineasr least square minimization
    6. Create a grid of AGN SED with different extinction
    '''
    if type(data.columns) is pd.core.index.MultiIndex:
        output = pd.DataFrame(index = data.index, columns = ['c_AGN','c_elip','c_sbc','c_im','ebv','chi_red'])
        path_tmp = home+'/lib/sedtemplates/'
        if sedtmp is None:
            a10 = pd.read_fwf(path_tmp+'assef_lrt_templates.dat',colspecs=[(0,16),(22,31),(36,46),(48,61),(63,76),(80,91)],\
                              comment='#',names=['wav','agn','agn2','e','sbc','im'])
            #normalize each templates to elip at 1 micron
            a10.agn = a10.agn.values*a10.loc[152,'e']/a10.loc[152,'agn']
            a10.sbc = a10.sbc.values*a10.loc[152,'e']/a10.loc[152,'sbc']
            a10.im = a10.im.values*a10.loc[152,'e']/a10.loc[152,'im']
        else:
            a10=sedtmp    
        ext_grid = np.hstack((np.linspace(0,0.1,11),np.linspace(0.15,1.,18),np.linspace(2.,10.,9)))
        resp = filter_resp(wav=a10.wav,all=True)
        #iterate data
        phot = data.xs('flux',level = 1,axis=1)
        err = data.xs('err',level = 1,axis=1)
        #bands = data.columns.levels[0][0:-1]
        wav = a10.wav.values
        e_tmp = a10.e.values
        sbc_tmp = a10.sbc.values
        im_tmp = a10.im.values
        for nuid, row in phot.iterrows():
            if sum(row.notnull()) > 0:
                z=data.loc[nuid,'redshift'].values
                bands = row[row.notnull()].index
                flux = row[row.notnull()]
                fluxerr = err.loc[nuid,bands]
                fluxerr[fluxerr.isnull()]=flux[fluxerr.isnull()] 
                fluxerr[fluxerr == 0] = 0.001*flux[fluxerr ==0]
                #if fluxerr = nan, use flux value(assuming 1 sigma UL)
                resp0 = resp.df.loc[:,bands]
                coeff = np.zeros((4,len(ext_grid)),dtype=float)
                chi = np.zeros(len(ext_grid),dtype=float)
                elip = int_over_resp(wav, e_tmp,resp0,z=z)/fluxerr.values
                sbc = int_over_resp(wav,sbc_tmp,resp0,z=z)/fluxerr.values
                im = int_over_resp(wav,im_tmp,resp0,z=z)/fluxerr.values   
                for id2, ext in enumerate(ext_grid):
                    agn_tmp = a10.agn*ext_smc_mky(wav,ext)
                    agn = int_over_resp(wav, agn_tmp,resp0,z=z)/fluxerr.values
                    models = np.vstack((agn,elip,sbc,im)).transpose()
                    coeff[:,id2], chi[id2] = nnls(models,flux/fluxerr.values)
                i_chimin = np.argmin(chi)
                output.loc[nuid,['c_AGN','c_elip','c_sbc','c_im']] = coeff[:,i_chimin]
                output.loc[nuid,'ebv'] = ext_grid[i_chimin]
                output.loc[nuid,'chi_red']=chi[i_chimin]/float(len(bands))
                if verbose:
                    print(nuid,' has reduced chi square of ',output.loc[nuid,'chi_red'])
            else:
                print(nuid,' has no data')
        return output
    else:
        print('input data should have pandas multiindex')



class sed_templates:
    '''
    Load known templates (Assef+2010), 
    or takes an input pandas DF
    with the first column being wavelengths.
    Could also set normalization `norm'
    '''
    def __init__(self,tmp=None,norm=None,verbose=True):
        if tmp is None:
            path_tmp = home+'/lib/sedtemplates/'
            self.df = pd.read_fwf(path_tmp+'assef_lrt_templates.dat',colspecs=[(0,16),(22,31),(36,46),(48,61),(63,76),(80,91)],\
                   comment='#',names=['wav','agn','agn2','e','sbc','im'])            
            #normalize each templates to elip at 1 micron
            #drop AGN2
            self.df.drop('agn2',axis=1,inplace=True)
            self.df.agn = self.df.agn.values*self.df.loc[152,'e']/self.df.loc[152,'agn']
            self.df.sbc = self.df.sbc.values*self.df.loc[152,'e']/self.df.loc[152,'sbc']
            self.df.im = self.df.im.values*self.df.loc[152,'e']/self.df.loc[152,'im']
            self.df['galaxy'] = self.df.im.values+self.df.e.values+self.df.sbc.values
            if verbose:
                print('Loading Assef 2010 templates')
            if norm is not None:
                if verbose:
                    print('norm should be an array with 5 or 6 elements')
                    print('in the order of agn, agn2, e, sbc, im (, ext)')
                if len(norm) == 5:
                    self.df.agn = self.df.agn.values*norm[0]
                    self.df.e = self.df.e.values*norm[2]
                    self.df.sbc = self.df.sbc.values*norm[3]
                    self.df.im = self.df.im.values*norm[4]
                    self.df['galaxy'] = self.df.im.values+self.df.e.values+self.df.sbc.values                
                    self.df['bestfit'] = self.df.im.values+self.df.e.values+self.df.sbc.values+self.df.agn.values                    
                if len(norm) == 6:
                    self.df.agn = self.df.agn.values*ext_smc_mky(self.df.wav.values,norm[5])*norm[0]
                    self.df.e = self.df.e.values*norm[2]
                    self.df.sbc = self.df.sbc.values*norm[3]
                    self.df.im = self.df.im.values*norm[4]
                    self.df['galaxy'] = self.df.im.values+self.df.e.values+self.df.sbc.values                                    
                    self.df['bestfit'] = self.df.im.values+self.df.e.values+self.df.sbc.values+self.df.agn.values
            else:
                if verbose:
                    print('normalized to elip at 1 micron')
        elif type(tmp) is not pd.DataFrame:
            print('tmp should be a dataframe with the first column being wavelengths(micron)')
        elif len(tmp.columns) <2:
            print('tmp should be a dataframe with the first column being wavelengths(micron)')
        else:
            self.df = tmp
            if self.df.columns[0] != 'wav':self.df.rename(columns={self.df.columns[0]:'wav'})
    def nulnu(self,wav_nu,z=None):
        freq = 3e14/self.df.wav
        self.nulnu = np.zeros(len(self.df.columns[1:]),dtype=float)
        if z is not None:
            dist = Distance(z=z).cgs
        else:
            dist=10*u.pc.cgs
        dist = dist.value
        for index, i in enumerate(self.df.columns[1:]):
            self.nulnu[index] = (1e-29*3e14/wav_nu)*4*np.pi*dist**2*np.interp(wav_nu,self.df.wav,self.df.loc[:,i])
    def rest_mag(self,resp=None,z=None,mujy=None,verbose=True):
        '''
        Set z if wants to derive redshifted magnitude
        '''
        if verbose:print('created self.mags')
        if resp is None:
            resp = filter_resp(wav=self.df.wav,all=True).df
        else: 
            if verbose:print('response function should be at the same wavelength array as the templates')
        if mujy is None:
            print('currently only works for mujy = True')
        else:
            self.mags = pd.DataFrame(index = self.df.columns[1:],columns = resp.columns[1:])
            #iterate over the columns
            for index, i in enumerate(self.df.columns[1:]):
                self.mags.loc[i, :] = int_over_resp(self.df.wav,self.df.loc[:,i],resp, z=z).values
    def plot_sed(self,phot = None, fname = None, ignore = None, err = None):
        '''
        make plots using ggplot
        '''
        wav = self.df.wav
        cols = list(self.df.columns[1:])
        if ignore is not None:
            for i in ignore:
                cols.remove(i)
        df_plot = pd.DataFrame({'log wav(um)':np.log10(wav),'log flux':np.log10(self.df.loc[:,cols[0]]),
                                'logf_l':np.log10(self.df.loc[:,cols[0]]),
                                'logf_h':np.log10(self.df.loc[:,cols[0]]),
                                'template':[cols[0] for x in range(len(wav))]})
        for i in cols[1:]:
            df = pd.DataFrame({'log wav(um)':np.log10(wav),'log flux':np.log10(self.df.loc[:,i]),
                'logf_l':np.log10(self.df.loc[:,i]),
                'logf_h':np.log10(self.df.loc[:,i]),
                'template':[i for x in range(len(wav))]})
            df_plot = pd.concat([df_plot,df])
        if phot is None:
            plt_out=ggplot(df_plot,aes(x='log wav(um)',y='log flux',color='template'))+geom_line()
        elif err is None:
            print('No error bars')
            if type(phot) != pd.Series:
                print('phot should be in pandas series')
            else:
                df_phot = pd.DataFrame({'log wav(um)':np.log10(np.asarray([dict_wav[x] for x in phot.index])),
                            'log flux':np.log10(phot.values.astype(float)),
                            'template':['Data' for x in range(len(phot))]})
                self.phot = df_phot                
                plt_out=ggplot(df_phot,aes(x='log wav(um)', y='log flux',color='template'))+\
                        geom_point()+geom_line(df_plot)+\
                        ylim(min(df_phot['log flux'])-1.5,max(df_phot['log flux'])+0.5)+\
                        xlim(-0.7,1.5)
                        #+geom_point(df_phot,size=40,color='red')
        else:
            if type(phot) != pd.Series or type(err) != pd.Series:
                print('phot and err should be in pandas series with band names as index')
            else:
                df_phot = pd.DataFrame({'log wav(um)':np.log10(np.asarray([dict_wav[x] for x in phot.index]).astype(float)),
                            'log flux':np.log10(phot.values.astype(float)),
                            'logf_l':np.log10(phot.values.astype(float)-0.95*err.values.astype(float)),
                            'logf_h':np.log10(phot.values.astype(float)+0.95*err.values.astype(float)),
                            'template':['Data' for x in range(len(phot))]})
            plt_out=ggplot(df_phot,aes(x='log wav(um)',y='log flux',ymax='logf_h',ymin='logf_l',color='template'))+\
            geom_point()+geom_pointrange()+geom_line(df_plot)+\
            ylim(min(df_phot['log flux'])-1.5,max(df_phot['log flux'])+0.5)+\
            xlim(-0.7,1.5)
            self.phot = df_phot
        #if fname is None:
        #    fname = 'plot'
        #ggsave(plt_out,fname+'.pdf')
        self.sed = plt_out


def plot_sed(tmp,phot = None, fname = None, ignore = None, err = None):
    '''
    make plots using ggplot
    '''
    wav = tmp.df.wav
    cols = list(tmp.df.columns[1:])
    if ignore is not None:
        for i in ignore:
            cols.remove(i)
    df_plot = pd.DataFrame({'log wav(um)':np.log10(wav),'log flux':np.log10(tmp.df.loc[:,cols[0]]),'template':[cols[0] for x in range(len(wav))]})
    for i in cols[1:]:
        df = pd.DataFrame({'log wav(um)':np.log10(wav),'log flux':np.log10(tmp.df.loc[:,i]),'template':[i for x in range(len(wav))]})
        df_plot = pd.concat([df_plot,df])
    if phot is None:
        plt_out=ggplot(df_plot,aes(x='log wav(um)',y='log flux',color='template'))+geom_line()
    elif err is None:
        if type(phot) != pd.Series:
            print('phot should be in pandas series')
        else:
            df_phot = ({'log wav(um)':np.log10(np.asarray([dict_wav[x] for x in phot.index])),
                        'log flux':np.log10(phot.values.astype(float)),
                        'template':['Data' for x in range(len(phot))]})
            plt_out=ggplot(df_phot,aes(x='log wav(um)', y='log flux',color='template'))+\
                    geom_point()+geom_line(df_plot)
    else:
        plt_out=ggplot(df_plot,aes(x='log wav(um)',y='log flux',color='template'))+\
        geom_line()+geom_point(data = df_phot)
    #if fname is None:
    #    fname = 'plot'
    #ggsave(plt_out,fname+'.pdf')
    self.sed = plt_out











