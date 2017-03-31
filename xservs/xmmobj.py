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


class xobj:
    '''
    XMM detected source
    '''
    def __init__(self, coor = None, band = None, obsid = None, ecf = None):
    def get_obsid(self,):
    def get_ecf(self):



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
