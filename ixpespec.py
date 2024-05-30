import os, sys
import ixpeobssim
from astropy.io import fits
import numpy as np
from time import gmtime, strftime
import heasoftpy as hsp
today = strftime("%Y%m%d", gmtime())
from heasoftpy.ftools.quzcif import quzcif
from typing import Literal
from ixpeobssim.binning import polarization
import ixpeobssim.core.pipeline as pipeline
from heasoftpy.ixpe import ixpecalcarf

scriptname = '/Users/cchen24/Desktop/'+today+'.txt'


class ixpefilehandler(object):
    '''
    For generating shell scripts to do spectra extration
    '''
    def __init__(self, datapath='/Users/cchen24/Desktop/ixpesoc/mrk501/',
                 obsid='02004601',scriptname='',rej=False):
        self.ixpeobssimpath = os.path.dirname(ixpeobssim.__file__) + '/'
        self.obsid = obsid
        self.datapath = os.path.abspath(datapath) +'/'
        evtlist = os.listdir(self.datapath + obsid+'/event_l2/')
        hklist =os.listdir(self.datapath + obsid+'/hk/')
        evtfiles = [i for i in evtlist if 'evt2_v' in i]
        attfiles = [i for i in hklist if 'att' in i]
        self.evt = []
        self.att = []
        self.rej = rej
        for d in range(3):
            if self.rej:
                evt = self.datapath + obsid +'/event_l2/'+[i for i in evtfiles if 'det'+str(d+1) in i and 'rej' in i][0]
            else:
                evt = self.datapath + obsid +'/event_l2/'+[i for i in evtfiles if 'det'+str(d+1) in i and 'rej' not in i][0]
            if os.path.isfile(evt):
                self.evt.append(evt)
            else:
                print(evt, "not found!!!!")
            att = self.datapath + obsid +'/hk/'+[i for i in attfiles if 'det'+str(d+1) in i][0]    
            if os.path.isfile(att):
                self.att.append(att)
            else:
                print(att, "not found!!!")
        self.scriptname = scriptname

    
    def xobsselect(self, srcreg, bgreg='',srcpath='', run=False, wait=True):
        from regions import Regions
        scriptname = self.scriptname
        if len(srcpath) == 0:
            srcpath = self.datapath +self.obsid+'/'
        if not os.path.isdir(srcpath):
            os.system(f'mkdir {srcpath}')
        if len(scriptname) > 0:
            with open(scriptname,'w') as f:
                f.write('\n\n#***xobsselect***\n\n')
                t = f'xpselect --regfile {srcreg} {self.evt[0]} {self.evt[1]} {self.evt[2]}\n'
                f.write(t)
                # renaming the source files
                for i in range(3):
                    d = i+1
                    evtstr = self.evt[i].split('.')[0]
                    f.write(f'mv {evtstr}_select.fits {srcpath}ixpe{self.obsid}_src_DU{d}.fits\n')
        if run:
            pipeline.xpselect(*self.evt,regfile=srcreg)
            for i in range(3):
                d = i+1
                evtstr = self.evt[i].split('.')[0]
                os.system(f'mv {evtstr}_select.fits {srcpath}ixpe{self.obsid}_src_DU{d}.fits')
 
        self.srcevt = [f'{srcpath}ixpe{self.obsid}_src_DU1.fits', f'{srcpath}ixpe{self.obsid}_src_DU2.fits',f'{srcpath}ixpe{self.obsid}_src_DU3.fits']
        # assuming single source region
        src_reg = Regions.read(srcreg,format='ds9')[0]
        self.srcscale = src_reg.radius.value**2
        
        print('region filtered events at:' ,self.srcevt)
        if len(bgreg) > 1:
            if len(scriptname) > 0:
            
                with open(scriptname,'a+') as f:
                    t = f'xpselect --regfile {bgreg} {self.evt[0]} {self.evt[1]} {self.evt[2]}\n'
                    f.write(t)
                    # renaming the bg files
                    for i in range(3):
                        d = i+1
                        f.write(f'mv {self.evt[i].split(".")[0]}_select.fits {srcpath}ixpe{self.obsid}_bg_DU{d}.fits\n')
            if run:
                pipeline.xpselect(*self.evt,regfile=bgreg)
                for i in range(3):
                    d = i+1
                    os.system(f'mv {self.evt[i].split(".")[0]}_select.fits {srcpath}ixpe{self.obsid}_bg_DU{d}.fits')
                
            self.bgevt = [f'{srcpath}ixpe{self.obsid}_bg_DU1.fits', f'{srcpath}ixpe{self.obsid}_bg_DU2.fits',f'{srcpath}ixpe{self.obsid}_bg_DU3.fits']
            bg_reg = Regions.read(bgreg,format='ds9')[0]
            # assuming annulus background
            self.bgscale = bg_reg.outer_radius.to(src_reg.radius.unit).value**2 -bg_reg.inner_radius.to(src_reg.radius.unit).value**2
            print('region filtered bg events at:' ,self.bgevt)

        else:
            self.bgevt = []
        if len(scriptname) > 0 and wait:
            input(f'Check {scriptname} for the xobsselect section...\n')
        

    def xpbinlc(self,lcpath='',tbins=4,run=True):
        if run:
            lclist = pipeline.xpbin(*self.srcevt,algorithm='LC',tbins=tbins)
            
        self.lclist = lclist
        #xpbin --algorithm LC --irfname="ixpe:obssim:v13" ixpe01004501_det1_evt2_v06.fits ixpe01004501_det2_evt2_v06.fits ixpe01004501_det3_evt2_v06.fits --tbins 4
    
    def splitevt_lc(self,bg=False):
        binnedevt = []
        binnedevt_bg = []
        for ilc, lc in enumerate(self.lclist):
            hdu = fits.open(lc)
            lcdata = hdu[1].data
            tstop = hdu[1].header['TSTOP']
            tstart = hdu[1].header['TSTART']
            #tmin_abs = np.min(lcdata['TIME'] - 0.5*lcdata['TIMEDEL'])
            #tmax_abs = np.max(lcdata['TIME'] + 0.5*lcdata['TIMEDEL'])
            for i in range(len(lcdata)):
                t = lcdata['TIME'][i]
                tdel = lcdata['TIMEDEL'][i]
                tmin = t - 0.5*tdel
                tmax = t + 0.5*tdel
                if tmax > tstop: tmax = tstop
                if tmin < tstart: tmin = tstart
                blist = pipeline.xpselect(self.srcevt[ilc],tmin=tmin,tmax=tmax,suffix=f"tbin{i}",overwrite=True)
                binnedevt.append(blist)
                if bg:
                    bglist = pipeline.xpselect(self.bgevt[ilc],tmin=tmin,tmax=tmax,suffix=f"tbin{i}",overwrite=True)
                    binnedevt_bg.append(blist)
        binnedevt = list(np.array(binnedevt).flatten())
        self.binnedevt = binnedevt
        if bg:
            binnedevt_bg = list(np.array(binnedevt_bg).flatten())
            self.binnedevt_bg = binnedevt_bg
        for e in binnedevt:
            hdu = fits.open(e)
            gti = hdu[2].data
            evt = hdu[1].data
            evttimemin = np.min(evt['TIME'])
            evttimemax = np.max(evt['TIME'])
            gti = gti[gti['START'] >= evttimemin]
            gti = gti[gti['STOP'] <= evttimemax]
            gti['START'][0] = evttimemin
            gti['STOP'][-1] = evttimemax+1e-5
            newlivetime = np.sum(gti['STOP'] - gti['START'])
            hdu[0].header['LIVETIME'] = newlivetime
            hdu[1].header['LIVETIME'] = newlivetime
            hdu[2].header['LIVETIME'] = newlivetime
            hdu[2].data = gti
            print(f'new livetime={newlivetime}')
            hdu.writeto(e,overwrite=True)
            hdu.close()
        if bg:
            for bevt, evt in zip(binnedevt_bg,binnedevt):
                bhdu = fits.open(bevt)
                hdu = fits.open(evt)
                bhdu[0].header['LIVETIME'] = hdu[0].header['LIVETIME']
                bhdu[1].header['LIVETIME'] = hdu[1].header['LIVETIME']
                bhdu[2].header['LIVETIME'] = hdu[2].header['LIVETIME']
                bhdu[2].data = hdu[2].data
                bhdu.close()
                hdu.close()
            

    def xobspcube(self,pcubepath='',ebins=1,acceptcorr=True,run=False,overwrite=True,wait=True,irfname='ixpe:obssim20240101:v13'):
        ebinstr=f'--ebins {ebins}'
        acceptcorrstr=f'--acceptcorr {acceptcorr}'
        scriptname = self.scriptname
        pcubelist = []
        bgpcubelist = []
        if len(pcubepath) == 0:
            pcubepath = self.datapath + self.obsid + '/pcube/'
        if not os.path.isdir(pcubepath):
            os.system(f'mkdir {pcubepath}')
        if len(scriptname) > 0:
            with open(scriptname,'a+') as f:
                f.write('\n\n#***xobspcube***\n\n')
                t = f'xpbin --algorithm PCUBE --irfname={irfname} {self.srcevt[0]} {self.srcevt[1]} {self.srcevt[2]} {ebinstr} {acceptcorrstr}\n'
                f.write(t)
                for i in range(3):
                    d = i+1
                    f.write(f'mv {self.srcevt[i].split(".")[0]}_pcube.fits {pcubepath}ixpe{self.obsid}_pcube_src_DU{d}.fits\n')
        if run:
            pipeline.xpbin(*self.srcevt, algorithm='PCUBE',irfname=irfname,
                           ebins=ebins,acceptcorr=acceptcorr,overwrite=overwrite)
            for i in range(3):
                d = i+1
                os.system(f'mv {self.srcevt[i].split(".")[0]}_pcube.fits {pcubepath}ixpe{self.obsid}_pcube_src_DU{d}.fits')
        for i in range(3):
            d = i + 1
            pcubelist.append(f'{pcubepath}ixpe{self.obsid}_pcube_src_DU{d}.fits')
        self.pcubelist = pcubelist
        if len(self.bgevt) > 0:
            with open(scriptname,'a+') as f:
                t = f'xpbin --algorithm PCUBE --irfname={irfname} {self.bgevt[0]} {self.bgevt[1]} {self.bgevt[2]} {ebinstr} {acceptcorrstr}\n'
                f.write(t)
                for i in range(3):
                    d = i+1
                    f.write(f'mv {self.bgevt[i].split(".")[0]}_pcube.fits {pcubepath}ixpe{self.obsid}_pcube_bg_DU{d}.fits\n')
            if run:
                pipeline.xpbin(*self.bgevt, algorithm='PCUBE',irfname={irfname},
                               ebins=ebins,acceptcorr=acceptcorr,overwrite=overwrite)
                for i in range(3):
                    d = i+1
                    os.system(f'mv {self.bgevt[i].split(".")[0]}_pcube.fits {pcubepath}ixpe{self.obsid}_pcube_bg_DU{d}.fits')
            for i in range(3):
                d = i + 1
                bgpcubelist.append(f'{pcubepath}ixpe{self.obsid}_pcube_bg_DU{d}.fits')
            self.bgpcubelist = bgpcubelist

        if len(scriptname) > 0 and wait:
            input(f'Check {scriptname} for the xobspcube section...\n')
        
    def xobspha(self,phapath='',mode='weighted',acceptcorr=False,alpha075=True,run=False,wait=True,irfname="ixpe:obssim:v13"):
        
        acceptcorrstr=f"--acceptcorr {acceptcorr}"
        resp_options = {
            'weighted':'alpha075_',
            'simple':'alpha075simple_',
            'unweighted':'20170101_0'
        }
        respstr = resp_options[mode]
        
        weightstr_options = {
            'weighted':'--weights True',
            'unweighted':'--weights False'        
        }    
        weightstr = weightstr_options[mode]
        weightbool = bool(weightstr.split()[-1])
        scriptname = self.scriptname
        if len(phapath) == 0:
            phapath = self.datapath + self.obsid + '/pha/'
        if not os.path.isdir(phapath):
            os.system(f'mkdir {phapath}')
        with open(scriptname,'a+') as f:
            f.write('\n\n#***xobspha:src***\n\n')
            t = f'xpbin --algorithm PHA1 --irfname={irfname} {self.srcevt[0]} {self.srcevt[1]} {self.srcevt[2]} {weightstr} {acceptcorrstr}\n'
            f.write(t)
            t = f'xpbin --algorithm PHA1Q --irfname={irfname} {self.srcevt[0]} {self.srcevt[1]} {self.srcevt[2]} {weightstr} {acceptcorrstr}\n'
            f.write(t)
            t = f'xpbin --algorithm PHA1U --irfname={irfname} {self.srcevt[0]} {self.srcevt[1]} {self.srcevt[2]} {weightstr} {acceptcorrstr}\n'
            f.write(t)
        if run:
            srcspecIlist = pipeline.xpbin(*self.srcevt,algorithm="PHA1",irfname=irfname,acceptcorr=acceptcorr,weights=weightbool)
            srcspecQlist = pipeline.xpbin(*self.srcevt,algorithm="PHA1Q",irfname=irfname,acceptcorr=acceptcorr,weights=weightbool)
            srcspecUlist = pipeline.xpbin(*self.srcevt,algorithm="PHA1U",irfname=irfname,acceptcorr=acceptcorr,weights=weightbool)
            
        self.srcspecI = []
        self.srcspecQ = []
        self.srcspecU = []
        with open(scriptname,'a+') as f:
            for i in range(3):
                d = i+1
                f.write(f'mv {self.srcevt[i].split(".")[0]}_pha1.fits {phapath}ixpe{self.obsid}_src_pha1_DU{d}.fits\n')
                f.write(f'mv {self.srcevt[i].split(".")[0]}_pha1q.fits {phapath}ixpe{self.obsid}_src_pha1q_DU{d}.fits\n')
                f.write(f'mv {self.srcevt[i].split(".")[0]}_pha1u.fits {phapath}ixpe{self.obsid}_src_pha1u_DU{d}.fits\n')
                self.srcspecI.append(f'{phapath}ixpe{self.obsid}_src_pha1_DU{d}.fits')
                self.srcspecQ.append(f'{phapath}ixpe{self.obsid}_src_pha1q_DU{d}.fits')
                self.srcspecU.append(f'{phapath}ixpe{self.obsid}_src_pha1u_DU{d}.fits')


        if len(self.bgevt) > 0:
            self.bgspecI = []
            self.bgspecQ = []
            self.bgspecU = []
            
            with open(scriptname,'a+') as f:
                f.write('\n\n#***xobspha:bg***\n\n')
                t = f'xpbin --algorithm PHA1 --irfname={irfname} {self.bgevt[0]} {self.bgevt[1]} {self.bgevt[2]} {weightstr} {acceptcorrstr}\n'
                f.write(t)
                t = f'xpbin --algorithm PHA1Q --irfname={irfname} {self.bgevt[0]} {self.bgevt[1]} {self.bgevt[2]} {weightstr} {acceptcorrstr}\n'
                f.write(t)
                t = f'xpbin --algorithm PHA1U --irfname={irfname} {self.bgevt[0]} {self.bgevt[1]} {self.bgevt[2]} {weightstr} {acceptcorrstr}\n'
                f.write(t)
                for i in range(3):
                    d = i + 1
                    f.write(f'mv {self.bgevt[i].split(".")[0]}_pha1.fits {phapath}ixpe{self.obsid}_bg_pha1_DU{d}.fits\n')
                    f.write(f'mv {self.bgevt[i].split(".")[0]}_pha1q.fits {phapath}ixpe{self.obsid}_bg_pha1q_DU{d}.fits\n')
                    f.write(f'mv {self.bgevt[i].split(".")[0]}_pha1u.fits {phapath}ixpe{self.obsid}_bg_pha1u_DU{d}.fits\n')
                    self.bgspecI.append(f'{phapath}ixpe{self.obsid}_bg_pha1_DU{d}.fits')
                    self.bgspecQ.append(f'{phapath}ixpe{self.obsid}_bg_pha1q_DU{d}.fits')
                    self.bgspecU.append(f'{phapath}ixpe{self.obsid}_bg_pha1u_DU{d}.fits')
            if run:
                bgspecIlist = pipeline.xpbin(*self.bgevt,algorithm="PHA1",irfname=irfname,acceptcorr=acceptcorr,weights=weightbool)
                bgspecQlist = pipeline.xpbin(*self.bgevt,algorithm="PHA1Q",irfname=irfname,acceptcorr=acceptcorr,weights=weightbool)
                bgspecUlist = pipeline.xpbin(*self.bgevt,algorithm="PHA1U",irfname=irfname,acceptcorr=acceptcorr,weights=weightbool)
        if wait:
            r = input(f'Run the xobspha:src and xobspha:bg sections in {scriptname} before hsp.fthedit...\n')
        if alpha075:
            for i in range(3):
                d = i +1
                hsp.fthedit(infile=self.srcspecI[i]+'+1', keyword='RESPFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/rmf/ixpe_d{d}_obssim_alpha075_v012.rmf',longstring='yes')
                hsp.fthedit(infile=self.srcspecQ[i]+'+1', keyword='RESPFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/rmf/ixpe_d{d}_obssim_alpha075_v012.rmf',longstring='yes')
                hsp.fthedit(infile=self.srcspecU[i]+'+1', keyword='RESPFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/rmf/ixpe_d{d}_obssim_alpha075_v012.rmf',longstring='yes')
                hsp.fthedit(infile=self.srcspecI[i]+'+1', keyword='ANCRFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/arf/ixpe_d{d}_obssim_alpha075_v012.arf',longstring='yes')
                hsp.fthedit(infile=self.srcspecQ[i]+'+1', keyword='ANCRFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/mrf/ixpe_d{d}_obssim_alpha075_v012.mrf',longstring='yes')
                hsp.fthedit(infile=self.srcspecU[i]+'+1', keyword='ANCRFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/mrf/ixpe_d{d}_obssim_alpha075_v012.mrf',longstring='yes')
                print(
                    fits.open(self.srcspecI[i])[1].header['ANCRFILE']
                )
                print(
                    fits.open(self.srcspecI[i])[1].header['RESPFILE']
                )
        self.linkbg(wait=wait)

    def xobssrc(self,ispec=[],qspec=[],uspec=[]):
        '''Manually accept list of source files
        '''
        self.srcspecI = ispec
        self.srcspecQ = qspec
        self.srcspecU = uspec

    def xobsbkg(self,ispec=[],qspec=[],uspec=[]):
        '''Manually accept list of bg files
        '''
        self.bgspecI = ispec
        self.bgspecQ = qspec
        self.bgspecU = uspec

    def xselectpi(self, srcreg, bgreg='',pipath='',mode=Literal['weighted','simple','unweighted'],wait=True,radius=1.0,offarcmin=-1.0,overwrite=True):
        '''
        xselectpi - manually add ANCRFILE and RESPFILE to spectral files
        assuming the
        :return:
        To be updated:

        p = subprocess.Popen('/bin/bash',stdin=subprocess.PIPE,stdout=subprocess.PIPE, text=True)
        c = """
        xselect

        exit
        no

        """
        out, err = p.communicate(c)
        p.kill()
        
        '''
        resp_options = {
            'weighted':'alpha075_',
            'simple':'alpha075simple_',
            'unweighted':'20170101_0'
        }
        respstr = resp_options[mode]
        if len(pipath) == 0:
            pipath = self.datapath + self.obsid + '/pha/'

        # start populating xselect .xco files
        for i, du in enumerate([1,2,3]):
            fn = self.datapath + self.obsid + f'/{self.obsid}DU{du}.xco'
            with open(fn,'w') as f:
                f.write(fn.split('/')[-1]+'\n')
                f.write(f'read event {self.evt[i].split("/")[-1]}\n')
                f.write(f'{self.datapath}{self.obsid}/event_l2\n')
                f.write('yes\n')
                f.write(f'filter region {srcreg}\n')
                f.write('extract spec stokes=NEFF\n')
                f.write(f'save spec {pipath}ixpe{self.obsid}_DU{du}.pi\n')
                if os.path.isfile(f'{pipath}ixpe{self.obsid}_DU{du}I.pi'):
                    if overwrite:
                        f.write('yes\n')
                    else:
                        f.write('no\n')
                if len(bgreg) > 0:
                    f.write(f'clear region\n')
                    f.write(f'filt region {bgreg}\n')
                    f.write('extract spec stokes=NEFF\n')
                    f.write(f'save spec {pipath}ixpe{self.obsid}_bg_DU{du}.pi\n')
                    if os.path.isfile(f'{pipath}ixpe{self.obsid}_bg_DU{du}I.pi'):
                        if overwrite:
                            f.write('yes\n')
                        else:
                            f.write('no\n')
                    f.write('exit\n')
                    f.write('no\n')
                    f.write('\n')
        if wait:
            input(f'\n\n#***run xselect files, e.g., {fn}, or  on event2 files to extract spectra, see should name them as e.g., {pipath}ixpe{self.obsid}_DU1I.pi***\n\n')
        self.srcspecI = []
        self.srcspecQ = []
        self.srcspecU = []
        for i in range(3):
            d = i+1
            self.srcspecI.append(f'{pipath}ixpe{self.obsid}_DU{d}I.pi')
            self.srcspecQ.append(f'{pipath}ixpe{self.obsid}_DU{d}Q.pi')
            self.srcspecU.append(f'{pipath}ixpe{self.obsid}_DU{d}U.pi')
            
        self.linkresp(mode=mode,offarcmin=offarcmin,radius=radius,overwrite=overwrite)
            
        if len(self.bgevt) > 0:
            self.bgspecI = []
            self.bgspecQ = []
            self.bgspecU = []
            for i in range(3):
                d = i+1
                self.bgspecI.append(f'{pipath}ixpe{self.obsid}_bg_DU{d}I.pi')
                self.bgspecQ.append(f'{pipath}ixpe{self.obsid}_bg_DU{d}Q.pi')
                self.bgspecU.append(f'{pipath}ixpe{self.obsid}_bg_DU{d}U.pi')

            # now link the background spectra with source spectra
            self.linkbg()
    def makeresp(self,mode=Literal['weighted','simple','unweighted'],pipath='',radius=1.0,offarcmin=-1.0,overwrite=True):
        if len(pipath) == 0:
            pipath = self.datapath + self.obsid + '/pha/'
        weight_options = {
            'unweighted':0,
            'weighted':1,
            'simple':2,
        }
        weight_int = weight_options[mode]
        ext_options = ['arf', 'mrf']
        '''
        Run ixpecalcarf
        '''
        if overwrite:
            clobberstr = 'yes'
        else:
            clobberstr = 'no'
        self.caldbarf = []
        self.caldbmrf = []

        if self.obsid == '01004501' or self.obsid == '01004601':
            radius = 0.
        else:
            radius = radius
        print("running ixpecalcarf using CALDB")
        for i, du in enumerate(['DU1','DU2','DU3']):
            d = i+1
            arfbase = f"{pipath}ixpe{self.obsid}_{du}."
            arfout = f"{arfbase}arf"
            self.caldbarf.append(arfout)
            mrfout = f"{arfbase}mrf"
            self.caldbmrf.append(mrfout)
            for ext in ext_options:
                arfout = f"{arfbase}{ext}"
                print(d, ext, self.evt[i], self.att[i])
                r = ixpecalcarf(
                    evtfile=self.evt[i],
                    attfile=self.att[i],
                    specfile='none',
                    radius=radius,
                    offarcmin=offarcmin,
                    weight=weight_int,
                    arfout=arfout,
                    clobber=clobberstr
                )
                with open(self.scriptname,'a+') as f:
                    f.write(f"ixpecalcarf evtfile={self.evt[i]} attfile={self.att[i]} specfile='none' radius={radius} weight={weight_int} arfout={arfout} offarcmin={offarcmin} clobber={clobberstr}\n")
                    print(f"ixpecalcarf evtfile={self.evt[i]} attfile={self.att[i]} specfile='none' radius={radius} weight={weight_int} arfout={arfout} offarcmin={offarcmin} clobber={clobberstr}")

    def linkresp(self,source='caldb',mode=Literal['weighted','simple','unweighted'],radius=1.0,overwrite=True,offarcmin=-1.0):
        resp_options = {
            'weighted':'alpha075_',
            'simple':'alpha075simple_',
            'unweighted':'20170101_'
        }
        respstr = resp_options[mode]

        if source == 'caldb':
            self.makeresp(mode=mode,overwrite=overwrite,radius=radius,offarcmin=offarcmin)
            for i, spec in enumerate(self.srcspecI):
                hdr = fits.open(spec)[1].header
                STOKESPR = hdr['STOKESPR']
                detnam = hdr['DETNAM']
                assert STOKESPR == 'Ixx'
                # get the RMF and ARF from caldb
                dateobs = '2023-01-01T12:00:00'
                cal_cnam = 'MATRIX'
                rmflist = quzcif(mission='ixpe',instrument='gpd',
                    codename=cal_cnam,detector=detnam, filter='-',
                    date=dateobs[:10],time=dateobs[11:], expr='-').stdout.split('1\n')
                
                rmffile = [i.strip() for i in rmflist if respstr in i][0]
                hsp.fthedit(infile=spec+'+1', keyword='RESPFILE',operation='add',
                            value=rmffile,longstring='yes')

                arffile = self.caldbarf[i]
                print(arffile)
                hsp.fthedit(infile=spec+'+1', keyword='ANCRFILE',operation='add',
                            value=arffile,longstring='yes')

            for i, spec in enumerate(self.srcspecQ):
                hdr = fits.open(spec)[1].header
                STOKESPR = hdr['STOKESPR']
                detnam = hdr['DETNAM']
                assert STOKESPR == 'xQx'
                # get the RMF and ARF from caldb
                dateobs = '2023-01-01T12:00:00'
                cal_cnam = 'MATRIX'
                rmflist = quzcif(mission='ixpe',instrument='gpd',
                    codename=cal_cnam,detector=detnam, filter='-',
                    date=dateobs[:10],time=dateobs[11:], expr='-').stdout.split('1\n')
                rmffile = [i.strip() for i in rmflist if respstr in i][0]
                hsp.fthedit(infile=spec+'+1', keyword='RESPFILE',operation='add',
                            value=rmffile,longstring='yes')
                
                arffile = self.caldbmrf[i]
                hsp.fthedit(infile=spec+'+1', keyword='ANCRFILE',operation='add',
                            value=arffile,longstring='yes')
                
            for i, spec in enumerate(self.srcspecU):
                hdr = fits.open(spec)[1].header
                STOKESPR = hdr['STOKESPR']
                detnam = hdr['DETNAM']
                assert STOKESPR == 'xxU'
                # get the RMF and ARF from caldb
                cal_cnam = 'MODSPECRESP'
                dateobs = '2023-01-01T12:00:00'
                rmflist = quzcif(mission='ixpe',instrument='gpd',
                    codename='MATRIX',detector=detnam, filter='-',
                    date=dateobs[:10],time=dateobs[11:], expr='-').stdout.split('1\n')
                rmffile = [fn.strip() for fn in rmflist if respstr in fn][0]
                
                hsp.fthedit(infile=spec+'+1', keyword='RESPFILE',operation='add',
                            value=rmffile,longstring='yes')

                arffile = self.caldbmrf[i]
                hsp.fthedit(infile=spec+'+1', keyword='ANCRFILE',operation='add',
                            value=arffile,longstring='yes')
        
        elif source == 'ixpeobssim':
            for i in range(3):
                d = i +1
                hsp.fthedit(infile=self.srcspecI[i]+'+1', keyword='RESPFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/rmf/ixpe_d{d}_obssim_{respstr}v012.rmf',longstring='yes')
                hsp.fthedit(infile=self.srcspecQ[i]+'+1', keyword='RESPFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/rmf/ixpe_d{d}_obssim__{respstr}v012.rmf',longstring='yes')
                hsp.fthedit(infile=self.srcspecU[i]+'+1', keyword='RESPFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/rmf/ixpe_d{d}_obssim__{respstr}v012.rmf',longstring='yes')
                hsp.fthedit(infile=self.srcspecI[i]+'+1', keyword='ANCRFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/arf/ixpe_d{d}_obssim__{respstr}v012.arf',longstring='yes')
                hsp.fthedit(infile=self.srcspecQ[i]+'+1', keyword='ANCRFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/mrf/ixpe_d{d}_obssim__{respstr}v012.mrf',longstring='yes')
                hsp.fthedit(infile=self.srcspecU[i]+'+1', keyword='ANCRFILE',operation='add',
                            value=f'{self.ixpeobssimpath}caldb/ixpe/gpd/cpf/mrf/ixpe_d{d}_obssim__{respstr}v012.mrf',longstring='yes')
                print(
                    fits.open(self.srcspecI[i])[1].header['ANCRFILE']
                )
                print(
                    fits.open(self.srcspecI[i])[1].header['RESPFILE']
                )        

    def linkbg(self,backscal=True,wait=True):
        if wait:
            input('run xobspha or xselectpi first to populate srcspecI/Q/U and bgspecI/Q/U')
        
        for i in range(3):
            # add backscal
            hsp.fthedit(infile=self.srcspecI[i], keyword='backscal',operation='add',value=self.srcscale)
            hsp.fthedit(infile=self.srcspecQ[i], keyword='backscal',operation='add',value=self.srcscale)
            hsp.fthedit(infile=self.srcspecU[i], keyword='backscal',operation='add',value=self.srcscale)
            if len(self.bgevt) > 0:
                hsp.fthedit(infile=self.bgspecI[i], keyword='backscal',operation='add',value=self.bgscale)
                hsp.fthedit(infile=self.bgspecQ[i], keyword='backscal',operation='add',value=self.bgscale)
                hsp.fthedit(infile=self.bgspecU[i], keyword='backscal',operation='add',value=self.bgscale)
                # link background files
                hsp.fthedit(infile=self.srcspecI[i], keyword='BACKFILE',operation='add',value=self.bgspecI[i],longstring='yes')
                hsp.fthedit(infile=self.srcspecQ[i], keyword='BACKFILE',operation='add',value=self.bgspecQ[i],longstring='yes')
                hsp.fthedit(infile=self.srcspecU[i], keyword='BACKFILE',operation='add',value=self.bgspecU[i],longstring='yes')
        
        
    def calcarf(self, replace=False,phapath='',wait=True):
        '''
        Calcarf:
        assuming phapath is the same
        :return:
        '''

        scriptname=self.scriptname
        
        with open(scriptname,'a+') as f:
            f.write('\n\n#***calcarf***\n\n')
            if replace:
                self.newsrcspecI = self.srcspecI
                self.newsrcspecQ = self.srcspecQ
                self.newsrcspecU = self.srcspecU
            else:
                self.newsrcspecI = [i.replace('_DU', '_newarf_DU') for i in self.srcspecI]
                self.newsrcspecQ = [i.replace('_DU', '_newarf_DU') for i in self.srcspecQ]
                self.newsrcspecU = [i.replace('_DU', '_newarf_DU') for i in self.srcspecU]
                for i in range(len(self.srcspecI)):
                    f.write(f'cp {self.srcspecI[i]} {self.newsrcspecI[i]}\n')
                    f.write(f'cp {self.srcspecQ[i]} {self.newsrcspecQ[i]}\n')
                    f.write(f'cp {self.srcspecU[i]} {self.newsrcspecU[i]}\n')
            for i in range(len(self.srcspecI)):
                #f.write(f'python /Users/cchen24/Desktop/ixpesoc/notebooks/ixpecalcarf.py -evtfile {self.evt[i]} -attfile {self.att[i]} -specfile {self.newsrcspecI[i]}\n')
                #f.write(f'python /Users/cchen24/Desktop/ixpesoc/notebooks/ixpecalcarf.py -evtfile {self.evt[i]} -attfile {self.att[i]} -specfile {self.newsrcspecQ[i]}\n')
                #f.write(f'python /Users/cchen24/Desktop/ixpesoc/notebooks/ixpecalcarf.py -evtfile {self.evt[i]} -attfile {self.att[i]} -specfile {self.newsrcspecU[i]}\n')
                f.write(f'ixpecalcarf evtfile={self.evt[i]} attfile={self.att[i]} specfile={self.newsrcspecI[i]} clobber=yes\n')
                f.write(f'ixpecalcarf evtfile={self.evt[i]} attfile={self.att[i]} specfile={self.newsrcspecQ[i]} clobber=yes\n')
                f.write(f'ixpecalcarf evtfile={self.evt[i]} attfile={self.att[i]} specfile={self.newsrcspecU[i]} clobber=yes\n')

        if wait:
            input(f'Check {scriptname} for the calcarf section...\n')
    
    def groupspec(self,icount=25, chan=5):
        '''
        group source spectra ,
        '''
        scriptname=self.scriptname
        input(f'Check {scriptname}, make sure the files exist before grouping, heasoftpy.ftgrouppha will be executed now...\n')
        extstr = self.srcspecI[0].split('.')[-1]
        print(f"extstr={extstr}")
        self.srcIgrp = []
        self.srcQgrp = []
        self.srcUgrp = []
        self.newsrcIgrp = []
        self.newsrcQgrp = []
        self.newsrcUgrp = []
        for i in range(len(self.srcspecI)):
            hsp.ftgrouppha(infile=self.srcspecI[i], outfile=self.srcspecI[i].replace('.'+extstr,f'_grp{icount}.{extstr}'), grouptype='min', groupscale=icount, clobber='yes')
            self.srcIgrp.append(self.srcspecI[i].replace('.'+extstr,f'_grp{icount}.{extstr}'))
        for i in range(len(self.srcspecI)):
            hsp.ftgrouppha(infile=self.srcspecQ[i], outfile=self.srcspecQ[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'), grouptype='const', groupscale=chan, clobber='yes')
            self.srcQgrp.append(self.srcspecQ[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'))
        for i in range(len(self.srcspecI)):
            hsp.ftgrouppha(infile=self.srcspecU[i], outfile=self.srcspecU[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'), grouptype='const', groupscale=chan, clobber='yes')
            self.srcUgrp.append(self.srcspecU[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'))
        try:
            for i in range(len(self.srcspecI)):
                hsp.ftgrouppha(infile=self.newsrcspecI[i], outfile=self.newsrcspecI[i].replace('.'+extstr,f'_grp{icount}.{extstr}'), grouptype='min', groupscale=icount, clobber='yes')
                self.newsrcIgrp.append(self.newsrcspecI[i].replace('.'+extstr,f'_grp{icount}.{extstr}'))
            for i in range(len(self.srcspecI)):
                hsp.ftgrouppha(infile=self.newsrcspecQ[i], outfile=self.newsrcspecQ[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'), grouptype='const', groupscale=chan, clobber='yes')
                self.newsrcQgrp.append(self.newsrcspecQ[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'))
            for i in range(len(self.srcspecI)):
                hsp.ftgrouppha(infile=self.newsrcspecU[i], outfile=self.newsrcspecU[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'), grouptype='const', groupscale=chan, clobber='yes')
                self.newsrcUgrp.append(self.newsrcspecU[i].replace('.'+extstr,f'_g{chan}chan.{extstr}'))
        except:
            print("skipping the grouping of newsrcspec...")

    def xspec(self,srclistI, srclistQ, srclistU,scriptname=''):
        '''
        Just some code to save some typing time...
        :return: xspec commands in a text file
        '''
        if len(scriptname) == 0:
            scriptname=self.scriptname
            fmode = 'a+'
        else:
            fmode = 'w'
        with open(scriptname,fmode) as f:
            f.write('\n\n#***xspec***\n\n')
            tI =  f'data 1:1 {srclistI[0]} 2:2 {srclistI[1]} 3:3 {srclistI[2]} '
            tQ = f'4:4 {srclistQ[0]} 5:5 {srclistQ[1]} 6:6 {srclistQ[2]} '
            tU = f'7:7 {srclistU[0]} 8:8 {srclistU[1]} 9:9 {srclistU[2]}'
            iglist = [f'{i}:{i} **-2. 8.-**' for i in range(1,10)]
            tig = 'ignore '+ ' '.join(iglist)
            print(tI + tQ + tU + '\n')
            print(tig + '\n')
            f.write(tI + tQ + tU + '\n')
            f.write(tig + '\n')