import multiprocessing as mp


def run_fit(coeff_queue, phot_id, models, flux, fluxerr):
    '''
    Run an NNLS for one set of template
    '''
    coeff_queue.put (nnls(models,flux/fluxerr.values))

    
for nuid, row in phot.iterrows():
    if sum(row.notnull()) > 0:
        z=data.loc[nuid,'redshift'].values
        bands = row[row.notnull()].index
        flux = row[row.notnull()]
        fluxerr = err.loc[nuid,bands]
        fluxerr[fluxerr.isnull()]=flux[fluxerr.isnull()] 
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
